"""
配置加载模块
============
负责读取 config.yaml，解析环境变量 ${VAR}，提供类型安全的属性访问。
支持运行时修改并持久化回 YAML 文件。

设计要点：
- 单例模式：全局唯一 Config 实例，通过 get_config() 获取
- 环境变量替换：_resolve_env 递归处理字符串/字典/列表中的 ${VAR}
- 文件回写：Web UI 修改配置后调用 Config.update() → Config.save()
- reset_config()：强制重新加载（切换场景/模型后使用）
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml


def _resolve_env(value: Any) -> Any:
    """
    递归解析值中的 ${VAR_NAME} 和 ${VAR_NAME:-default} 环境变量。

    示例：
      "${OPENAI_API_KEY}" → os.environ["OPENAI_API_KEY"]
      "${PORT:-8000}"     → os.environ["PORT"] 或 "8000"

    支持嵌套在 dict/list 中递归替换。
    """
    if isinstance(value, str):
        def replace_env(m: re.Match) -> str:
            expr = m.group(1)
            if ":-" in expr:
                var, default = expr.split(":-", 1)
                return os.environ.get(var.strip(), default.strip())
            return os.environ.get(expr.strip(), "")

        return re.sub(r"\$\{([^}]+)\}", replace_env, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


class Config:
    """
    应用配置类

    持有两份数据：
      _raw  — YAML 原始数据（未解析环境变量），用于保存回文件
      _data — 解析后的数据（环境变量已替换），供运行时使用

    这样设计是为了保存时保留 ${ENV_VAR} 占位符，不把密钥明文写死。
    """

    def __init__(self, path: str = "config.yaml"):
        self._path = path
        # 读取原始 YAML（保留 ${VAR} 占位符）
        self._raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        # 生成运行时使用的已解析版本
        self._data = _resolve_env(self._raw)

    def reload(self):
        """从磁盘重新加载配置（用于文件被外部修改后刷新）"""
        self._raw = yaml.safe_load(Path(self._path).read_text(encoding="utf-8"))
        self._data = _resolve_env(self._raw)

    # ---- 类型安全的属性访问 ----
    # 每个属性从 _data 读取（环境变量已解析），带默认值保护

    @property
    def model(self) -> dict:
        return self._data["model"]

    @property
    def api_docs_urls(self) -> list[str]:
        return self._data["api_docs"]["urls"]

    @property
    def api_docs_timeout(self) -> int:
        return self._data["api_docs"].get("timeout", 30)

    @property
    def api_auth(self) -> dict:
        # 后端 API 认证配置
        return self._data.get("api_auth", {"type": "none"})

    @property
    def auto_login(self) -> dict:
        """自动登录配置（项目通用：可配置 header 名称、提示文字、开关）"""
        defaults = {"enabled": True, "header_name": "X-Dts-Admin-Token", "login_hint": "手机号"}
        al = self._data.get("auto_login", {})
        return {**defaults, **al} if isinstance(al, dict) else defaults

    @property
    def project_dir(self) -> str:
        """项目源码目录（用于错误分析时读取代码）"""
        return self._data.get("project_dir", "") or ""

    @property
    def global_params(self) -> list[dict]:
        """全局参数：每个 API 请求都会附加这些 header/query"""
        gp = self._data.get("global_params")
        return gp if isinstance(gp, list) else []

    @property
    def api_scenarios(self) -> dict:
        """请求场景配置（多环境切换）"""
        return self._data.get("api_scenarios", {"active": "default", "list": []})

    def get_active_scenario_mapping(self) -> dict[str, str]:
        """
        获取当前激活场景的 URL 映射表。

        返回格式：{"源地址前缀": "目标地址前缀"}
        例如：{"http://192.168.10.112:7928": "https://api.prod.example.com"}

        api_executor 会用它替换请求的 base_url。
        """
        scenarios = self.api_scenarios
        active_name = scenarios.get("active", "default")
        for s in scenarios.get("list", []):
            if s.get("name") == active_name:
                return s.get("mapping", {})
        return {}

    @property
    def server(self) -> dict:
        return self._data.get("server", {"host": "0.0.0.0", "port": 8000})

    def raw_data(self) -> dict:
        """返回已解析的完整配置（供 Web API 读取）"""
        return self._data

    def save(self):
        """
        将当前 _raw 写回 YAML 文件。

        注意：写的是 _raw（保留 ${VAR} 占位符），而非已解析的 _data。
        这样密钥等敏感值不会以明文写入文件。
        """
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(self._raw, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        self.reload()

    def update(self, section: str, value: Any):
        """
        运行时修改配置并立即持久化。

        用法：
          config.update("global_params", [{"name": "X-Token", "value": "...", "type": "header"}])
          config.update("api_scenarios", {"active": "production", "list": [...]})

        修改 _raw（保持原始值），然后 save 到磁盘。
        """
        self._raw[section] = value
        self.save()


# ============================================================
# 单例管理
# ============================================================
# 整个应用共享一个 Config 实例，避免重复读取和解析文件。

_config_instance: Config | None = None


def get_config(path: str = "config.yaml") -> Config:
    """获取全局唯一 Config 实例（懒加载）"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(path)
    return _config_instance


def reset_config():
    """
    强制重置配置实例。
    用于场景切换/模型更换后需要重新读取配置的场景。
    """
    global _config_instance
    _config_instance = None
