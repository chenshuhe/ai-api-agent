"""
API 执行器
==========
接收 AI 决定的工具调用，构造 HTTP 请求，发送到后端接口，返回结果。

职责：
  1. 根据 tool_name 查找对应的 Endpoint
  2. 解析路径参数（/api/{id} → /api/123）
  3. 分配参数到 query / body / header
  4. 应用认证（Bearer / Basic / API Key）
  5. 应用全局参数（header / query）
  6. 应用场景 URL 映射（环境切换）
  7. 发送 HTTP 请求，返回 JSON 或文本

参数路由逻辑：
  AI 传入的 arguments 是一个扁平字典，需要按参数位置路由：
    - path 参数 → 替换 URL 中的 {param}
    - query 参数 → ?key=value
    - body 参数 → JSON 请求体字段
    - header 参数 → 由全局参数/认证模块处理，不在此路由
"""

import json
import re
from urllib.parse import urljoin

import httpx

from src.api_docs import Endpoint
from src.config import Config


def _resolve_path(path_pattern: str, arguments: dict) -> str:
    """
    替换路径中的 {param} 占位符。

    示例：
      path="/api/game/v3/games/{id}", arguments={"id": "123"}
      → "/api/game/v3/games/123"
    """
    def replace(m: re.Match) -> str:
        name = m.group(1)
        val = arguments.get(name, m.group(0))
        return str(val)
    return re.sub(r"\{(\w+)\}", replace, path_pattern)


def _build_request(
    ep: Endpoint,
    arguments: dict,
    auth: dict,
    global_params: list[dict] | None = None,
    url_mapping: dict[str, str] | None = None,
) -> dict:
    """
    根据 Endpoint 和参数构建 httpx 请求字典。

    Args:
        ep: API 端点定义
        arguments: AI 传来的参数字典
        auth: 认证配置
        global_params: 全局参数列表（当前对话的）
        url_mapping: 场景 URL 映射（环境切换用）

    Returns:
        httpx 兼容的请求参数：{"url": "...", "method": "...", "headers": {...}, ...}
    """
    # ---- 1. 解析 URL（含场景映射）----
    path = _resolve_path(ep.path, arguments)
    base = ep.base_url

    # 场景 URL 映射：将文档中的 base_url 替换为目标地址
    # 例如 "http://192.168.10.112:7928" → "https://api.prod.example.com"
    if url_mapping and base:
        for src, dst in url_mapping.items():
            if base == src or base.startswith(src):
                base = dst + base[len(src):]
                break

    url = urljoin(base + "/" if base else "", path.lstrip("/"))

    # ---- 2. 参数分类：path / query / body ----
    # 从 Endpoint 中获取各类型参数名集合
    path_param_names = {p.name for p in ep.parameters if p.location == "path"}
    query_param_names = {p.name for p in ep.parameters if p.location == "query"}
    body_param_names = set()
    if ep.request_body and ep.request_body.schema:
        props = ep.request_body.schema.get("properties", {})
        body_param_names = set(props.keys())

    params = {}
    json_body = {}
    for key, val in arguments.items():
        if key in path_param_names:
            continue  # 已在 URL 中处理
        if key in query_param_names:
            params[key] = val
        elif key in body_param_names:
            json_body[key] = val
        elif ep.request_body and not body_param_names:
            # 自由格式 body：尝试解析 JSON
            if isinstance(val, str):
                try:
                    json_body = json.loads(val)
                except json.JSONDecodeError:
                    json_body = {"body": val}
            else:
                json_body = val
        elif key not in path_param_names:
            # 未识别的参数 → 放到 query 中
            params[key] = val

    # ---- 3. 认证 ----
    headers = {}
    auth_type = auth.get("type", "none")
    if auth_type == "bearer":
        token = auth.get("token", "")
        if token:  # 仅在有值时才添加（空 token = 不认证）
            headers["Authorization"] = f"Bearer {token}"
    elif auth_type == "basic":
        import base64
        user = auth.get("username", "")
        pwd = auth.get("password", "")
        if user or pwd:
            creds = base64.b64encode(f"{user}:{pwd}".encode()).decode()
            headers["Authorization"] = f"Basic {creds}"
    elif auth_type == "api_key":
        key_name = auth.get("key_name", "X-API-Key")
        token = auth.get("token", "")
        if token:
            headers[key_name] = token

    # ---- 4. 全局参数（当前对话的额外 header/query）----
    if global_params:
        for gp in global_params:
            ptype = gp.get("type", "header")
            pname = gp.get("name", "")
            pvalue = gp.get("value", "")
            if not pname or not pvalue:
                continue
            if ptype == "header":
                headers[pname] = pvalue
            elif ptype == "query":
                params[pname] = pvalue

    # ---- 5. 组装返回 ----
    req_kwargs = {"url": url, "method": ep.method, "headers": headers}
    if params:
        req_kwargs["params"] = params
    if json_body:
        req_kwargs["json"] = json_body

    return req_kwargs


async def execute_tool_call(
    tool_name: str,
    arguments: dict,
    tool_index: dict[str, Endpoint],
    config: Config,
    timeout: int = 30,
    override_params: list[dict] | None = None,
    url_mapping: dict[str, str] | None = None,
) -> str:
    """
    执行一次工具调用（即一次 HTTP API 请求）。

    Args:
        tool_name: AI 选择的工具名
        arguments: AI 传入的参数
        tool_index: 工具名 → Endpoint 映射表
        config: 全局配置
        timeout: HTTP 超时（秒）
        override_params: 当前对话的全局参数（优先于 config.global_params）
        url_mapping: 场景 URL 映射

    Returns:
        JSON 字符串：成功时是 API 响应，失败时是 {"error": "..."}
    """
    # 查找端点定义
    ep = tool_index.get(tool_name)
    if not ep:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # 确定全局参数和 URL 映射
    gps = override_params if override_params is not None else config.global_params
    mapping = url_mapping if url_mapping is not None else config.get_active_scenario_mapping()

    # 构建请求
    req_kwargs = _build_request(ep, arguments, config.api_auth, gps, url_mapping=mapping)

    # 发送 HTTP 请求
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            resp = await client.request(**req_kwargs)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                return json.dumps(resp.json(), ensure_ascii=False, indent=2)
            return resp.text[:8000]  # 截断过长响应
    except httpx.HTTPStatusError as e:
        # HTTP 错误（4xx/5xx）→ 返回详细信息给 AI 分析
        return json.dumps({
            "error": f"HTTP {e.response.status_code}",
            "detail": e.response.text[:2000],
            "url": str(e.request.url),
        })
    except Exception as e:
        # 网络错误等
        return json.dumps({"error": str(e)})
