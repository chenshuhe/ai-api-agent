"""
API 端点 → LLM 工具定义 转换器
================================
将内部 Endpoint 数据结构转换为 LLM 可理解的 function-calling tool 格式。

支持两种输出格式：
  - OpenAI tool 格式（兼容 Ollama、DeepSeek 等 OpenAI 兼容接口）
  - Anthropic tool 格式

核心挑战：工具名唯一性
  不同服务的 OpenAPI 文档可能用相同的 operationId。
  例如 game 服务和 prostore 服务都有 "update" 端点。
  去重策略：
    1. 统计重名
    2. 为重名的端点追加服务标识（tag 名或 URL 域名）
    3. 仍冲突则追加数字后缀
    4. 截断到 64 字符（Anthropic 限制）

ASCII 限制：
  OpenAI 要求工具名匹配 ^[a-zA-Z0-9_-]+$
  中文 tag 名会被转换为下划线序列。
"""

from collections import Counter
from urllib.parse import urlparse

from src.api_docs import Endpoint

MAX_TOOL_NAME_LEN = 64  # Anthropic 工具名长度上限


def _sanitize(name: str) -> str:
    """
    净化字符串：只保留 ASCII 字母、数字、下划线和连字符。
    中文等非 ASCII 字符被替换为下划线。

    这是必须的——OpenAI 的 tool.name 字段只接受 [a-zA-Z0-9_-]+
    """
    result = []
    for c in name:
        if c in "_-" or ("a" <= c <= "z") or ("A" <= c <= "Z") or ("0" <= c <= "9"):
            result.append(c)
        else:
            result.append("_")
    return "".join(result)


def _base_name(ep: Endpoint) -> str:
    """
    从端点生成候选工具名。

    优先使用 operationId（语义化），其次用 method + path 拼接。
    示例：
      operationId="getUserById"      → "getUserById"
      无 operationId, POST /api/xxx → "post_api_xxx"
    """
    if ep.operation_id:
        return ep.operation_id.replace("-", "_").replace(" ", "_")
    parts = ep.path.strip("/").split("/")
    parts = [p for p in parts if p and not p.startswith("{")]
    return f"{ep.method.lower()}_{'_'.join(parts)}"


def _short_service(ep: Endpoint) -> str:
    """
    提取短服务标识符，用于区分不同服务的同名端点。

    优先级：tags[0] > base_url 域名
    示例：
      端点 tag="game-items-controller" → "game_items_controller"
      无 tag, base_url="http://192.168.10.112:7928" → "192_168_10_112"
    """
    if ep.tags:
        return ep.tags[0].lower().replace("-", "_").replace(" ", "_")
    if ep.base_url:
        host = urlparse(ep.base_url).netloc.split(":")[0]
        return host.replace(".", "_")
    return "svc"


def _deduplicate_names(endpoints: list[Endpoint]) -> list[str]:
    """
    为所有端点生成唯一的工具名列表。

    算法：
      1. 先用 _base_name 生成候选名
      2. 用 Counter 统计重名
      3. 重名的端点追加 _short_service
      4. 仍冲突的追加数字后缀 _2, _3...
      5. 截断到 MAX_TOOL_NAME_LEN
    """
    counts: Counter[str] = Counter()

    # 第一遍：统计每个候选名出现次数
    for ep in endpoints:
        base = _sanitize(_base_name(ep))
        counts[base] += 1

    seen: Counter[str] = Counter()
    names = []

    # 第二遍：为每个端点生成唯一名
    for ep in endpoints:
        base = _sanitize(_base_name(ep))
        if counts[base] > 1:
            # 重名 → 追加服务标识
            svc = _sanitize(_short_service(ep))
            candidate = f"{base}_{svc}"
        else:
            candidate = base

        seen[candidate] += 1
        if seen[candidate] > 1:
            # 仍冲突 → 数字后缀
            candidate = f"{candidate}_{seen[candidate]}"

        # 截断到最大长度
        if len(candidate) > MAX_TOOL_NAME_LEN:
            candidate = candidate[:MAX_TOOL_NAME_LEN - 4] + "_etc"

        names.append(candidate)

    return names


def _json_schema_type(param_type: str) -> str:
    """OpenAPI 类型 → JSON Schema 类型映射"""
    mapping = {"integer": "integer", "number": "number", "boolean": "boolean", "string": "string"}
    return mapping.get(param_type, "string")


def _build_properties(ep: Endpoint) -> tuple[dict, list[str]]:
    """
    从 Endpoint 构建 JSON Schema properties 和 required 数组。

    处理三种参数来源：
      1. path/query 参数 → 展开为 properties
      2. requestBody → 解析 JSON Schema properties
      3. 自由格式 body → 字符串类型的 body 字段
    """
    properties = {}
    required_params = []

    # 1) URL / Query 参数
    for p in ep.parameters:
        if p.location in ("header", "cookie"):
            continue  # 跳过 header/cookie 参数（由全局参数或认证模块处理）
        prop_def = {
            "type": _json_schema_type(p.param_type),
            "description": p.description or f"{p.param_type} value for {p.name}",
        }
        if p.enum:
            prop_def["enum"] = p.enum
        properties[p.name] = prop_def
        if p.required:
            required_params.append(p.name)

    # 2) 请求体（结构化 JSON Schema）
    if ep.request_body and ep.request_body.schema:
        body_schema = ep.request_body.schema
        if "properties" in body_schema:
            for key, val in body_schema["properties"].items():
                properties[key] = {
                    "type": _json_schema_type(val.get("type", "string")),
                    "description": val.get("description", f"Field: {key}"),
                }
            if body_schema.get("required"):
                required_params.extend(body_schema["required"])
        else:
            # 3) 自由格式 body（$ref 或无 properties）
            ref = body_schema.get("$ref", "")
            type_info = ref.rsplit("/", 1)[-1] if ref else body_schema.get("type", "object")
            properties["body"] = {
                "type": "string",
                "description": f"Request body as JSON string (schema: {type_info})",
            }

    return properties, required_params


def _build_description(ep: Endpoint) -> str:
    """构建工具描述文本（AI 依赖它理解何时调用该工具）"""
    desc = ep.summary or ep.description or f"Call the {ep.method} {ep.path} API"
    if ep.base_url:
        desc += f"\nBase URL: {ep.base_url}"
    desc += f"\nHTTP Method: {ep.method}"
    desc += f"\nPath: {ep.path}"
    return desc[:1024]  # OpenAI 限制


# ---- 对外接口 ----

def convert_all_to_openai(endpoints: list[Endpoint]) -> list[dict]:
    """
    将所有端点转换为 OpenAI function-calling tool 格式。

    返回格式：
      [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
    """
    tool_names = _deduplicate_names(endpoints)
    tools = []
    for ep, name in zip(endpoints, tool_names):
        properties, required = _build_properties(ep)
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": _build_description(ep),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return tools


def convert_all_to_anthropic(endpoints: list[Endpoint]) -> list[dict]:
    """
    将所有端点转换为 Anthropic tool 格式。

    返回格式：
      [{"name": "...", "description": "...", "input_schema": {"type": "object", ...}}]
    """
    tool_names = _deduplicate_names(endpoints)
    tools = []
    for ep, name in zip(endpoints, tool_names):
        properties, required = _build_properties(ep)
        tools.append({
            "name": name,
            "description": _build_description(ep),
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })
    return tools


def build_tool_index(endpoints: list[Endpoint]) -> dict[str, Endpoint]:
    """
    构建 工具名 → Endpoint 的映射表。

    当 AI 返回 tool_call 时，agent 通过 tool_index 查找对应的 Endpoint，
    然后调用 api_executor 发送 HTTP 请求。
    """
    tool_names = _deduplicate_names(endpoints)
    return dict(zip(tool_names, endpoints))
