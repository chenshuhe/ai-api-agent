"""
OpenAPI 文档解析模块
====================
核心职责：从多个 URL 抓取 OpenAPI 3.x JSON 文档，解析为统一的内部数据结构。

数据流：
  配置文件 URL 列表
    → fetch_api_docs() 并发 HTTP GET
    → parse_openapi() 遍历 paths 提取 Endpoint
    → get_all_endpoints() 合并所有文档的端点

设计要点：
- 并发抓取：asyncio.gather 同时请求多个文档源
- 容错：单个文档失败不影响其他文档的解析
- Schema 简化：_schema_to_str() 将 JSON Schema 转为人类可读字符串
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx


# ---- 数据模型 ----

@dataclass
class Parameter:
    """API 参数（路径参数 / 查询参数 / 请求体字段）"""
    name: str                    # 参数名
    location: str                # path=路径参数, query=查询参数, header=请求头
    required: bool               # 是否必填
    param_type: str              # JSON Schema 类型：string / integer / boolean / ...
    description: str
    example: Any = None
    enum: list[str] | None = None
    default: Any = None


@dataclass
class RequestBody:
    """请求体描述"""
    content_type: str            # 通常是 application/json
    schema: dict                 # JSON Schema 定义
    required: bool
    description: str


@dataclass
class Endpoint:
    """
    单个 API 端点的完整描述。
    这是内部核心数据结构——后续 tool_converter 将其转为 LLM 工具定义。
    """
    method: str                  # GET / POST / PUT / DELETE / PATCH
    path: str                    # /api/game/v3/games/{id}
    summary: str                 # 简短描述（OpenAPI summary 字段）
    description: str             # 完整描述（含参数/返回值信息）
    tags: list[str] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)
    request_body: RequestBody | None = None
    response_schema: dict | None = None
    base_url: str = ""           # 来源 OpenAPI 文档的 servers[0].url
    operation_id: str = ""       # OpenAPI operationId（用于生成唯一工具名）


@dataclass
class ApiDoc:
    """一个完整的 OpenAPI 文档解析结果"""
    title: str
    version: str
    description: str
    base_url: str                # 文档声明的服务器地址
    endpoints: list[Endpoint] = field(default_factory=list)


# ---- 解析辅助函数 ----

def _extract_base_url(openapi: dict) -> str:
    """从 OpenAPI 文档的 servers 数组提取基础 URL"""
    servers = openapi.get("servers", [])
    if servers:
        return servers[0].get("url", "").rstrip("/")
    return ""


def _schema_to_str(schema: dict, depth: int = 0) -> str:
    """
    将 JSON Schema 片段转为可读字符串。

    用于生成 AI 工具的 description，帮助 AI 理解参数类型。
    深度限制防止递归引用导致无限展开。

    示例：
      {"type": "array", "items": {"type": "string"}} → "array[string]"
      {"type": "object", "properties": {"name": {"type": "string"}}} → "{name:string}"
    """
    if not schema:
        return "any"
    if depth > 3:
        return "object"

    ref = schema.get("$ref", "")
    if ref:
        return ref.rsplit("/", 1)[-1]

    stype = schema.get("type", "any")
    if stype == "array":
        items = schema.get("items", {})
        return f"array[{_schema_to_str(items, depth + 1)}]"
    if stype == "object":
        props = schema.get("properties", {})
        if not props:
            return "object"
        parts = [f"{k}:{_schema_to_str(v, depth + 1)}" for k, v in list(props.items())[:10]]
        return "{" + ", ".join(parts) + "}"
    if stype == "string":
        enums = schema.get("enum")
        if enums:
            return f"string(enum:{','.join(str(e) for e in enums)})"
        return "string"
    return stype


def _resolve_ref(ref: str, openapi: dict) -> dict:
    """解析 OpenAPI $ref 引用，返回引用的 schema 定义。"""
    if not ref.startswith("#/"):
        return {}
    parts = ref[2:].split("/")  # ["components", "schemas", "StoreCourseLesson"]
    node = openapi
    for part in parts:
        node = node.get(part, {})
    return node if isinstance(node, dict) else {}


def _parse_parameters(params_spec: list[dict], openapi: dict | None = None) -> list[Parameter]:
    """
    解析 OpenAPI parameters 数组为 Parameter 对象列表。

    处理两种 Spring @ModelAttribute 模式：
      模式1 — 内联 object：
        {"name":"obj","in":"query","schema":{"type":"object","properties":{...}}}
      模式2 — $ref 引用（SpringDoc 常见）：
        {"name":"storeCourseLesson","in":"query","schema":{"$ref":"#/components/schemas/StoreCourseLesson"}}

    两种模式都展平为独立 query 参数，才能正确发送：
      GET /api?id=1&searchPhone=xxx   ✓
    而不是：
      GET /api?storeCourseLesson={"id":1}  ✗
    """
    if openapi is None:
        openapi = {}

    params = []
    for p in params_spec:
        schema = p.get("schema", {})
        location = p.get("in", "query")
        required = p.get("required", False)

        # 解析 $ref 引用（模式2）
        ref = schema.get("$ref", "")
        resolved_schema = schema
        if ref and location == "query":
            resolved_schema = _resolve_ref(ref, openapi)

        # 检测 @ModelAttribute 模式：query + 有 properties（type 可能缺失但 properties 存在即为 object）
        if location == "query" and "properties" in resolved_schema:
            props = resolved_schema["properties"]
            obj_required = resolved_schema.get("required") or []
            for prop_name, prop_schema in props.items():
                prop_ref = prop_schema.get("$ref", "")
                if prop_ref:
                    prop_type = _resolve_ref(prop_ref, openapi).get("type", "string")
                else:
                    prop_type = prop_schema.get("type", "string")
                params.append(Parameter(
                    name=prop_name,
                    location="query",
                    required=prop_name in obj_required,  # 只有 schema 中显式标记 required 的字段才是必填
                    param_type=prop_type,
                    description=prop_schema.get("description", ""),
                    example=prop_schema.get("example"),
                    enum=prop_schema.get("enum"),
                ))
        else:
            params.append(Parameter(
                name=p.get("name", ""),
                location=location,
                required=required,
                param_type=schema.get("type", "string"),
                description=p.get("description", ""),
                example=p.get("example"),
                enum=schema.get("enum"),
                default=schema.get("default"),
            ))
    return params


def _parse_request_body(body_spec: dict) -> RequestBody | None:
    """解析 OpenAPI requestBody 为 RequestBody 对象"""
    if not body_spec:
        return None
    content = body_spec.get("content", {})
    # 优先取 JSON content-type，兼容通配符
    json_content = content.get("application/json") or content.get("*/*")
    if not json_content:
        return None
    return RequestBody(
        content_type="application/json",
        schema=json_content.get("schema", {}),
        required=body_spec.get("required", False),
        description=body_spec.get("description", ""),
    )


# ---- 核心解析函数 ----

def parse_openapi(openapi: dict, source_url: str = "") -> ApiDoc:
    """
    将 OpenAPI 3.x JSON 字典解析为 ApiDoc 对象。

    遍历 openapi.paths 下的每个 HTTP 方法，提取：
    - 路径参数 / 查询参数 / 请求头
    - 请求体 Schema
    - 200 响应 Schema
    - 生成供 AI 阅读的 description 文本

    Args:
        openapi: OpenAPI JSON 字典
        source_url: 文档来源 URL（仅用于日志）
    """
    info = openapi.get("info", {})
    base_url = _extract_base_url(openapi)
    doc = ApiDoc(
        title=info.get("title", "Untitled"),
        version=info.get("version", "unknown"),
        description=info.get("description", ""),
        base_url=base_url,
    )

    paths = openapi.get("paths", {})
    for path_pattern, methods in paths.items():
        # 遍历每个 HTTP 方法（只处理标准 REST 方法）
        for method_name in ("get", "post", "put", "delete", "patch", "options"):
            operation = methods.get(method_name)
            if not operation:
                continue

            # ---- 解析各部分 ----
            params = _parse_parameters(operation.get("parameters", []), openapi)
            body = _parse_request_body(operation.get("requestBody"))
            op_id = operation.get("operationId", "")

            # 提取 200/201 响应的 Schema
            responses = operation.get("responses", {})
            resp200 = responses.get("200") or responses.get("201")
            resp_schema = None
            if resp200 and "content" in resp200:
                ct = resp200["content"].get("application/json", {})
                resp_schema = ct.get("schema")

            # ---- 构建详细描述（帮助 AI 理解接口）----
            desc_parts = [p for p in [operation.get("summary", ""), operation.get("description", "")] if p]
            description = ". ".join(desc_parts)

            detail_parts = []
            for p in params:
                req = "required" if p.required else "optional"
                detail_parts.append(f"  {p.location} param {p.name} ({p.param_type}, {req}): {p.description}")
            if body:
                detail_parts.append(f"  request body ({body.content_type}): {_schema_to_str(body.schema)}")
            if resp_schema:
                detail_parts.append(f"  response 200: {_schema_to_str(resp_schema)}")
            if detail_parts:
                description += "\nParameters:\n" + "\n".join(detail_parts)

            doc.endpoints.append(Endpoint(
                method=method_name.upper(),
                path=path_pattern,
                summary=operation.get("summary", ""),
                description=description.strip(),
                tags=operation.get("tags", []),
                parameters=params,
                request_body=body,
                response_schema=resp_schema,
                base_url=base_url,
                operation_id=op_id,
            ))

    return doc


# ---- 网络层 ----

async def fetch_api_docs(url: str, timeout: int = 30) -> dict:
    """从指定 URL 抓取 OpenAPI JSON 文档（异步 HTTP GET）"""
    async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


async def load_all_api_docs(urls: list[str], timeout: int = 30) -> list[ApiDoc]:
    """
    并发抓取多个 OpenAPI 文档并解析。

    使用 asyncio.gather 同时请求所有 URL，失败不阻塞其他。
    单个文档抓取/解析失败时打印警告并继续处理其余文档。
    """
    # 并发请求所有文档（return_exceptions=True 防止一个失败导致全部取消）
    results: list[dict] = list(await asyncio.gather(
        *[fetch_api_docs(url, timeout) for url in urls],
        return_exceptions=True,
    ))

    docs = []
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            print(f"[WARN] 获取 API 文档失败 {url}: {result}")
            continue
        try:
            doc = parse_openapi(result, url)
            docs.append(doc)
            print(f"[OK] 从 {url} 加载 {len(doc.endpoints)} 个端点 ({doc.title})")
        except Exception as e:
            print(f"[WARN] 解析 API 文档失败 {url}: {e}")
    return docs


def get_all_endpoints(docs: list[ApiDoc]) -> list[Endpoint]:
    """合并多个 ApiDoc 的所有端点到一个扁平列表"""
    endpoints = []
    for doc in docs:
        for ep in doc.endpoints:
            # 如果端点没有自己的 base_url（罕见），继承文档的
            if not ep.base_url:
                ep.base_url = doc.base_url
            endpoints.append(ep)
    return endpoints
