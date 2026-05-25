"""HTTP API call executor - builds and sends requests based on tool calls."""

import json
import re
from urllib.parse import urljoin

import httpx
from loguru import logger

from ..api_loader.parser import Endpoint
from ..settings import Settings


def execute(
    tool_name: str,
    arguments: dict,
    tool_index: dict[str, Endpoint],
    settings: Settings,
    override_params: list[dict] | None = None,
    timeout: int = 30,
) -> str:
    """Execute a tool call: build HTTP request, send, return response."""
    import asyncio
    return asyncio.run(_execute_async(tool_name, arguments, tool_index, settings, override_params, timeout))


async def execute_async(
    tool_name: str,
    arguments: dict,
    tool_index: dict[str, Endpoint],
    settings: Settings,
    override_params: list[dict] | None = None,
    timeout: int = 30,
) -> str:
    """Async version of execute."""
    ep = tool_index.get(tool_name)
    if not ep:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    req_kwargs = _build_request(ep, arguments, settings, override_params)

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            resp = await client.request(**req_kwargs)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                return json.dumps(resp.json(), ensure_ascii=False, indent=2)
            return resp.text[:8000]
    except httpx.HTTPStatusError as e:
        return json.dumps({
            "error": f"HTTP {e.response.status_code}",
            "detail": e.response.text[:2000],
            "url": str(e.request.url),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def _build_request(ep: Endpoint, arguments: dict, settings: Settings,
                   override_params: list[dict] | None = None) -> dict:
    """Build httpx-compatible request dict from endpoint + arguments."""
    path = _resolve_path(ep.path, arguments)
    base = ep.base_url

    # Apply scenario URL mapping
    mapping = settings.api_scenarios.get_active_mapping()
    if mapping and base:
        for src, dst in mapping.items():
            if base.startswith(src):
                base = dst + base[len(src):]
                break

    url = urljoin(base + "/" if base else "", path.lstrip("/"))

    # Classify params
    path_names = {p.name for p in ep.parameters if p.location == "path"}
    query_names = {p.name for p in ep.parameters if p.location == "query"}
    body_names = set()
    if ep.request_body and ep.request_body.schema:
        body_names = set(ep.request_body.schema.get("properties", {}).keys())

    query_params = {}
    json_body = {}
    for key, val in arguments.items():
        if key in path_names:
            continue
        if key in query_names:
            query_params[key] = val
        elif key in body_names:
            json_body[key] = val
        elif ep.request_body and not body_names:
            if isinstance(val, str):
                try:
                    json_body = json.loads(val)
                except json.JSONDecodeError:
                    json_body = {"body": val}
            else:
                json_body = val
        elif key not in path_names:
            query_params[key] = val

    # Auth
    headers = {}
    auth = settings.api_auth
    if auth.type == "bearer" and auth.token:
        headers["Authorization"] = f"Bearer {auth.token}"
    elif auth.type == "basic" and (auth.username or auth.password):
        import base64
        creds = base64.b64encode(f"{auth.username}:{auth.password}".encode()).decode()
        headers["Authorization"] = f"Basic {creds}"
    elif auth.type == "api_key" and auth.token:
        headers[auth.key_name] = auth.token

    # Global params
    gps = override_params if override_params is not None else [p.model_dump() for p in settings.global_params]
    for gp in gps:
        pname = gp.get("name", "")
        pvalue = gp.get("value", "")
        if not pname or not pvalue:
            continue
        if gp.get("type") == "header":
            headers[pname] = pvalue
        elif gp.get("type") == "query":
            query_params[pname] = pvalue

    req = {"url": url, "method": ep.method, "headers": headers}
    if query_params:
        req["params"] = query_params
    if json_body:
        req["json"] = json_body
    return req


def _resolve_path(path_pattern: str, arguments: dict) -> str:
    def replace(m: re.Match) -> str:
        return str(arguments.get(m.group(1), m.group(0)))
    return re.sub(r"\{(\w+)\}", replace, path_pattern)
