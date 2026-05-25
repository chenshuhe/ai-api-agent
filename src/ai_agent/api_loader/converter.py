"""Convert parsed API Endpoints into LangChain StructuredTool objects."""

import json
from collections import Counter
from urllib.parse import urlparse

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from .parser import Endpoint

MAX_NAME_LEN = 64


def _sanitize(name: str) -> str:
    """Keep only [a-zA-Z0-9_-] for tool name compatibility."""
    result = []
    for c in name:
        if c in "_-" or ("a" <= c <= "z") or ("A" <= c <= "Z") or ("0" <= c <= "9"):
            result.append(c)
        else:
            result.append("_")
    return "".join(result)


def _uniq_names(endpoints: list[Endpoint]) -> list[str]:
    """Generate unique tool names."""
    def base(ep: Endpoint) -> str:
        if ep.operation_id:
            return ep.operation_id.replace("-", "_").replace(" ", "_")
        parts = [p for p in ep.path.strip("/").split("/") if p and not p.startswith("{")]
        return f"{ep.method.lower()}_{'_'.join(parts)}"

    def svc(ep: Endpoint) -> str:
        if ep.tags:
            return ep.tags[0].lower().replace("-", "_").replace(" ", "_")
        if ep.base_url:
            return urlparse(ep.base_url).netloc.split(":")[0].replace(".", "_")
        return "svc"

    counts = Counter(_sanitize(base(ep)) for ep in endpoints)
    seen: Counter[str] = Counter()
    names = []
    for ep in endpoints:
        b = _sanitize(base(ep))
        candidate = f"{b}_{_sanitize(svc(ep))}" if counts[b] > 1 else b
        seen[candidate] += 1
        if seen[candidate] > 1:
            candidate = f"{candidate}_{seen[candidate]}"
        if len(candidate) > MAX_NAME_LEN:
            candidate = candidate[:MAX_NAME_LEN - 4] + "_etc"
        names.append(candidate)
    return names


def _build_pydantic_model(ep: Endpoint, name: str) -> type[BaseModel]:
    """Build a dynamic Pydantic model for the tool's input schema."""
    fields: dict = {}

    for p in ep.parameters:
        if p.location in ("header", "cookie"):
            continue
        py_type = {"integer": int, "number": float, "boolean": bool}.get(p.param_type, str)
        desc = p.description or f"{p.param_type} value"
        if p.required:
            fields[p.name] = (py_type, Field(description=desc))
        else:
            fields[p.name] = (py_type | None, Field(default=None, description=desc))

    if ep.request_body and ep.request_body.schema:
        body = ep.request_body.schema
        if "properties" in body:
            for key, val in body["properties"].items():
                py_type = {"integer": int, "number": float, "boolean": bool}.get(val.get("type", "string"), str)
                desc = val.get("description", f"Field: {key}")
                fields[key] = (py_type | None, Field(default=None, description=desc))
        else:
            ref = body.get("$ref", "")
            type_info = ref.rsplit("/", 1)[-1] if ref else "object"
            fields["body"] = (str | None, Field(default=None, description=f"Request body as JSON ({type_info})"))

    if not fields:
        fields["no_params"] = (str | None, Field(default=None, description="No parameters needed"))

    model_name = f"Tool_{name}"[:64]
    # Replace any field names that start with underscore
    clean_fields = {}
    for k, v in fields.items():
        clean_k = k.lstrip("_") or "field"
        clean_fields[clean_k] = v
    return create_model(model_name, **clean_fields)  # type: ignore


def _build_description(ep: Endpoint) -> str:
    desc = ep.summary or ep.description or f"Call {ep.method} {ep.path}"
    if ep.base_url:
        desc += f"\nBase URL: {ep.base_url}"
    desc += f"\nHTTP Method: {ep.method}\nPath: {ep.path}"
    return desc[:1024]


def endpoint_to_tool(ep: Endpoint, name: str) -> StructuredTool:
    """Convert a single Endpoint to a LangChain StructuredTool.

    The tool's func is a placeholder - the actual HTTP call is handled
    by the executor, which is injected at runtime via the agent graph.
    """
    model = _build_pydantic_model(ep, name)

    def _placeholder(**kwargs) -> str:
        """Placeholder - actual execution handled by agent graph."""
        return json.dumps({"tool": name, "args": kwargs})

    _placeholder.__name__ = name

    return StructuredTool(
        name=name,
        description=_build_description(ep),
        func=_placeholder,
        args_schema=model,
    )


def convert_all(endpoints: list[Endpoint]) -> tuple[list[StructuredTool], dict[str, Endpoint]]:
    """Convert all endpoints to LangChain tools + index."""
    names = _uniq_names(endpoints)
    tools = [endpoint_to_tool(ep, name) for ep, name in zip(endpoints, names)]
    index = dict(zip(names, endpoints))
    return tools, index
