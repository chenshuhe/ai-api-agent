"""OpenAPI 3.x parser: extract endpoints, resolve $ref, flatten @ModelAttribute params."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Parameter:
    name: str
    location: str       # path, query, header
    required: bool
    param_type: str
    description: str
    example: Any = None
    enum: list[str] | None = None


@dataclass
class RequestBody:
    content_type: str
    schema: dict
    required: bool
    description: str


@dataclass
class Endpoint:
    method: str
    path: str
    summary: str
    description: str
    tags: list[str] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)
    request_body: RequestBody | None = None
    response_schema: dict | None = None
    base_url: str = ""
    operation_id: str = ""


@dataclass
class ApiDoc:
    title: str
    version: str
    description: str
    base_url: str
    endpoints: list[Endpoint] = field(default_factory=list)


def _resolve_ref(ref: str, openapi: dict) -> dict:
    if not ref.startswith("#/"):
        return {}
    node = openapi
    for part in ref[2:].split("/"):
        node = node.get(part, {})
    return node if isinstance(node, dict) else {}


def _schema_to_str(schema: dict, depth: int = 0) -> str:
    if not schema or depth > 3:
        return "any"
    ref = schema.get("$ref", "")
    if ref:
        return ref.rsplit("/", 1)[-1]
    stype = schema.get("type", "any")
    if stype == "array":
        return f"array[{_schema_to_str(schema.get('items', {}), depth + 1)}]"
    if stype == "object":
        props = schema.get("properties", {})
        return "{" + ", ".join(f"{k}:{_schema_to_str(v, depth+1)}" for k, v in list(props.items())[:10]) + "}" if props else "object"
    return stype


def _parse_parameters(params_spec: list[dict], openapi: dict) -> list[Parameter]:
    """Parse parameters, flattening Spring @ModelAttribute object params."""
    params = []
    for p in params_spec:
        schema = p.get("schema", {})
        location = p.get("in", "query")
        required = p.get("required", False)

        # Resolve $ref
        ref = schema.get("$ref", "")
        resolved_schema = _resolve_ref(ref, openapi) if ref and location == "query" else schema

        # Flatten @ModelAttribute: query + object with properties
        if location == "query" and "properties" in resolved_schema:
            props = resolved_schema["properties"]
            obj_required = resolved_schema.get("required") or []
            for prop_name, prop_schema in props.items():
                prop_ref = prop_schema.get("$ref", "")
                prop_type = _resolve_ref(prop_ref, openapi).get("type", "string") if prop_ref else prop_schema.get("type", "string")
                params.append(Parameter(
                    name=prop_name, location="query",
                    required=prop_name in obj_required,
                    param_type=prop_type,
                    description=prop_schema.get("description", ""),
                    example=prop_schema.get("example"),
                    enum=prop_schema.get("enum"),
                ))
        else:
            params.append(Parameter(
                name=p.get("name", ""), location=location, required=required,
                param_type=schema.get("type", "string"),
                description=p.get("description", ""),
                example=p.get("example"), enum=schema.get("enum"),
            ))
    return params


def parse_openapi(openapi: dict) -> ApiDoc:
    """Parse OpenAPI 3.x dict into ApiDoc with all endpoints."""
    info = openapi.get("info", {})
    base_url = (openapi.get("servers", [{}])[0].get("url", "")).rstrip("/")
    doc = ApiDoc(title=info.get("title", ""), version=info.get("version", ""),
                 description=info.get("description", ""), base_url=base_url)

    for path_pattern, methods in openapi.get("paths", {}).items():
        for method_name in ("get", "post", "put", "delete", "patch"):
            op = methods.get(method_name)
            if not op:
                continue

            params = _parse_parameters(op.get("parameters", []), openapi)
            body_spec = op.get("requestBody", {})
            body = None
            if body_spec:
                content = body_spec.get("content", {})
                json_ct = content.get("application/json") or content.get("*/*")
                if json_ct:
                    body = RequestBody(content_type="application/json", schema=json_ct.get("schema", {}),
                                       required=body_spec.get("required", False),
                                       description=body_spec.get("description", ""))

            # Build description for AI
            desc_parts = [p for p in [op.get("summary", ""), op.get("description", "")] if p]
            description = ". ".join(desc_parts)
            detail = []
            for p in params:
                req = "required" if p.required else "optional"
                detail.append(f"  {p.location} param {p.name} ({p.param_type}, {req}): {p.description}")
            if body:
                detail.append(f"  request body ({body.content_type}): {_schema_to_str(body.schema)}")
            if detail:
                description += "\nParameters:\n" + "\n".join(detail)

            resp_schema = None
            resp200 = (op.get("responses", {}).get("200") or op.get("responses", {}).get("201"))
            if resp200 and "content" in resp200:
                resp_schema = resp200["content"].get("application/json", {}).get("schema")

            doc.endpoints.append(Endpoint(
                method=method_name.upper(), path=path_pattern,
                summary=op.get("summary", ""), description=description.strip(),
                tags=op.get("tags", []), parameters=params, request_body=body,
                response_schema=resp_schema, base_url=base_url,
                operation_id=op.get("operationId", ""),
            ))
    return doc
