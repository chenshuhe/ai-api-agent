"""Tests for OpenAPI parser."""

from src.ai_agent.api_loader.parser import Endpoint, parse_openapi


def test_parse_basic(sample_openapi):
    """Test basic endpoint parsing."""
    doc = parse_openapi(sample_openapi)
    assert doc.title == "Test API"
    assert doc.base_url == "http://localhost:8080"
    assert len(doc.endpoints) == 3  # GET /users, POST /users, GET /users/{id}


def test_parse_get_endpoint(sample_openapi):
    """Test GET endpoint with query params."""
    doc = parse_openapi(sample_openapi)
    get_users = [e for e in doc.endpoints if e.operation_id == "getUsers"][0]
    assert get_users.method == "GET"
    assert get_users.path == "/users"
    assert len(get_users.parameters) == 2
    assert get_users.parameters[0].name == "page"
    assert get_users.parameters[0].location == "query"
    assert get_users.parameters[0].param_type == "integer"


def test_parse_path_param(sample_openapi):
    """Test path parameter parsing."""
    doc = parse_openapi(sample_openapi)
    get_user = [e for e in doc.endpoints if e.operation_id == "getUser"][0]
    assert get_user.method == "GET"
    assert get_user.path == "/users/{id}"
    assert len(get_user.parameters) == 1
    assert get_user.parameters[0].name == "id"
    assert get_user.parameters[0].location == "path"
    assert get_user.parameters[0].required is True


def test_parse_post_body(sample_openapi):
    """Test POST endpoint with request body."""
    doc = parse_openapi(sample_openapi)
    create_user = [e for e in doc.endpoints if e.operation_id == "createUser"][0]
    assert create_user.method == "POST"
    assert create_user.request_body is not None
    assert create_user.request_body.content_type == "application/json"
    assert "name" in str(create_user.request_body.schema.get("properties", {}))


def test_parse_model_attribute_flatten():
    """Test Spring @ModelAttribute parameter flattening via $ref."""
    import copy
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test"},
        "paths": {
            "/videos/stats": {
                "get": {
                    "summary": "Stats",
                    "operationId": "getVideoStats",
                    "parameters": [
                        {
                            "name": "filter",
                            "in": "query",
                            "required": True,
                            "schema": {"$ref": "#/components/schemas/SearchFilter"},
                        }
                    ],
                    "responses": {"200": {"description": "OK"}},
                }
            }
        },
        "components": {
            "schemas": {
                "SearchFilter": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "搜索关键词"},
                        "status": {"type": "string"},
                    },
                },
            },
        },
    }
    doc = parse_openapi(spec)
    endpoint = doc.endpoints[0]
    # Should have flattened keyword and status, not a single 'filter' param
    names = [p.name for p in endpoint.parameters]
    assert "keyword" in names
    assert "status" in names
    assert "filter" not in names
