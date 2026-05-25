"""Pytest fixtures."""

import pytest
from src.ai_agent.settings import Settings


@pytest.fixture
def settings():
    """Create a test Settings instance with defaults."""
    return Settings()


@pytest.fixture
def sample_openapi():
    """A minimal OpenAPI 3.x spec for testing."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0"},
        "servers": [{"url": "http://localhost:8080"}],
        "paths": {
            "/users": {
                "get": {
                    "summary": "List users",
                    "operationId": "getUsers",
                    "parameters": [
                        {"name": "page", "in": "query", "schema": {"type": "integer"}},
                        {"name": "size", "in": "query", "schema": {"type": "integer"}},
                    ],
                    "responses": {"200": {"description": "OK"}},
                },
                "post": {
                    "summary": "Create user",
                    "operationId": "createUser",
                    "requestBody": {
                        "content": {"application/json": {"schema": {"type": "object", "properties": {"name": {"type": "string"}, "email": {"type": "string"}}}}},
                    },
                    "responses": {"201": {"description": "Created"}},
                },
            },
            "/users/{id}": {
                "get": {
                    "summary": "Get user",
                    "operationId": "getUser",
                    "parameters": [
                        {"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}},
                    ],
                    "responses": {"200": {"description": "OK"}},
                },
            },
        },
        "components": {
            "schemas": {
                "SearchFilter": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string"},
                        "status": {"type": "string"},
                    },
                },
            },
        },
    }
