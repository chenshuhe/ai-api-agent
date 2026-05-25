"""Pydantic request/response schemas for the FastAPI routes."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ConversationCreate(BaseModel):
    name: str = Field(default="新对话", min_length=1, max_length=100)


class ConversationSwitch(BaseModel):
    id: str


class ConfigUpdate(BaseModel):
    api_docs: dict | None = None
    model: dict | None = None
    api_auth: dict | None = None
    api_scenarios: dict | None = None
    auto_login: dict | None = None
    global_params: list[dict] | None = None
    project_dir: str | None = None


class StatusResponse(BaseModel):
    ready: bool
    endpoints: int
    model: str
    provider: str
    loading: bool
    load_error: str | None
    current_conv: str
