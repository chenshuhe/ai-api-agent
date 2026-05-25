"""LangGraph agent state definition."""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The state of the agent graph.

    messages: Conversation history (system + user + assistant + tool messages).
              Using add_messages reducer automatically merges new messages.
    endpoints_loaded: Whether API docs have been loaded.
    custom_params: Per-conversation global params (headers/query).
    pending_edit: Pending code edit awaiting user confirmation.
    """
    messages: Annotated[list[Any], add_messages]
    endpoints_loaded: bool
    custom_params: list[dict]
    pending_edit: dict | None
