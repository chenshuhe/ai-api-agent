"""LLM provider abstraction layer — enterprise-grade, extensible."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ..settings import Settings


@dataclass
class MessageConverter:
    """Converts between LangChain messages and provider-specific API format.

    Subclass this to handle provider-specific fields like DeepSeek's reasoning_content.
    """

    def to_api_messages(self, messages: list[BaseMessage], system_prompt: str) -> list[dict]:
        """Convert LangChain messages to API dicts. Override for provider-specific fields."""
        result = [{"role": "system", "content": system_prompt}]
        for m in messages:
            if isinstance(m, SystemMessage):
                continue
            role = _message_role(m)
            if role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": getattr(m, "tool_call_id", ""),
                    "content": str(m.content),
                })
            elif role == "assistant" and isinstance(m, AIMessage) and m.tool_calls:
                result.append(self._assistant_with_tools(m))
            elif role == "assistant":
                result.append({"role": "assistant", "content": str(m.content) if m.content else ""})
            else:
                result.append({"role": "user", "content": str(m.content) if m.content else ""})
        return result

    def _assistant_with_tools(self, m: AIMessage) -> dict:
        """Build assistant message dict with tool_calls."""
        import json
        tc_list = []
        for tc in m.tool_calls:
            args = tc.get("args") or tc.get("input", {})
            tc_list.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })
        return {"role": "assistant", "content": m.content or "", "tool_calls": tc_list}

    def capture_response_metadata(self, api_message: Any, ai_message: AIMessage) -> None:
        """Capture provider-specific response metadata onto the AIMessage."""
        pass

    def inject_request_metadata(self, api_messages: list[dict]) -> None:
        """Inject provider-specific fields into outgoing API messages."""
        pass


class BaseLLMProvider(ABC):
    """Abstract provider: creates a chat model + message converter + optional callback.

    To add a new provider:
      1. Subclass this
      2. Implement get_chat_model() and create()
      3. Register in factory.PROVIDER_REGISTRY
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_cfg = settings.model

    @abstractmethod
    def create(self) -> tuple[BaseChatModel, MessageConverter, BaseCallbackHandler | None]:
        """Return (chat_model, message_converter, optional_callback)."""
        ...


def _message_role(m: BaseMessage) -> str:
    if isinstance(m, AIMessage):
        return "assistant"
    if isinstance(m, HumanMessage):
        return "user"
    if isinstance(m, SystemMessage):
        return "system"
    if isinstance(m, ToolMessage):
        return "tool"
    name = m.__class__.__name__.lower()
    if "tool" in name:
        return "tool"
    return "user"
