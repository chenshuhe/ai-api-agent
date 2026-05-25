"""Concrete LLM provider implementations.

Each provider creates:
  1. A LangChain chat model
  2. A MessageConverter for provider-specific serialization
  3. An optional callback (e.g., for reasoning_content)
"""

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from ..settings import Settings
from .base import BaseLLMProvider, MessageConverter
from .callbacks import ReasoningContentCallback


class DeepSeekMessageConverter(MessageConverter):
    """Handles DeepSeek's reasoning_content: store after response, inject before request."""

    def __init__(self):
        self._rc_storage: dict[str, str] = {}  # tool_call_id → reasoning_content

    def capture_response_metadata(self, api_message: Any, ai_message: AIMessage) -> None:
        rc = getattr(api_message, "reasoning_content", None)
        if rc:
            ai_message.additional_kwargs["reasoning_content"] = rc
            if hasattr(api_message, "tool_calls") and api_message.tool_calls:
                for tc in api_message.tool_calls:
                    self._rc_storage[tc.id] = rc
                    logger.debug(f"Captured reasoning_content for tool_call {tc.id[:12]}")

    def inject_request_metadata(self, api_messages: list[dict]) -> None:
        injected = False
        for msg in api_messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    rc = self._rc_storage.get(tc.get("id", ""))
                    if rc:
                        msg["reasoning_content"] = rc
                        injected = True
                        logger.debug(f"Injected reasoning_content for tool_call {tc['id'][:12]}")
                        break
        if not injected:
            logger.info("No reasoning_content found to inject in any assistant message")

    def _assistant_with_tools(self, m: AIMessage) -> dict:
        result = super()._assistant_with_tools(m)
        # Preserve reasoning_content from AIMessage if it was captured by callback
        rc = (m.additional_kwargs or {}).get("reasoning_content")
        if rc:
            result["reasoning_content"] = rc
        return result


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek: uses raw OpenAI client for full reasoning_content control.

    Why not ChatOpenAI? The LangChain wrapper drops reasoning_content during
    message serialization. We use raw AsyncOpenAI + MessageConverter instead.
    """

    def create(self) -> tuple[BaseChatModel, MessageConverter, BaseCallbackHandler | None]:
        converter = DeepSeekMessageConverter()
        callback = ReasoningContentCallback()
        # Use raw OpenAI-compatible client directly
        from openai import AsyncOpenAI
        chat_model = ChatOpenAI(
            model=self.model_cfg.name,
            base_url=self.model_cfg.base_url,
            api_key=self.model_cfg.api_key or "sk-not-needed",
            temperature=self.model_cfg.temperature,
            max_tokens=self.model_cfg.max_tokens,
        )
        return chat_model, converter, callback


class StandardOpenAIProvider(BaseLLMProvider):
    """Standard OpenAI provider — no special handling needed."""

    def create(self) -> tuple[BaseChatModel, MessageConverter, BaseCallbackHandler | None]:
        return ChatOpenAI(
            model=self.model_cfg.name,
            base_url=self.model_cfg.base_url,
            api_key=self.model_cfg.api_key or "sk-not-needed",
            temperature=self.model_cfg.temperature,
            max_tokens=self.model_cfg.max_tokens,
        ), MessageConverter(), None


class OllamaProvider(BaseLLMProvider):
    """Ollama via OpenAI-compatible endpoint."""

    def create(self) -> tuple[BaseChatModel, MessageConverter, BaseCallbackHandler | None]:
        return ChatOpenAI(
            model=self.model_cfg.name,
            base_url=self.model_cfg.base_url or "http://localhost:11434/v1",
            api_key=self.model_cfg.api_key or "ollama",
            temperature=self.model_cfg.temperature,
            max_tokens=self.model_cfg.max_tokens,
        ), MessageConverter(), None


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude via official SDK."""

    def create(self) -> tuple[BaseChatModel, MessageConverter, BaseCallbackHandler | None]:
        return ChatAnthropic(
            model=self.model_cfg.name,
            api_key=self.model_cfg.api_key or None,
            base_url=self.model_cfg.base_url or None,
            temperature=self.model_cfg.temperature,
            max_tokens=self.model_cfg.max_tokens,
        ), MessageConverter(), None
