"""LLM provider registry + factory — single entry point for creating LLM instances."""

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel

from ..settings import Settings
from .base import BaseLLMProvider, MessageConverter
from .callbacks import ReasoningContentCallback
from . import _providers

# Provider registry: map config name → provider class
PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "openai": _providers.StandardOpenAIProvider,
    "deepseek": _providers.DeepSeekProvider,
    "ollama": _providers.OllamaProvider,
    "anthropic": _providers.AnthropicProvider,
}


def register_provider(name: str, provider_cls: type[BaseLLMProvider]):
    """Register a custom provider at runtime (plugin system)."""
    PROVIDER_REGISTRY[name] = provider_cls


def create_llm(settings: Settings) -> "LLMHandle":
    """Factory: create the appropriate LLM + message converter + callback for the configured provider.

    Auto-detects DeepSeek from base_url even if provider is set to 'openai'.
    """
    provider_name = settings.model.provider.lower()
    base_url = settings.model.base_url.lower()

    # Auto-detect: DeepSeek needs special handling for reasoning_content
    if "deepseek" in base_url or "deepseek" in settings.model.name.lower():
        provider_name = "deepseek"
    elif "ollama" in base_url or provider_name == "ollama":
        provider_name = "ollama"
    elif "anthropic" in base_url or provider_name == "anthropic":
        provider_name = "anthropic"

    provider_cls = PROVIDER_REGISTRY.get(provider_name)
    if provider_cls is None:
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Available: {list(PROVIDER_REGISTRY.keys())}. "
            f"Use register_provider() to add custom providers."
        )

    provider = provider_cls(settings)
    chat_model, converter, callback = provider.create()
    return LLMHandle(
        chat_model=chat_model,
        converter=converter,
        callback=callback,
        model_name=settings.model.name,
        temperature=settings.model.temperature,
        max_tokens=settings.model.max_tokens,
    )


class LLMHandle:
    """Bundled LLM resources: model + converter + callback + config."""

    def __init__(self, chat_model: BaseChatModel, converter: MessageConverter,
                 callback: BaseCallbackHandler | None, model_name: str,
                 temperature: float, max_tokens: int):
        self.chat_model = chat_model
        self.converter = converter
        self.callback = callback
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_schemas: list[dict] = []

    def set_tool_schemas(self, tools: list[Any]):
        """Generate OpenAI-format tool schemas from LangChain tools."""
        self.tool_schemas = []
        for t in tools:
            schema = {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                },
            }
            if hasattr(t, "args_schema") and t.args_schema:
                try:
                    schema["function"]["parameters"] = t.args_schema.model_json_schema()
                except Exception:
                    schema["function"]["parameters"] = {"type": "object", "properties": {}}
            self.tool_schemas.append(schema)
