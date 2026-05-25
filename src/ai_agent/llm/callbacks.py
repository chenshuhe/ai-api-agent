"""LangChain callbacks for provider-specific behavior (enterprise pattern)."""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from loguru import logger


class ReasoningContentCallback(BaseCallbackHandler):
    """Captures and injects DeepSeek reasoning_content via LangChain callback system.

    This is the enterprise alternative to monkey-patching the client.
    It hooks into LangChain's lifecycle without touching internal APIs.
    """

    def __init__(self):
        self.reasoning_store: dict[str, str] = {}  # tool_call_id → reasoning_content

    def on_llm_end(self, response, **kwargs) -> None:
        """After LLM responds, capture reasoning_content from the raw API response."""
        # Access the raw OpenAI response through llm_output
        llm_output = getattr(response, "llm_output", {}) or {}
        raw = llm_output.get("raw_response")
        if raw is None:
            return

        for gen in response.generations:
            for gen_item in gen if isinstance(gen, list) else [gen]:
                msg = gen_item.message if hasattr(gen_item, "message") else gen_item
                if not isinstance(msg, AIMessage):
                    continue

                # Extract reasoning_content from raw response choices
                choices = getattr(raw, "choices", [])
                if choices:
                    rc = getattr(choices[0].message, "reasoning_content", None)
                    if rc:
                        msg.additional_kwargs["reasoning_content"] = rc
                        # Store for injection on next call
                        if hasattr(choices[0].message, "tool_calls") and choices[0].message.tool_calls:
                            for tc in choices[0].message.tool_calls:
                                self.reasoning_store[tc.id] = rc
                        logger.debug(f"Captured reasoning_content ({len(rc)} chars)")

    def inject_into_messages(self, api_messages: list[dict]) -> None:
        """Inject stored reasoning_content into outgoing API messages."""
        for msg in api_messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    rc = self.reasoning_store.get(tc.get("id", ""))
                    if rc:
                        msg["reasoning_content"] = rc
                        break
