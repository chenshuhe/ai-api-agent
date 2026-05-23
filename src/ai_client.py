"""
AI 模型抽象层
=============
为三种 LLM 提供商提供统一接口：
  - OpenAI 兼容（包括 DeepSeek、Ollama 等 OpenAI 协议的服务）
  - Anthropic Claude（原生 SDK）
  - Ollama（通过 OpenAI 兼容端点）

设计模式：策略模式 + 工厂方法
  - AIClientBase      抽象基类，定义 chat() / chat_stream() 接口
  - OpenAIClient      封装 AsyncOpenAI SDK
  - AnthropicClient   封装 AsyncAnthropic SDK
  - AIClientBase.create()  工厂方法，根据 config.model.provider 返回对应实例

关键设计：reasoning_content 透传
  DeepSeek V4 等 reasoning 模型在 tool_call 响应中会附带 reasoning_content 字段。
  API 要求后续请求必须将其原样传回。AIMessage.extra 字典用于存储这类
  provider-specific 的元数据，在 _convert_messages 时自动合并。

流式 vs 非流式：
  chat()         一次性返回完整响应（用于 tool_call 决策阶段）
  chat_stream()  异步生成器逐 token 输出（用于最终回复展示）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from src.config import Config


# ---- 内部消息模型 ----
# 不与任何特定 LLM SDK 耦合，统一表达对话消息

@dataclass
class AIMessage:
    """一条对话消息（系统提示 / 用户输入 / AI 回复 / 工具调用结果）"""
    role: str                        # system / user / assistant / tool
    content: str                     # 消息文本
    tool_call_id: str | None = None  # tool 角色时：关联的 tool_call id
    tool_calls: list[dict] | None = None  # assistant 角色时：发出的 tool_call 列表
    extra: dict | None = None        # provider-specific 额外字段（如 DeepSeek 的 reasoning_content）


@dataclass
class ToolCall:
    """AI 返回的单个工具调用"""
    id: str          # tool_call 唯一 ID（后续 tool_result 需关联此 ID）
    name: str        # 工具名（对应 Endpoint 转换后的名称）
    arguments: dict  # 参数键值对


@dataclass
class AIResponse:
    """AI 模型的一次响应（可能包含文本、工具调用，或两者都有）"""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    extra: dict | None = None  # provider-specific 元数据


# ---- 抽象基类 ----

class AIClientBase(ABC):
    """AI 客户端的抽象基类，定义统一接口"""

    @abstractmethod
    async def chat(
        self, messages: list[AIMessage], tools: list[dict], stream: bool = False
    ) -> AIResponse:
        """发送消息，获取完整响应（可能含 tool_calls）"""
        ...

    @abstractmethod
    async def chat_stream(
        self, messages: list[AIMessage], tools: list[dict]
    ) -> AsyncGenerator[str, None]:
        """流式对话：逐 token 产出文本"""
        ...

    @staticmethod
    def create(config: Config) -> "AIClientBase":
        """工厂方法：根据配置返回对应的客户端实例"""
        provider = config.model.get("provider", "openai").lower()
        if provider == "anthropic":
            return AnthropicClient(config)
        elif provider == "ollama":
            return OllamaClient(config)
        else:
            return OpenAIClient(config)


# ---- OpenAI 兼容客户端 ----
# 覆盖：OpenAI / DeepSeek / 所有 OpenAI 兼容接口

class OpenAIClient(AIClientBase):
    """
    使用 OpenAI Python SDK 的客户端。

    注意：DeepSeek V4 等模型需要传递 reasoning_content。
    实现方式：
      - 响应中通过 msg.model_extra 捕获额外字段
      - 存储到 AIResponse.extra → AIMessage.extra
      - _convert_messages 时合并到消息字典中
    """

    def __init__(self, config: Config):
        from openai import AsyncOpenAI

        self.config = config
        # API key 为空时用占位符（Ollama 等本地模型不需要真实 key）
        api_key = config.model.get("api_key") or "sk-not-needed"
        self.client = AsyncOpenAI(
            base_url=config.model.get("base_url"),
            api_key=api_key,
        )
        self.model = config.model.get("name", "gpt-4o")
        self.temperature = config.model.get("temperature", 0.1)
        self.max_tokens = config.model.get("max_tokens", 4096)

    def _validate_messages(self, messages: list[dict]):
        """Ensure every 'tool' message follows an assistant with matching tool_calls."""
        valid_ids = set()
        for i, m in enumerate(messages):
            if m["role"] == "assistant" and m.get("tool_calls"):
                valid_ids = {tc["id"] for tc in m["tool_calls"]}
            elif m["role"] == "tool":
                tid = m.get("tool_call_id", "")
                if tid not in valid_ids:
                    ctx = [msg["role"] for msg in messages[max(0,i-3):i]]
                    raise ValueError(
                        f"Message #{i}: tool call_id '{tid}' not in preceding tool_calls "
                        f"(valid: {valid_ids}). Context: {ctx} -> tool"
                    )

    def _convert_messages(self, messages: list[AIMessage]) -> list[dict]:
        """
        将内部 AIMessage 列表转为 OpenAI API 格式。

        关键：如果 AIMessage.extra 有值（如 reasoning_content），
        会合并到输出字典中，确保 provider-specific 字段被透传。
        """
        result = []
        for m in messages:
            msg: dict = {"role": m.role, "content": m.content}
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                msg["tool_calls"] = m.tool_calls
            if m.extra:
                msg.update(m.extra)  # 注入 reasoning_content 等
            result.append(msg)
        self._validate_messages(result)
        return result

    async def chat(self, messages: list[AIMessage], tools: list[dict], stream: bool = False) -> AIResponse:
        """发送非流式请求，返回完整响应（用于 tool_call 决策）"""
        converted = self._convert_messages(messages)
        kwargs: dict = {
            "model": self.model,
            "messages": converted,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"  # 让模型自行决定是否调用工具

        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        # 解析工具调用
        tool_calls = []
        if msg.tool_calls:
            import json  # 延迟导入，仅在需要时
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        # 捕获 provider-specific 额外字段（如 DeepSeek reasoning_content）
        extra = None
        raw = msg.model_extra  # pydantic model_extra 存有 API 返回的额外字段
        if raw:
            extra = {k: v for k, v in raw.items() if k not in ("role", "content", "tool_calls")}

        return AIResponse(content=msg.content or "", tool_calls=tool_calls, extra=extra)

    async def chat_stream(self, messages: list[AIMessage], tools: list[dict]) -> AsyncGenerator[str, None]:
        """流式对话：逐 token 生成文本（用于最终回复展示）"""
        kwargs: dict = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},  # 请求包含 token 用量统计
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content


class OllamaClient(OpenAIClient):
    """Ollama 客户端：继承 OpenAI 客户端，通过 OpenAI 兼容协议通信"""
    pass


# ---- Anthropic Claude 客户端 ----

class AnthropicClient(AIClientBase):
    """使用 Anthropic 原生 SDK 的客户端"""

    def __init__(self, config: Config):
        import anthropic

        self.config = config
        self.client = anthropic.AsyncAnthropic(
            api_key=config.model.get("api_key"),
            base_url=config.model.get("base_url"),
        )
        self.model = config.model.get("name", "claude-sonnet-4-6")
        self.temperature = config.model.get("temperature", 0.1)
        self.max_tokens = config.model.get("max_tokens", 4096)

    def _convert_messages(self, messages: list[AIMessage]) -> list[dict]:
        """
        AIMessage → Anthropic Messages API 格式。

        与 OpenAI 格式的主要差异：
          - 工具调用结果用 content block 而非 tool_call_id
          - assistant 的工具调用是 content block 列表
          - 文本消息用 content block 数组表达
        """
        result = []
        for m in messages:
            if m.role == "system":
                # Anthropic 的 system 消息是顶层参数，但 SDK 处理方式特殊
                result.append({"role": "user", "content": m.content})
            elif m.role == "tool":
                # 工具执行结果 → tool_result content block
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                    }],
                })
            elif m.role == "assistant" and m.tool_calls:
                # AI 发出的工具调用 → tool_use content block
                tool_uses = [
                    {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
                    for tc in m.tool_calls
                ]
                result.append({"role": "assistant", "content": tool_uses})
            else:
                # 普通文本消息 → text content block
                result.append({"role": m.role, "content": [{"type": "text", "text": m.content}]})
        return result

    async def chat(self, messages: list[AIMessage], tools: list[dict], stream: bool = False) -> AIResponse:
        kwargs: dict = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        resp = await self.client.messages.create(**kwargs)

        # Anthropic 响应是 content block 列表
        content = ""
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return AIResponse(content=content, tool_calls=tool_calls)

    async def chat_stream(self, messages: list[AIMessage], tools: list[dict]) -> AsyncGenerator[str, None]:
        """流式对话（通过 Anthropic streaming SDK）"""
        kwargs: dict = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "text":
                    yield event.text
