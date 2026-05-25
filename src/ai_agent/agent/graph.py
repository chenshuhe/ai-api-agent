"""
LangGraph agent graph.

The graph consists of two nodes in a loop:
  agent_node: LLM decides what to do (respond or call tools)
  tool_node:  Execute tool calls (API calls or internal tools)

Uses raw OpenAI client for DeepSeek reasoning_content compatibility.
"""

import json
from typing import Any, AsyncGenerator, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from ..api_loader.converter import convert_all
from ..api_loader.fetcher import load_all
from ..api_loader.parser import ApiDoc, Endpoint
from ..execution.executor import execute_async
from ..settings import Settings, get_settings
from .prompts import build as build_prompt
from .state import AgentState
from .tools import get_internal_tools, set_project_dir


class ApiAgent:
    """LangGraph-based API agent wrapping the StateGraph."""

    def __init__(self, settings: Settings | None = None, preload_messages: list | None = None,
                 custom_params: list[dict] | None = None):
        self.settings = settings or get_settings()
        self.tool_index: dict[str, Endpoint] = {}
        self.api_tools: list = []
        self.all_tools: list = []
        self.graph: StateGraph | None = None
        self.checkpointer = MemorySaver()
        self._config: RunnableConfig = {"configurable": {"thread_id": "default"}}
        self._custom_params = custom_params or []

        # Set project dir for code tools
        if self.settings.project_dir:
            set_project_dir(self.settings.project_dir)

    async def load_endpoints(self) -> int:
        """Fetch API docs and convert endpoints to LangChain tools."""
        logger.info(f"Loading API docs from {len(self.settings.api_docs.urls)} sources...")
        docs = await load_all(self.settings.api_docs.urls, self.settings.api_docs.timeout)

        endpoints: list[Endpoint] = []
        for doc in docs:
            for ep in doc.endpoints:
                if not ep.base_url:
                    ep.base_url = doc.base_url
                endpoints.append(ep)

        self.api_tools, self.tool_index = convert_all(endpoints)
        self.all_tools = self.api_tools + get_internal_tools()
        logger.info(f"Loaded {len(endpoints)} API tools + {len(get_internal_tools())} internal tools")
        return len(endpoints)

    def build_graph(self):
        """Build the LangGraph StateGraph using the LLM factory layer."""
        from ..llm.factory import create_llm

        self._llm_handle = create_llm(self.settings)
        self._llm_handle.set_tool_schemas(self.all_tools)
        system_msg = SystemMessage(content=build_prompt(self.settings))
        h = self._llm_handle  # shorthand

        use_raw_client = self.settings.model.provider in ("deepseek", "openai", "ollama")

        async def agent_node(state: AgentState) -> dict:
            messages = state.get("messages", [])

            if use_raw_client:
                # Use raw AsyncOpenAI via LLMHandle's MessageConverter
                from openai import AsyncOpenAI
                client = AsyncOpenAI(
                    base_url=self.settings.model.base_url,
                    api_key=self.settings.model.api_key or "sk-not-needed",
                )
                api_messages = h.converter.to_api_messages(messages, system_msg.content)
                h.converter.inject_request_metadata(api_messages)

                resp = await client.chat.completions.create(
                    model=h.model_name, messages=api_messages,
                    temperature=h.temperature, max_tokens=h.max_tokens,
                    tools=h.tool_schemas or None,
                    tool_choice="auto" if h.tool_schemas else None,
                )
                choice = resp.choices[0]
                raw_msg = choice.message

                if raw_msg.tool_calls:
                    lc_tool_calls = [{
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments) if tc.function.arguments else {},
                        "id": tc.id,
                    } for tc in raw_msg.tool_calls]
                    ai_msg = AIMessage(content=raw_msg.content or "", tool_calls=lc_tool_calls)
                else:
                    ai_msg = AIMessage(content=raw_msg.content or "")

                h.converter.capture_response_metadata(raw_msg, ai_msg)
                return {"messages": [ai_msg]}
            else:
                # Anthropic via LangChain
                lc_messages = list(messages)
                if not lc_messages or not isinstance(lc_messages[0], SystemMessage):
                    lc_messages = [system_msg] + lc_messages
                response = await h.chat_model.bind_tools(self.all_tools).ainvoke(lc_messages)
                return {"messages": [response]}

        async def tool_node(state: AgentState) -> dict:
            messages = state.get("messages", [])
            last_msg = messages[-1] if messages else None
            if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
                return {}

            tool_messages = []
            for tc in last_msg.tool_calls:
                tool_name = tc["name"]
                args = tc.get("args") or tc.get("input", {})
                logger.debug(f"Executing tool: {tool_name}")

                if tool_name.startswith("internal_"):
                    result = await _handle_internal(tool_name, args, self)
                else:
                    gps = state.get("custom_params") or self._custom_params
                    result = await execute_async(
                        tool_name, args, self.tool_index, self.settings,
                        override_params=gps,
                    )
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

            return {"messages": tool_messages}

        def should_continue(state: AgentState) -> Literal["tool_node", END]:
            """Route: if last message has tool_calls, go to tool_node, else END."""
            messages = state.get("messages", [])
            last_msg = messages[-1] if messages else None
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                return "tool_node"
            return END

        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)  # type: ignore
        workflow.add_node("tool_node", tool_node)  # type: ignore
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, {"tool_node": "tool_node", END: END})
        workflow.add_edge("tool_node", "agent")

        self.graph = workflow.compile(checkpointer=self.checkpointer)
        return self.graph

    async def chat(self, user_input: str, thread_id: str = "default") -> str:
        """Send a message and get the full response."""
        if not self.graph:
            self.build_graph()

        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        result = await self.graph.ainvoke(  # type: ignore
            {"messages": [HumanMessage(content=user_input)]},
            config,
        )
        messages = result.get("messages", [])
        last = messages[-1] if messages else None
        return last.content if hasattr(last, "content") else str(last)

    async def chat_stream(self, user_input: str, thread_id: str = "default") -> AsyncGenerator[str, None]:
        """Stream the agent's response: use ainvoke for tool loop, yield final content."""
        if not self.graph:
            self.build_graph()

        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # Use non-streaming ainvoke for reliability with DeepSeek
        result = await self.graph.ainvoke(  # type: ignore
            {"messages": [HumanMessage(content=user_input)]},
            config,
        )
        messages = result.get("messages", [])
        # Yield the content of the last message
        if messages:
            last = messages[-1]
            if hasattr(last, "content") and last.content:
                yield last.content

    def set_thread_id(self, thread_id: str):
        self._config = {"configurable": {"thread_id": thread_id}}

    def clear_history(self, thread_id: str = "default"):
        """Clear conversation by creating a new thread."""
        # LangGraph MemorySaver auto-forgets when thread expires
        pass

    @property
    def endpoint_count(self) -> int:
        return len(self.tool_index)


# ---- Internal tool handler ----

async def _handle_internal(tool_name: str, args: dict, agent: "ApiAgent") -> str:
    """Route internal tool calls to their handlers."""
    if tool_name == "internal_set_global_header":
        name = args.get("name", "")
        value = args.get("value", "")
        if not name:
            return json.dumps({"error": "header name required"})
        # Update custom params
        gps = [p for p in agent._custom_params if p.get("name") != name]
        gps.append({"name": name, "value": value, "type": "header"})
        agent._custom_params = gps
        # Persist to settings
        from ..settings import GlobalParam
        agent.settings.global_params = [GlobalParam(**p) for p in gps]
        agent.settings.save_to_yaml()
        return json.dumps({"status": "ok", "message": f"Header '{name}' set."})

    elif tool_name == "internal_list_global_headers":
        return json.dumps(agent._custom_params, ensure_ascii=False)

    elif tool_name == "internal_switch_scenario":
        sname = args.get("name", "")
        scenarios = agent.settings.api_scenarios
        valid = [s.get("name") for s in scenarios.scenarios]
        if sname not in valid:
            return json.dumps({"error": f"Unknown scenario '{sname}'. Available: {valid}"})
        scenarios.active = sname
        agent.settings.save_to_yaml()
        return json.dumps({"status": "ok", "message": f"Switched to '{sname}'."})

    elif tool_name == "internal_run_test":
        return json.dumps({
            "status": "ok",
            "message": f"Testing '{args.get('feature', '')}'. Follow: CREATE → internal_test_step → QUERY → internal_test_step → UPDATE → internal_test_step → DELETE → internal_test_step → Report."
        })

    elif tool_name == "internal_test_step":
        return json.dumps({"status": "ok", "logged": f"[{args.get('step', '?')}] {args.get('api', '?')}: {args.get('status', '?')} - {args.get('detail', '')}"})

    elif tool_name in ("internal_search_code", "internal_read_code", "internal_edit_code"):
        # These are @tool decorated - call their func directly
        from .tools import internal_search_code, internal_read_code, internal_edit_code
        mapping = {
            "internal_search_code": internal_search_code,
            "internal_read_code": internal_read_code,
            "internal_edit_code": internal_edit_code,
        }
        func = mapping[tool_name]
        return await func.ainvoke(args)

    return json.dumps({"error": f"Unknown internal tool: {tool_name}"})


def get_all_endpoints(docs: list[ApiDoc]) -> list[Endpoint]:
    """Flatten all endpoints from multiple ApiDocs."""
    endpoints = []
    for doc in docs:
        for ep in doc.endpoints:
            if not ep.base_url:
                ep.base_url = doc.base_url
            endpoints.append(ep)
    return endpoints
