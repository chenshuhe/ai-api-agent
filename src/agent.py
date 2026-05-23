"""
核心 Agent 编排模块
===================
这是项目的大脑——负责对话循环、工具调用编排、自动登录流程。

架构概览：

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  用户输入     │ →   │  AI Client   │ →   │ API Executor │
  │  (自然语言)   │ ←   │  (LLM判断)   │ ←   │  (HTTP请求)  │
  └──────────────┘     └──────────────┘     └──────────────┘
                              ↓
                       ┌──────────────┐
                       │ Internal Tool│  (set_global_header,
                       │   Handler    │   run_test, etc.)
                       └──────────────┘

工具调用循环 (Tool Call Loop)：
  1. 用户输入 → AI 决定是否调用工具
  2. 如果 AI 返回 tool_calls → 执行工具 → 结果追加到对话
  3. 再次调用 AI → 可能产生更多 tool_calls
  4. 循环直到 AI 返回纯文本回复（最多 20 轮）

内部工具 vs 外部工具：
  - 外部工具：对应 OpenAPI 文档中的 API 端点，通过 api_executor 发送 HTTP 请求
  - 内部工具：以 "internal_" 开头，由 _handle_internal_tool() 本地处理
    - internal_set_global_header   设置全局请求头（登录后保存 token）
    - internal_clear_global_header 清除全局请求头
    - internal_list_global_headers 列出当前全局请求头
    - internal_switch_scenario     切换请求环境
    - internal_run_test           启动自动化接口测试
    - internal_test_step          记录测试步骤结果
"""

import json
from typing import AsyncGenerator

from src.ai_client import AIClientBase, AIMessage, AIResponse, ToolCall
from src.api_docs import Endpoint
from src.api_executor import execute_tool_call
from src.config import Config

# ============================================================
# 系统提示词
# ============================================================
# 这是 AI 的"角色设定"和"行为规范"。它会出现在每条对话的最前面。
# 占位符在 __init__ 中被替换：
#   "The current API request scenario..." → "Current scenario: production"
#   "Current global headers..."           → "Current global headers: X-Dts-Admin-Token=eyJ..."

def _build_system_prompt(config: Config) -> str:
    """根据配置动态生成系统提示词（项目通用）。"""
    al = config.auto_login
    enabled = al.get("enabled", True)
    header_name = al.get("header_name", "X-Dts-Admin-Token")
    login_hint = al.get("login_hint", "手机号")

    prompt = """You are an AI API assistant that helps users interact with backend APIs.

You have access to API tools and internal tools. Internal tools start with "internal_".

"""

    if enabled:
        prompt += f"""## CRITICAL: Auto-Login Flow

When a user asks to "login and do X" or provides a phone number:

STEP 1 - Find and call the token/login API with the {login_hint}.
STEP 2 - After receiving the token, you MUST call internal_set_global_header with name="{header_name}" and value=<the complete token>. Do NOT describe what you will do — just call the tool.
STEP 3 - After the header is set, call the original API the user wanted.

When an API returns an authentication error (401, 403, "未登录", "需要登录"):
- Ask the user: "这个接口需要登录，请提供{login_hint}。"
- When they provide it, follow STEP 1-3 above, then retry the original API.

## CRITICAL: Tool Usage Rules
- NEVER output text about "I will now call X" — just call the tool directly.
- After getting a token, internal_set_global_header MUST be called in the same turn.
- Token value must be the complete string, no truncation.

"""

    project_dir = config.project_dir
    if project_dir:
        prompt += f"""## Code Analysis & Fix (项目源码: {project_dir})

When an API returns an error (500, validation error, unexpected result), you can:
1. Call internal_search_code to find the relevant controller/service class
2. Call internal_read_code to examine the source code with line numbers
3. Identify the root cause by reading the code logic
4. Explain the bug clearly to the user
5. If the user asks you to fix it, call internal_edit_code with confirmed=false FIRST
   - This shows a preview of the change
   - Wait for the user to reply with "同意修改" or "确认"
6. After approval, call internal_edit_code again with confirmed=true to apply

CRITICAL: NEVER call internal_edit_code with confirmed=true without the user's explicit approval.

"""

    prompt += """## Request Scenario
The current API request scenario is shown at the start. All API calls target this environment.
Use internal_switch_scenario to change environments (e.g. "production" for live servers).

## Global Headers
Current global headers are shown at the start. Use internal_set_global_header to manage them.

## Guidelines
- Use the tool that best matches the user's request.
- After receiving an API response, summarize the result in natural language.
- Always respond in the same language the user uses."""
    return prompt

# ============================================================
# 内部工具定义（添加到外部 API 工具列表中）
# ============================================================
# 这些工具不被发送到后端 API，而是由 agent.py 本地处理。
# 它们给 AI 提供了"元操作"能力：管理认证、切换环境、运行测试。

INTERNAL_TOOLS = [
    # --- 全局请求头管理 ---
    {
        "type": "function",
        "function": {
            "name": "internal_set_global_header",
            "description": (
                "Set a global request header that will be automatically included in ALL subsequent API calls. "
                "Use this after obtaining a login token to persist authentication. "
                "The header will be saved to config.yaml and survive restarts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Header name, e.g. X-Dts-Admin-Token"},
                    "value": {"type": "string", "description": "Header value, e.g. the JWT token string"},
                },
                "required": ["name", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "internal_clear_global_header",
            "description": "Remove a global header so it is no longer sent with API requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Header name to remove"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "internal_list_global_headers",
            "description": "List all currently configured global request headers.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # --- 环境切换 ---
    {
        "type": "function",
        "function": {
            "name": "internal_switch_scenario",
            "description": "Switch the API request target environment (e.g. '切换到线上').",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Scenario name: default, production, etc."},
                },
                "required": ["name"],
            },
        },
    },
    # --- 自动化测试 ---
    {
        "type": "function",
        "function": {
            "name": "internal_run_test",
            "description": (
                "Start automated API testing for a feature. "
                "When called, systematically test all CRUD APIs related to the given feature. "
                "Use this when the user says '测试XX功能'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string", "description": "Feature name, e.g. 资讯评价, 商品管理"},
                },
                "required": ["feature"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "internal_test_step",
            "description": "Report a single test step result during automated testing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step": {"type": "string", "description": "Step: CREATE / QUERY / UPDATE / DELETE"},
                    "api": {"type": "string", "description": "The API endpoint called"},
                    "status": {"type": "string", "description": "pass or fail"},
                    "detail": {"type": "string", "description": "Result detail or error message"},
                },
                "required": ["step", "api", "status", "detail"],
            },
        },
    },
    # --- 源码分析 ---
    {
        "type": "function",
        "function": {
            "name": "internal_search_code",
            "description": (
                "Search the project source code for keywords, class names, method names, or error messages. "
                "Use this to locate the relevant source file when an API returns an error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword, e.g. class name, method name, error message"},
                    "file_pattern": {"type": "string", "description": "Optional glob pattern, e.g. **/*Controller.java"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "internal_read_code",
            "description": "Read a source file from the project directory with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative path in the project directory"},
                    "start_line": {"type": "integer", "description": "Optional start line (1-based)"},
                    "end_line": {"type": "integer", "description": "Optional end line (1-based)"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "internal_edit_code",
            "description": (
                "Propose a code change. On FIRST call, set confirmed=false to show the user what will change. "
                "The tool will return a confirmation prompt. After the user approves, call again with confirmed=true. "
                "Provide the BEFORE code to match and the AFTER code to replace it with."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative path in the project directory"},
                    "old_code": {"type": "string", "description": "Exact code to replace (must match exactly)"},
                    "new_code": {"type": "string", "description": "Replacement code"},
                    "confirmed": {"type": "boolean", "description": "Set false for preview, true to apply"},
                    "summary": {"type": "string", "description": "One-line summary of what this change does (in Chinese)"},
                },
                "required": ["file_path", "old_code", "new_code", "confirmed", "summary"],
            },
        },
    },
]


# ============================================================
# ApiAgent 类
# ============================================================

class ApiAgent:
    """
    核心编排器：管理 AI 对话 + 工具调用循环 + 自动登录。

    关键属性：
      config          - 全局配置（模型、认证、场景等）
      client          - AI 模型客户端（OpenAI/Anthropic/Ollama）
      endpoints       - 所有 API 端点列表
      tool_index      - 工具名 → Endpoint 映射（用于执行工具调用）
      openai_tools    - OpenAI 格式的工具列表（含内部工具）
      anthropic_tools - Anthropic 格式的工具列表
      messages        - 对话历史（AIMessage 列表）
      _custom_params  - 当前对话的私有全局参数（独立于其他对话）
    """

    def __init__(self, config: Config, custom_global_params: list[dict] | None = None,
                 preload_messages: list[dict] | None = None):
        self.config = config
        self.client = AIClientBase.create(config)  # 工厂方法创建 AI 客户端
        self.endpoints: list[Endpoint] = []
        self.tool_index: dict[str, Endpoint] = {}
        self.openai_tools: list[dict] = []
        self.anthropic_tools: list[dict] = []
        self._custom_params = custom_global_params  # 当前对话的私有全局参数
        self._pending_edit: dict | None = None       # 待确认的代码修改

        # 替换系统提示词中的占位符
        scenario = config.api_scenarios.get("active", "default")
        gps = self._global_headers_desc()
        prompt = _build_system_prompt(self.config).replace(
            "The current API request scenario is shown at the start.",
            f"Current scenario: {scenario}"
        ).replace(
            "Current global headers are shown at the start.",
            f"Current global headers: {gps}" if gps else "No global headers are currently set."
        )
        # 对话历史始终以系统提示词开始
        self.messages: list[AIMessage] = [AIMessage(role="system", content=prompt)]

        # 恢复历史消息（从磁盘加载的对话）
        # 必须恢复 tool_calls 和 extra，否则 tool 消息找不到对应的 assistant tool_calls
        if preload_messages:
            for m in preload_messages:
                self.messages.append(AIMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    tool_call_id=m.get("tool_call_id"),
                    tool_calls=m.get("tool_calls"),
                    extra=m.get("extra"),
                ))

    # ---- 属性 ----

    @property
    def active_global_params(self) -> list[dict]:
        """
        获取当前有效的全局参数。

        优先级：对话私有参数 > 配置文件全局参数。
        这样每个对话可以有独立的 token，互不干扰。
        """
        if self._custom_params is not None:
            return self._custom_params
        return self.config.global_params

    def _global_headers_desc(self) -> str:
        """生成全局请求头的简短描述（展示给 AI）"""
        gps = [p for p in self.active_global_params if p.get("type") == "header"]
        if not gps:
            return ""
        return ", ".join(f"{p['name']}={p['value'][:20]}..." for p in gps)

    @property
    def is_ready(self) -> bool:
        """Agent 是否就绪（至少有一个 API 端点被加载）"""
        return len(self.endpoints) > 0

    @property
    def tools_for_model(self) -> list[dict]:
        """根据配置的模型提供商返回对应格式的工具列表"""
        provider = self.config.model.get("provider", "openai")
        if provider == "anthropic":
            return self.anthropic_tools
        return self.openai_tools

    # ---- 工具加载 ----

    def load_tools(self, endpoints: list[Endpoint]) -> None:
        """
        加载 API 端点为 AI 工具。

        分为两步：
          1. 将 Endpoint 列表转换为 LLM function-calling 格式
          2. 追加内部工具（internal_*）到工具列表
        """
        from src.tool_converter import (
            build_tool_index,
            convert_all_to_anthropic,
            convert_all_to_openai,
        )

        self.endpoints = endpoints
        self.tool_index = build_tool_index(endpoints)
        # 外部 API 工具 + 内部工具合并
        self.openai_tools = convert_all_to_openai(endpoints) + INTERNAL_TOOLS
        self.anthropic_tools = convert_all_to_anthropic(endpoints) + [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in INTERNAL_TOOLS
        ]

    # ---- 工具调用处理 ----

    def _handle_internal_tool(self, tc: ToolCall) -> str:
        """
        执行内部工具（以 "internal_" 开头的工具名）。

        内部工具不发送 HTTP 请求，而是直接操作本地状态：
          - 修改全局参数（_custom_params）
          - 切换场景（config.update）
          - 返回测试指令
        """
        name = tc.name
        args = tc.arguments

        if name == "internal_switch_scenario":
            sname = args.get("name", "")
            scenarios = self.config.api_scenarios
            valid = [s["name"] for s in scenarios.get("list", [])]
            if sname not in valid:
                return json.dumps({"error": f"Unknown scenario '{sname}'. Available: {valid}"})
            # 更新配置文件中的激活场景
            self.config.update("api_scenarios", {**scenarios, "active": sname})
            return json.dumps({"status": "ok", "message": f"Switched to scenario '{sname}'."})

        elif name == "internal_set_global_header":
            hname = args.get("name", "")
            hvalue = args.get("value", "")
            if not hname:
                return json.dumps({"error": "header name is required"})
            # 更新当前对话的私有参数（不影响其他对话）
            if self._custom_params is None:
                self._custom_params = []
            gps = [p for p in self._custom_params if p.get("type") != "header" or p.get("name") != hname]
            gps.append({"name": hname, "value": hvalue, "type": "header"})
            self._custom_params = gps
            return json.dumps({"status": "ok", "message": f"Global header '{hname}' has been set."})

        elif name == "internal_clear_global_header":
            hname = args.get("name", "")
            if self._custom_params is not None:
                self._custom_params = [p for p in self._custom_params if p.get("name") != hname]
            return json.dumps({"status": "ok", "message": f"Global header '{hname}' has been removed."})

        elif name == "internal_list_global_headers":
            gps = [p for p in self.active_global_params if p.get("type") == "header"]
            return json.dumps(gps if gps else [], ensure_ascii=False)

        elif name == "internal_run_test":
            # 返回测试指令，引导 AI 执行 CRUD 测试流程
            feature = args.get("feature", "")
            return json.dumps({
                "status": "ok",
                "message": f"开始测试「{feature}」功能。请按以下步骤执行：\n"
                           f"1. 查找与「{feature}」相关的 CREATE/POST 接口，生成测试数据并调用\n"
                           f"2. 调用 internal_test_step 报告创建结果\n"
                           f"3. 查找 QUERY/GET 接口，查询刚创建的数据\n"
                           f"4. 调用 internal_test_step 报告查询结果\n"
                           f"5. 查找 UPDATE/PUT 接口，修改数据\n"
                           f"6. 调用 internal_test_step 报告更新结果\n"
                           f"7. 查找 DELETE 接口，删除测试数据\n"
                           f"8. 调用 internal_test_step 报告删除结果\n"
                           f"9. 汇总测试结果，输出测试报告",
            })

        elif name == "internal_test_step":
            return json.dumps({
                "status": "ok",
                "logged": f"[{args.get('step', '?')}] {args.get('api', '?')}: "
                          f"{args.get('status', '?')} - {args.get('detail', '')}",
            })

        elif name == "internal_search_code":
            import fnmatch, os as _os
            query = args.get("query", "")
            pattern = args.get("file_pattern", "**/*.java")
            proj_dir = self.config.project_dir
            if not proj_dir or not _os.path.isdir(proj_dir):
                return json.dumps({"error": "project_dir not configured or not found"})
            results = []
            import subprocess, shutil
            # 优先用 grep -rn 快速搜索
            rg = shutil.which("rg") or shutil.which("grep")
            if rg:
                try:
                    cmd = [rg, "-rn", "--include=" + pattern.replace("**/", ""), query, proj_dir]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=proj_dir)
                    lines = proc.stdout.strip().split("\n")[:50]
                    for line in lines:
                        parts = line.split(":", 2)
                        if len(parts) >= 3:
                            results.append({"file": parts[0], "line": parts[1], "content": parts[2][:200]})
                except Exception:
                    pass
            if not results:
                for root, dirs, files in _os.walk(proj_dir):
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "target", ".git", "__pycache__")]
                    for f in files:
                        if fnmatch.fnmatch(f, pattern.replace("**/", "")) or pattern == "**/*.java":
                            fpath = _os.path.join(root, f)
                            try:
                                content = open(fpath, encoding="utf-8", errors="ignore").read()
                                for i, line in enumerate(content.split("\n"), 1):
                                    if query.lower() in line.lower():
                                        results.append({"file": _os.path.relpath(fpath, proj_dir), "line": str(i), "content": line.strip()[:200]})
                                        if len(results) >= 50:
                                            break
                            except Exception:
                                pass
                            if len(results) >= 50:
                                break
                    if len(results) >= 50:
                        break
            if not results:
                return json.dumps({"results": [], "message": f"No matches for '{query}' in {proj_dir}"})
            return json.dumps({"results": results[:50], "count": len(results), "project_dir": proj_dir}, ensure_ascii=False)

        elif name == "internal_read_code":
            import os as _os
            proj_dir = self.config.project_dir
            if not proj_dir:
                return json.dumps({"error": "project_dir not configured"})
            file_path = args.get("file_path", "")
            full_path = _os.path.join(proj_dir, file_path)
            if not _os.path.isfile(full_path) or ".." in file_path:
                return json.dumps({"error": f"File not found: {file_path}"})
            try:
                content = open(full_path, encoding="utf-8", errors="ignore").read()
                lines = content.split("\n")
                start = max(1, int(args.get("start_line", 1) or 1))
                end = min(len(lines), int(args.get("end_line", len(lines)) or len(lines)))
                result = [f"{i}: {lines[i-1]}" for i in range(start, end + 1)]
                return json.dumps({
                    "file": file_path,
                    "total_lines": len(lines),
                    "start_line": start,
                    "end_line": end,
                    "code": "\n".join(result),
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)})

        elif name == "internal_edit_code":
            import os as _os
            proj_dir = self.config.project_dir
            if not proj_dir:
                return json.dumps({"error": "project_dir not configured"})
            file_path = args.get("file_path", "")
            full_path = _os.path.join(proj_dir, file_path)
            if not _os.path.isfile(full_path) or ".." in file_path:
                return json.dumps({"error": f"File not found: {file_path}"})
            old_code = args.get("old_code", "")
            new_code = args.get("new_code", "")
            confirmed = args.get("confirmed", False)
            summary = args.get("summary", "No summary")

            if not confirmed:
                # Preview mode: store pending edit and ask for confirmation
                self._pending_edit = {
                    "file_path": file_path,
                    "old_code": old_code,
                    "new_code": new_code,
                    "summary": summary,
                }
                diff = f"- {old_code[:120]}...\n+ {new_code[:120]}..."
                return json.dumps({
                    "status": "pending_confirmation",
                    "message": f"确认修改 {file_path}?\n修改摘要: {summary}\n\n修改内容:\n{diff}",
                    "instruction": "请用户回复'同意修改'或'确认'来应用此修改，或'取消'来放弃。",
                }, ensure_ascii=False)

            # Apply mode
            try:
                content = open(full_path, encoding="utf-8", errors="ignore").read()
                if old_code not in content:
                    return json.dumps({"error": "old_code not found in file. File may have changed."})
                new_content = content.replace(old_code, new_code, 1)
                open(full_path, "w", encoding="utf-8").write(new_content)
                self._pending_edit = None
                return json.dumps({"status": "ok", "message": f"已修改 {file_path}: {summary}"}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)})

        return json.dumps({"error": f"Unknown internal tool: {name}"})

    def _convert_tool_calls_to_openai(self, tool_calls: list[ToolCall]) -> list[dict]:
        """
        将内部 ToolCall 列表转换为 OpenAI API 的 tool_calls 格式。

        OpenAI 期望的格式：
          {"id": "call_xxx", "type": "function", "function": {"name": "...", "arguments": "{...}"}}
        """
        result = []
        for tc in tool_calls:
            result.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            })
        return result

    def _record_assistant_tool_call(self, response: AIResponse):
        """将 AI 的工具调用请求记录到对话历史中"""
        provider = self.config.model.get("provider", "openai")
        if provider == "anthropic":
            tool_call_dicts = [
                {"id": tc.id, "name": tc.name, "input": tc.arguments}
                for tc in response.tool_calls
            ]
            self.messages.append(AIMessage(
                role="assistant", content=response.content or "",
                tool_calls=tool_call_dicts,
            ))
        else:
            self.messages.append(AIMessage(
                role="assistant", content=response.content or "",
                tool_calls=self._convert_tool_calls_to_openai(response.tool_calls),
                extra=response.extra,  # 保留 reasoning_content
            ))

    async def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[str]:
        """
        执行一批工具调用。

        内部工具由 _handle_internal_tool 本地执行；
        外部工具委托给 api_executor.execute_tool_call 发送 HTTP 请求。
        """
        results = []
        for tc in tool_calls:
            if tc.name.startswith("internal_"):
                results.append(self._handle_internal_tool(tc))
            else:
                results.append(await execute_tool_call(
                    tc.name, tc.arguments, self.tool_index, self.config,
                    override_params=self.active_global_params,  # 对话私有全局参数
                ))
        return results

    # ---- 核心对话方法 ----

    async def chat(self, user_input: str) -> str:
        """
        处理用户消息，返回 AI 完整回复（非流式，用于 CLI 模式）。

        核心循环：
          while AI 返回 tool_calls:
              执行工具 → 追加结果 → 再次调用 AI
          return AI 文本回复
        """
        self.messages.append(AIMessage(role="user", content=user_input))

        for _ in range(30):  # 最多 20 轮工具调用（防止死循环）
            response = await self.client.chat(self.messages, self.tools_for_model)

            if not response.tool_calls:
                # AI 认为不需要更多工具 → 返回纯文本
                content = response.content or "I'm sorry, I couldn't generate a response."
                self.messages.append(AIMessage(role="assistant", content=content))
                return content

            # 记录 AI 的工具调用请求
            self._record_assistant_tool_call(response)
            # 执行工具
            results = await self._execute_tool_calls(response.tool_calls)
            # 将每个工具的执行结果追加到对话
            for tc, result in zip(response.tool_calls, results):
                self.messages.append(AIMessage(
                    role="tool", content=result, tool_call_id=tc.id,
                ))

        return "Reached maximum tool call iterations."

    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        处理用户消息，流式返回 AI 回复（用于 Web UI 的 SSE 展示）。

        与 chat() 的区别：
          - 工具执行过程中发送进度提示（"[调用接口: xxx...]"）
          - 最终回复用 stream 逐 token 返回

        核心循环与 chat() 相同：while tool_calls → execute → loop
        """
        self.messages.append(AIMessage(role="user", content=user_input))

        for _ in range(30):
            response = await self.client.chat(self.messages, self.tools_for_model)

            if not response.tool_calls:
                # 最终回复 → 流式输出
                if response.content:
                    yield response.content
                self.messages.append(AIMessage(role="assistant", content=response.content or ""))
                return

            self._record_assistant_tool_call(response)

            for tc in response.tool_calls:
                # 进度提示（区分内部操作和 API 调用）
                if tc.name.startswith("internal_"):
                    yield f"\n[内部操作: {tc.name}...]\n"
                else:
                    yield f"\n[调用接口: {tc.name}...]\n"

                # 执行工具
                if tc.name.startswith("internal_"):
                    result = self._handle_internal_tool(tc)
                else:
                    result = await execute_tool_call(
                        tc.name, tc.arguments, self.tool_index, self.config,
                        override_params=self.active_global_params,
                    )
                self.messages.append(AIMessage(
                    role="tool", content=result, tool_call_id=tc.id,
                ))

        yield "\n已达到最大工具调用次数。"

    # ---- 对话管理 ----

    def export_messages(self) -> list[dict]:
        """
        导出对话消息（排除系统提示词），用于保存到磁盘。

        必须保存 tool_calls 和 extra 字段，否则恢复后 OpenAI API 会报错：
        "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"
        """
        result = []
        for m in self.messages:
            if m.role == "system":
                continue
            item: dict = {"role": m.role, "content": m.content, "tool_call_id": m.tool_call_id}
            if m.tool_calls:
                item["tool_calls"] = m.tool_calls
            if m.extra:
                item["extra"] = m.extra
            result.append(item)
        return result

    def get_custom_params(self) -> list[dict]:
        """返回当前对话的私有全局参数"""
        return self._custom_params if self._custom_params is not None else []

    def clear_history(self):
        """清除对话历史（保留系统提示词和全局参数）"""
        scenario = self.config.api_scenarios.get("active", "default")
        gps = self._global_headers_desc()
        prompt = _build_system_prompt(self.config).replace(
            "The current API request scenario is shown at the start.",
            f"Current scenario: {scenario}"
        ).replace(
            "Current global headers are shown at the start.",
            f"Current global headers: {gps}" if gps else "No global headers are currently set."
        )
        self.messages = [AIMessage(role="system", content=prompt)]

    def get_tool_summary(self) -> str:
        """
        生成可用 API 工具的摘要（CLI 模式下展示用）。

        按 tag 分组显示，每个端点一行：METHOD PATH — SUMMARY
        """
        lines = []
        groups: dict[str, list[Endpoint]] = {}
        for ep in self.endpoints:
            tag = ep.tags[0] if ep.tags else "Other"
            groups.setdefault(tag, []).append(ep)

        for tag, eps in groups.items():
            lines.append(f"\n## {tag}")
            for ep in eps:
                lines.append(f"  {ep.method:6} {ep.path}")
                if ep.summary:
                    lines.append(f"        {ep.summary}")
        return "\n".join(lines)
