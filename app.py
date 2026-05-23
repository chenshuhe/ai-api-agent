"""
FastAPI Web 服务入口
====================
提供 Web UI 和 REST API，负责协调前端、Agent、对话存储之间的交互。

路由一览：
  GET  /                    前端页面
  GET  /api/status          运行状态
  GET  /api/tools           可用 API 工具列表
  POST /api/reload          重新加载 API 文档
  POST /api/chat            发送消息（SSE 流式响应）
  POST /api/clear           清除当前对话
  GET  /api/config          读取配置
  POST /api/config          更新配置
  GET  /api/conversations   对话列表
  POST /api/conversations   创建对话
  POST /api/conversations/switch  切换对话
  DELETE /api/conversations/{id}  删除对话

关键设计：
  - 延迟加载：API 文档在首次请求时加载，不阻塞启动
  - 启动预加载：lifespan 事件中后台加载 agent，首页即可显示就绪
  - 对话持久化：每次 AI 回复后自动保存当前对话到磁盘
  - SSE 流式：/api/chat 使用 Server-Sent Events 实时推送 AI 回复
"""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api_docs import get_all_endpoints, load_all_api_docs
from src.agent import ApiAgent
from src.config import Config, get_config, reset_config
from src.conversation_store import Conversation, ConversationStore

# ---- 全局状态 ----
# 这些模块级变量在整个服务生命周期中共享

_config = get_config()                          # 全局配置单例
_store = ConversationStore()                    # 对话存储管理器
_current_conv: Conversation | None = None       # 当前激活的对话
_agent: ApiAgent | None = None                  # 当前 Agent 实例
_lock = asyncio.Lock()                          # 防并发加载锁
_loading = False                                # 是否正在加载
_load_error: str | None = None                  # 加载错误信息


# ---- Agent 生命周期管理 ----

def _get_or_create_agent(conv: Conversation) -> ApiAgent:
    """
    为指定对话创建 Agent 实例。

    每个对话有独立的：
      - global_params（token 等）
      - messages（对话历史）
    """
    global _agent
    endpoints = getattr(_agent, "endpoints", []) if _agent else []
    agent = ApiAgent(
        _config,
        custom_global_params=conv.global_params if conv.global_params else None,
        preload_messages=conv.messages,
    )
    if endpoints:
        agent.load_tools(endpoints)
    return agent


async def _load_agent() -> ApiAgent:
    """
    加载 Agent（线程安全，带防重入锁）。

    如果 Agent 已存在则直接返回；
    如果正在加载中则等待；
    否则创建新 Agent。
    """
    global _agent, _loading, _load_error, _current_conv
    async with _lock:
        if _agent is not None:
            return _agent
        if _loading:
            # 另一个请求正在加载，等待它完成
            for _ in range(100):
                await asyncio.sleep(0.1)
                if _agent is not None:
                    return _agent
        _loading = True
        _load_error = None
        try:
            # 确保有默认对话
            if _current_conv is None:
                _current_conv = _store.get_or_create_default()

            # 并发抓取所有 API 文档
            docs = await load_all_api_docs(
                _config.api_docs_urls, _config.api_docs_timeout
            )
            endpoints = get_all_endpoints(docs)
            # 创建 Agent（注入对话专属参数和历史）
            _agent = ApiAgent(
                _config,
                custom_global_params=_current_conv.global_params or None,
                preload_messages=_current_conv.messages,
            )
            _agent.load_tools(endpoints)
            return _agent
        except Exception as e:
            _load_error = str(e)
            return None
        finally:
            _loading = False


def _save_current_conv():
    """
    将 Agent 当前状态写回对话存储。

    在每次 AI 回复后调用，确保：
      - 对话历史被持久化
      - 全局参数（token 等）随对话保存
    """
    global _agent, _current_conv
    if _agent is None or _current_conv is None:
        return
    _current_conv.messages = _agent.export_messages()
    _current_conv.global_params = _agent.get_custom_params()
    _store.save(_current_conv)


# ---- FastAPI 应用 ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理。

    启动时后台加载 Agent，这样用户打开页面时已经显示 🟢 就绪。
    """
    asyncio.create_task(_load_agent())  # 不阻塞启动
    yield


app = FastAPI(title="AI API Agent", lifespan=lifespan)
# 挂载 web/ 目录为静态文件（app.js 等）
app.mount("/static", StaticFiles(directory="web"), name="static")


# ============================================================
# 页面
# ============================================================

@app.get("/")
async def index():
    """返回前端聊天界面"""
    return HTMLResponse(Path("web/index.html").read_text(encoding="utf-8"))


# ============================================================
# 对话管理 API
# ============================================================

@app.get("/api/conversations")
async def list_conversations():
    """列出所有对话及其摘要信息"""
    convs = _store.list_all()
    current_id = _current_conv.id if _current_conv else ""
    return {"conversations": convs, "current_id": current_id}


@app.post("/api/conversations")
async def create_conversation(request: Request):
    """
    创建新对话。

    流程：
      1. 保存当前对话
      2. 创建新对话（空历史、空参数）
      3. 立即加载 Agent 到新对话
    """
    global _agent, _current_conv
    body = await request.json()
    name = body.get("name", "新对话").strip() or "新对话"

    _save_current_conv()
    conv = _store.create(name)
    _current_conv = conv
    _agent = None
    await _load_agent()  # 立即加载，状态立即可用

    return {"status": "ok", "conversation": conv.to_dict()}


@app.post("/api/conversations/switch")
async def switch_conversation(request: Request):
    """切换到指定对话"""
    global _agent, _current_conv
    body = await request.json()
    conv_id = body.get("id", "")

    conv = _store.get(conv_id)
    if not conv:
        return {"status": "error", "message": "Conversation not found"}

    _save_current_conv()
    _current_conv = conv
    _agent = None
    await _load_agent()  # 立即加载新对话的 Agent

    return {"status": "ok", "conversation": conv.to_dict()}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """
    删除对话。

    如果删除的是当前对话，自动回退到默认对话。
    """
    global _agent, _current_conv
    if _store.delete(conv_id):
        if _current_conv and _current_conv.id == conv_id:
            _save_current_conv()
            _current_conv = _store.get_or_create_default()
            _agent = None
            await _load_agent()
        return {"status": "ok"}
    return {"status": "error", "message": "Not found"}


# ============================================================
# 状态 & 工具 API
# ============================================================

@app.get("/api/tools")
async def list_tools():
    """
    返回可用 API 工具列表。

    前端侧边栏用此数据显示可用接口。
    内部工具（internal_*）被过滤掉，不展示给用户。
    """
    agent = await _load_agent()
    if agent is None:
        return {"count": 0, "load_error": _load_error or "Agent init failed", "tools": []}
    return {
        "count": len(agent.endpoints),
        "load_error": _load_error,
        "tools": [
            {"name": t.get("function", {}).get("name", ""),
             "description": t.get("function", {}).get("description", "")}
            for t in agent.openai_tools
            if not t.get("function", {}).get("name", "").startswith("internal_")
        ],
    }


@app.get("/api/status")
async def status():
    """返回运行状态（前端轮询用）"""
    return {
        "ready": _agent is not None and _agent.is_ready,
        "endpoints": len(_agent.endpoints) if _agent else 0,
        "model": _config.model.get("name"),
        "provider": _config.model.get("provider"),
        "loading": _loading,
        "load_error": _load_error,
        "current_conv": _current_conv.name if _current_conv else "",
    }


@app.post("/api/reload")
async def reload_docs():
    """
    强制重新加载 API 文档。

    用户修改文档地址或场景映射后调用此接口刷新。
    """
    global _agent, _load_error
    _save_current_conv()
    _agent = None
    _load_error = None
    agent = await _load_agent()
    if agent is None:
        return {"status": "error", "endpoints": 0, "load_error": _load_error or "Agent init failed"}
    return {"status": "ok", "endpoints": len(agent.endpoints), "load_error": _load_error}


# ============================================================
# 对话 API（核心）
# ============================================================

def _error_stream(msg: str):
    """生成器：返回一个只包含错误消息的 SSE 流"""
    async def gen():
        yield f"data: {json.dumps({'chunk': msg})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/chat")
async def chat(request: Request):
    """
    发送消息并获取 AI 回复（Server-Sent Events 流式）。

    SSE 数据格式：
      {"chunk": "文本片段"}     — 流式内容（逐 token）
      {"done": true, "full": "..."} — 完成标志
      {"error": "..."}         — 错误

    前端 EventSource 或 fetch + ReadableStream 接收。
    """
    body = await request.json()
    user_input = body.get("message", "").strip()
    if not user_input:
        return {"error": "Empty message"}

    agent = await _load_agent()
    if agent is None:
        return _error_stream(f"Agent initialization failed: {_load_error}")
    if not agent.is_ready:
        return _error_stream(f"No API endpoints loaded: {_load_error}")

    async def generate():
        full_text = ""
        try:
            async for chunk in agent.chat_stream(user_input):
                full_text += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full': full_text})}\n\n"
            _save_current_conv()  # 自动保存对话到磁盘
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/clear")
async def clear_history():
    """清除当前对话的历史记录"""
    agent = await _load_agent()
    if agent is None:
        return {"status": "error", "message": _load_error or "Agent not initialized"}
    agent.clear_history()
    _save_current_conv()
    return {"status": "ok"}


# ============================================================
# 配置 API
# ============================================================

@app.get("/api/config")
async def get_full_config():
    """返回完整配置（前端配置面板使用）"""
    raw = _config.raw_data()
    return {
        "api_docs": {
            "urls": raw.get("api_docs", {}).get("urls", []),
            "timeout": raw.get("api_docs", {}).get("timeout", 30),
        },
        "model": raw.get("model", {}),
        "api_auth": raw.get("api_auth", {}),
        "api_scenarios": {
            **raw.get("api_scenarios", {"active": "default", "list": []}),
            "_active": raw.get("api_scenarios", {}).get("active", "default"),
        },
        "global_params": raw.get("global_params", []),
        "project_dir": raw.get("project_dir", ""),
        "auto_login": raw.get("auto_login", {}),
    }


@app.post("/api/config")
async def update_config(request: Request):
    """
    更新配置（前端配置面板使用）。

    支持部分更新：只需传要修改的 section。
    更新后自动重置 Agent 以应用新配置。
    """
    body = await request.json()
    updated = []

    if "api_docs" in body:
        _config.update("api_docs", body["api_docs"])
        updated.append("api_docs")
    if "model" in body:
        _config.update("model", body["model"])
        updated.append("model")
    if "api_auth" in body:
        _config.update("api_auth", body["api_auth"])
        updated.append("api_auth")
    if "api_scenarios" in body:
        _config.update("api_scenarios", body["api_scenarios"])
        updated.append("api_scenarios")
    if "global_params" in body:
        _config.update("global_params", body["global_params"])
        updated.append("global_params")
    if "project_dir" in body:
        _config.update("project_dir", body["project_dir"])
        updated.append("project_dir")
    if "auto_login" in body:
        _config.update("auto_login", body["auto_login"])
        updated.append("auto_login")

    if updated:
        global _agent, _load_error
        _agent = None
        _load_error = None
        reset_config()   # 强制重新加载配置单例
        _config.reload()

    return {"status": "ok", "updated": updated}


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    host = _config.server.get("host", "0.0.0.0")
    port = _config.server.get("port", 8000)
    uvicorn.run(app, host=host, port=port)
