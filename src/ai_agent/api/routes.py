"""FastAPI routes for the AI Agent."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from ..agent.graph import ApiAgent
from ..conversations.store import Conversation, ConversationStore
from ..settings import Settings, get_settings, reload_settings
from .schemas import ChatRequest, ConfigUpdate, ConversationCreate, ConversationSwitch

# Global state
_settings = get_settings()
_store = ConversationStore()
_agent: ApiAgent | None = None
_current_conv: Conversation | None = None
_lock = asyncio.Lock()
_loading = False
_load_error: str | None = None


async def _load_agent() -> ApiAgent | None:
    global _agent, _loading, _load_error, _current_conv
    async with _lock:
        if _agent is not None:
            return _agent
        if _loading:
            for _ in range(100):
                await asyncio.sleep(0.1)
                if _agent is not None:
                    return _agent
        _loading = True
        _load_error = None
        try:
            if _current_conv is None:
                _current_conv = _store.get_or_create_default()
            _agent = ApiAgent(_settings, custom_params=_current_conv.global_params)
            await _agent.load_endpoints()
            _agent.build_graph()
            return _agent
        except Exception as e:
            _load_error = str(e)
            logger.error(f"Agent load failed: {e}")
            return None
        finally:
            _loading = False


def _save_current_conv():
    if _agent is None or _current_conv is None:
        return
    _current_conv.global_params = _agent._custom_params


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        asyncio.create_task(_load_agent())
        yield

    app = FastAPI(title="AI API Agent v2", lifespan=lifespan)
    web_dir = Path("web")
    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory="web"), name="static")

    @app.get("/")
    async def index():
        html_path = Path("web/index.html")
        return HTMLResponse(html_path.read_text(encoding="utf-8")) if html_path.exists() else HTMLResponse("<h1>Web UI not found</h1>")

    # ---- Conversations ----
    @app.get("/api/conversations")
    async def list_conversations():
        convs = _store.list_all()
        return {"conversations": convs, "current_id": _current_conv.id if _current_conv else ""}

    @app.post("/api/conversations")
    async def create_conversation(req: ConversationCreate):
        global _agent, _current_conv
        _save_current_conv()
        conv = _store.create(req.name)
        _current_conv = conv
        _agent = None
        await _load_agent()
        return {"status": "ok", "conversation": conv.to_dict()}

    @app.post("/api/conversations/switch")
    async def switch_conversation(req: ConversationSwitch):
        global _agent, _current_conv
        conv = _store.get(req.id)
        if not conv:
            return {"status": "error", "message": "Not found"}
        _save_current_conv()
        _current_conv = conv
        _agent = None
        await _load_agent()
        return {"status": "ok", "conversation": conv.to_dict()}

    @app.delete("/api/conversations/{conv_id}")
    async def delete_conversation(conv_id: str):
        global _agent, _current_conv
        if _store.delete(conv_id):
            if _current_conv and _current_conv.id == conv_id:
                _current_conv = _store.get_or_create_default()
                _agent = None
                await _load_agent()
            return {"status": "ok"}
        return {"status": "error", "message": "Not found"}

    # ---- Status & Tools ----
    @app.get("/api/status")
    async def status():
        return {
            "ready": _agent is not None and _agent.endpoint_count > 0,
            "endpoints": _agent.endpoint_count if _agent else 0,
            "model": _settings.model.name,
            "provider": _settings.model.provider,
            "loading": _loading,
            "load_error": _load_error,
            "current_conv": _current_conv.name if _current_conv else "",
        }

    @app.get("/api/tools")
    async def list_tools():
        agent = await _load_agent()
        if agent is None:
            return {"count": 0, "tools": []}
        api_tools = [t for t in agent.all_tools if not t.name.startswith("internal_")]
        return {"count": len(api_tools), "tools": [{"name": t.name, "description": t.description} for t in api_tools]}

    @app.post("/api/reload")
    async def reload_docs():
        global _agent, _load_error
        _save_current_conv()
        _agent = None
        agent = await _load_agent()
        if agent is None:
            return {"status": "error", "message": _load_error}
        return {"status": "ok", "endpoints": agent.endpoint_count}

    # ---- Chat ----
    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        agent = await _load_agent()
        if agent is None:
            async def err():
                yield f"data: {json.dumps({'chunk': f'Error: {_load_error}'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            return StreamingResponse(err(), media_type="text/event-stream")

        tid = _current_conv.id if _current_conv else "default"

        async def generate():
            text = ""
            try:
                async for chunk in agent.chat_stream(req.message, tid):
                    text += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                yield f"data: {json.dumps({'done': True, 'full': text})}\n\n"
                _save_current_conv()
                _store.save(_current_conv)
            except Exception as e:
                logger.error(f"Chat error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/api/clear")
    async def clear():
        global _agent
        agent = await _load_agent()
        if agent:
            tid = _current_conv.id if _current_conv else "default"
            agent.clear_history(tid)
            agent.set_thread_id(tid)
            _save_current_conv()
            _store.save(_current_conv)
        return {"status": "ok"}

    # ---- Config ----
    @app.get("/api/config")
    async def get_config():
        s = _settings
        return {
            "api_docs": {"urls": s.api_docs.urls, "timeout": s.api_docs.timeout},
            "model": s.model.model_dump(),
            "api_auth": s.api_auth.model_dump(),
            "api_scenarios": {"active": s.api_scenarios.active, "list": s.api_scenarios.scenarios, "_active": s.api_scenarios.active},
            "auto_login": s.auto_login.model_dump(),
            "project_dir": s.project_dir,
            "global_params": [p.model_dump() for p in s.global_params],
        }

    @app.post("/api/config")
    async def update_config(req: ConfigUpdate):
        global _agent, _settings
        updated = []
        data = req.model_dump(exclude_none=True)
        for key in data:
            if hasattr(_settings, key):
                setattr(_settings, key, data[key])
                updated.append(key)
        if updated:
            _settings.save_to_yaml()
            reload_settings()
            _agent = None
        return {"status": "ok", "updated": updated}

    return app
