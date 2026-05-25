"""FastAPI application entry point.

Usage:
    python -m ai_agent.app
    uvicorn ai_agent.app:app --reload
"""

from .api.routes import create_app
from .settings import get_settings

app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)
