from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import init_schema
from .routes.ask import router as ask_router
from .routes.auth import router as auth_router
from .routes.episodes import router as episodes_router
from .routes.feed import router as feed_router
from .routes.feedback import router as feedback_router
from .routes.me import router as me_router
from .routes.pipeline import router as pipeline_router
from .routes.status import router as status_router
from .routes.topics import router as topics_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema()
    yield


app = FastAPI(
    title="Neuropod API",
    description="Multi-user research-paper-to-podcast pipeline (BYOK).",
    version="0.3.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(me_router)
app.include_router(episodes_router)
app.include_router(topics_router)
app.include_router(feed_router)
app.include_router(ask_router)
app.include_router(feedback_router)
app.include_router(status_router)
app.include_router(pipeline_router)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
