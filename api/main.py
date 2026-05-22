from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .db import cursor, init_schema
from .observability import RequestIdMiddleware, configure_logging
from .routes.ask import router as ask_router
from .routes.auth import router as auth_router
from .routes.episodes import router as episodes_router
from .routes.feed import router as feed_router
from .routes.feedback import router as feedback_router
from .routes.me import router as me_router
from .routes.pipeline import router as pipeline_router
from .routes.status import router as status_router
from .routes.topics import router as topics_router


configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema()
    yield


app = FastAPI(
    title="Neuropod API",
    description="Multi-user research-paper RAG pipeline with optional podcast audio.",
    version="0.4.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)
app.add_middleware(RequestIdMiddleware)

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
def liveness() -> dict[str, str]:
    """Liveness probe - returns 200 as long as the process is running."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": app.version,
    }


@app.get("/healthz")
def readiness() -> JSONResponse:
    """Readiness probe - verifies the database is reachable.

    Use this for load-balancer health checks. Returns 503 if the DB is down so
    AWS can stop routing traffic to a pod that can't serve real requests."""
    try:
        with cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "error": str(exc)[:200]},
        )
    return JSONResponse(content={"status": "ok"})
