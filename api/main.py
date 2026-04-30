from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .dependencies import get_store
from .routes.ask import router as ask_router
from .routes.episodes import router as episodes_router
from .routes.feedback import router as feedback_router
from .routes.feed import router as feed_router
from .routes.pipeline import router as pipeline_router
from .routes.status import router as status_router
from .routes.topics import router as topics_router


def _maybe_start_scheduler(app: FastAPI) -> None:
    settings = get_settings()
    if not settings.enable_scheduler:
        app.state.scheduler = None
        return

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except Exception:
        app.state.scheduler = None
        return

    store = get_store()
    scheduler = BackgroundScheduler(timezone=settings.scheduler_timezone)
    scheduler.add_job(
        lambda: store.run_pipeline(store.get_topics(), settings.default_episode_count, settings.discovery_window_days),
        trigger="interval",
        hours=24,
        id="neuropod-daily-pipeline",
        replace_existing=True,
    )
    scheduler.start()
    app.state.scheduler = scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = get_store()
    store.ensure_seeded(settings.default_topics, settings.default_episode_count, settings.discovery_window_days)
    _maybe_start_scheduler(app)
    yield
    scheduler = getattr(app.state, "scheduler", None)
    if scheduler:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="Neuropod API",
    description="Research-paper-to-podcast pipeline.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
