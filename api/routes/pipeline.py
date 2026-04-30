from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Query

from ..config import Settings, get_settings
from ..dependencies import get_store
from ..storage import DemoStore

logger = logging.getLogger("neuropod.pipeline")

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "running": False,
    "started_at": None,
    "completed_at": None,
    "error": None,
    "last_error": None,
    "last_count": 0,
    "window_days": None,
}


def _set(**kwargs: Any) -> None:
    with _state_lock:
        _state.update(kwargs)


def _run_pipeline(store: DemoStore, topics: list[str], count: int, window_days: int) -> None:
    _set(running=True, started_at=datetime.now(timezone.utc).isoformat(), error=None, window_days=window_days)
    try:
        result = store.run_pipeline(topics=topics, episode_count=count, window_days=window_days)
        _set(
            running=False,
            completed_at=datetime.now(timezone.utc).isoformat(),
            last_count=len(result.get("episodes", [])),
            last_error=None,
        )
    except Exception as exc:
        message = str(exc)
        logger.exception("pipeline run failed")
        _set(
            running=False,
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=message,
            last_error=message,
        )


@router.post("/run")
def trigger_run(
    background: BackgroundTasks,
    window: int | None = Query(default=None, ge=1, le=365, description="Discovery window in days"),
    store: DemoStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    with _state_lock:
        already_running = _state["running"]
    if already_running:
        return {"queued": False, "running": True, "state": dict(_state)}

    topics = store.get_topics() or settings.default_topics
    window_days = window if window is not None else settings.discovery_window_days
    background.add_task(_run_pipeline, store, topics, settings.default_episode_count, window_days)
    return {"queued": True, "running": True, "window_days": window_days}


@router.get("/state")
def get_state() -> dict[str, Any]:
    with _state_lock:
        return dict(_state)
