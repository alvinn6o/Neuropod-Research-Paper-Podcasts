from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from .. import store_db
from ..auth import AuthUser, CurrentUser
from ..config import get_settings
from ..models import JobResponse
from ..pipeline_runner import run_for_user

logger = logging.getLogger("neuropod.pipeline")

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_active_jobs: set[uuid.UUID] = set()
_active_lock = threading.Lock()


def _run_job(job_id: uuid.UUID, user_id: uuid.UUID, topics: list[str], episode_count: int, window: int) -> None:
    with _active_lock:
        _active_jobs.add(job_id)
    store_db.update_job(job_id, status="running", started_at=True)
    try:
        result = run_for_user(
            user_id,
            topics=topics,
            episode_count=episode_count,
            window_days=window,
            synthesize_audio=get_settings().generate_audio_on_pipeline,
        )
        store_db.update_job(
            job_id,
            status="done",
            finished_at=True,
            result_count=result["result_count"],
            skipped_count=result.get("skipped_existing", 0),
        )
    except Exception as exc:
        logger.exception("pipeline job %s failed", job_id)
        store_db.update_job(job_id, status="error", finished_at=True, error=str(exc)[:500])
    finally:
        with _active_lock:
            _active_jobs.discard(job_id)


@router.post("/run", response_model=JobResponse)
def trigger_run(
    background: BackgroundTasks,
    window: int | None = Query(default=None, ge=1, le=365),
    user: AuthUser = CurrentUser,
) -> JobResponse:
    settings = get_settings()
    topics = store_db.get_topics(user.id) or settings.default_topics
    categories = store_db.get_categories(user.id)
    if not topics and not categories:
        raise HTTPException(status_code=400, detail="add at least one topic or category before running")

    _, allowed = store_db.increment_rate_limit(
        user.id, "pipeline_run", daily_max=settings.daily_pipeline_limit
    )
    if not allowed:
        raise HTTPException(status_code=429, detail="daily pipeline limit reached")

    window_days = window or settings.discovery_window_days
    job_id = store_db.create_job(
        user.id,
        window_days=window_days,
        topics=topics,
        episode_count=settings.default_episode_count,
    )
    background.add_task(_run_job, job_id, user.id, topics, settings.default_episode_count, window_days)

    job = store_db.get_job(job_id)
    return JobResponse(**job)  # type: ignore[arg-type]


@router.get("/state", response_model=JobResponse | None)
def get_latest_state(user: AuthUser = CurrentUser):
    latest = store_db.latest_job(user.id)
    if not latest:
        return None
    return JobResponse(**latest)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, user: AuthUser = CurrentUser) -> JobResponse:
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid job id")
    job = store_db.get_job(jid)
    if not job or str(job.get("user_id")) != str(user.id):
        raise HTTPException(status_code=404, detail="job not found")
    return JobResponse(**job)
