"""Pipeline worker — Fargate task entrypoint.

Usage:
  python -m pipeline.worker --job-id <uuid>
  python -m pipeline.worker --user-id <uuid> [--window 7] [--episodes 3] [--topics ...]

Reads NEUROPOD_DATABASE_URL + NEUROPOD_MASTER_KEY from env.
Loads the user's BYOK keys from Postgres, runs the pipeline, writes episodes
+ audio. Exits non-zero on failure so ECS reports failed task to CloudWatch.
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid

from api import store_db
from api.config import get_settings
from api.db import init_schema
from api.pipeline_runner import run_for_user

logger = logging.getLogger("neuropod.worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _resolve_job(job_id: uuid.UUID) -> tuple[uuid.UUID, list[str], int, int]:
    job = store_db.get_job(job_id)
    if not job:
        raise SystemExit(f"job {job_id} not found")
    return (
        uuid.UUID(str(job["user_id"])),
        list(job.get("topics") or []),
        int(job.get("episode_count") or 3),
        int(job.get("window_days") or 7),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Neuropod pipeline worker")
    parser.add_argument("--job-id", help="Pipeline job UUID to claim and run")
    parser.add_argument("--user-id", help="User UUID (used when --job-id is omitted)")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--topics", nargs="*", default=None)
    args = parser.parse_args(argv)

    init_schema()
    settings = get_settings()

    if args.job_id:
        job_id = uuid.UUID(args.job_id)
        user_id, topics, episode_count, window_days = _resolve_job(job_id)
        store_db.update_job(job_id, status="running", started_at=True)
        try:
            result = run_for_user(
                user_id,
                topics=topics,
                episode_count=episode_count,
                window_days=window_days,
                require_user_keys=settings.require_user_keys,
            )
            store_db.update_job(
                job_id,
                status="done",
                finished_at=True,
                result_count=result["result_count"],
            )
            logger.info("job %s completed: %d episodes", job_id, result["result_count"])
            return 0
        except Exception as exc:
            logger.exception("job failed")
            store_db.update_job(job_id, status="error", finished_at=True, error=str(exc)[:500])
            return 1

    if not args.user_id:
        parser.error("either --job-id or --user-id is required")

    user_id = uuid.UUID(args.user_id)
    topics = args.topics or store_db.get_topics(user_id) or settings.default_topics
    episode_count = args.episodes or settings.default_episode_count
    window_days = args.window or settings.discovery_window_days

    job_id = store_db.create_job(
        user_id, window_days=window_days, topics=topics, episode_count=episode_count
    )
    store_db.update_job(job_id, status="running", started_at=True)
    try:
        result = run_for_user(
            user_id,
            topics=topics,
            episode_count=episode_count,
            window_days=window_days,
            require_user_keys=settings.require_user_keys,
        )
        store_db.update_job(
            job_id, status="done", finished_at=True, result_count=result["result_count"]
        )
        logger.info("ad-hoc job completed: %s episodes", result["result_count"])
        return 0
    except Exception as exc:
        logger.exception("ad-hoc job failed")
        store_db.update_job(job_id, status="error", finished_at=True, error=str(exc)[:500])
        return 1


if __name__ == "__main__":
    sys.exit(main())
