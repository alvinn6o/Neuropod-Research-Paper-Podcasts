"""Pipeline worker - Fargate task entrypoint.

Usage:
  python -m pipeline.worker --job-id <uuid>
  python -m pipeline.worker --user-id <uuid> [--window 7] [--episodes 3] [--topics ...]

Reads NEUROPOD_DATABASE_URL from env, runs the script pipeline, and writes
episodes. Audio is optional via --with-audio or NEUROPOD_GENERATE_AUDIO_ON_PIPELINE.
Exits non-zero on failure so ECS reports failed task to CloudWatch.
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
    parser.add_argument("--with-audio", action="store_true", help="also synthesize TTS audio for generated scripts")
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
                synthesize_audio=args.with_audio or settings.generate_audio_on_pipeline,
            )
            store_db.update_job(
                job_id,
                status="done",
                finished_at=True,
                result_count=result["result_count"],
                skipped_count=result.get("skipped_existing", 0),
            )
            logger.info(
                "job %s completed: %d new, %d skipped (already existed)",
                job_id, result["result_count"], result.get("skipped_existing", 0),
            )
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
            synthesize_audio=args.with_audio or settings.generate_audio_on_pipeline,
        )
        store_db.update_job(
            job_id, status="done", finished_at=True,
            result_count=result["result_count"],
            skipped_count=result.get("skipped_existing", 0),
        )
        logger.info(
            "ad-hoc job completed: %d new, %d skipped (already existed)",
            result["result_count"], result.get("skipped_existing", 0),
        )
        return 0
    except Exception as exc:
        logger.exception("ad-hoc job failed")
        store_db.update_job(job_id, status="error", finished_at=True, error=str(exc)[:500])
        return 1


if __name__ == "__main__":
    sys.exit(main())
