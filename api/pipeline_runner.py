"""Pipeline runner - turns a job row into actual database rows.

Used by:
  * /pipeline/run web handler (synchronous, dev mode)
  * pipeline.worker CLI (Fargate task entrypoint, prod mode)

Provider keys come from operator-level env vars (ANTHROPIC_API_KEY,
OPENAI_API_KEY, ELEVENLABS_API_KEY) or, for Bedrock, from the task's
IAM role when NEUROPOD_BEDROCK_OPERATOR=true. Script generation and TTS are
separate: the pipeline persists scripts first, and audio can be synthesized
on demand or enabled explicitly for a run.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

from pipeline.orchestrator import build_demo_payload
from pipeline.synthesize.tts import TTSProvider

from . import audio_store, store_db

logger = logging.getLogger("neuropod.runner")


def run_for_user(
    user_id: uuid.UUID,
    *,
    topics: list[str],
    episode_count: int,
    window_days: int,
    categories: list[str] | None = None,
    synthesize_audio: bool = False,
) -> dict[str, Any]:
    """Run the pipeline for one user. Persists papers, chunks, and scripts."""

    feedback = store_db.feedback_for_user(user_id)
    prior = store_db.list_episodes(user_id, limit=24)

    result = build_demo_payload(
        topics=topics,
        num_episodes=episode_count,
        window_days=window_days,
        feedback_events=feedback,
        prior_episodes=prior,
        categories=categories or store_db.get_categories(user_id),
    )

    # papers + chunks
    paper_id_map: dict[str, uuid.UUID] = {}
    for paper in result["papers"]:
        new_id = store_db.upsert_paper(paper)
        paper_id_map[paper["id"]] = new_id

    grouped: dict[str, list[dict]] = {}
    for chunk in result["chunks"]:
        grouped.setdefault(chunk["paper_id"], []).append(chunk)
    for old_paper_id, chunks in grouped.items():
        new_paper_id = paper_id_map.get(old_paper_id)
        if new_paper_id is None:
            continue
        store_db.replace_chunks_for_paper(new_paper_id, chunks)

    # Episodes are script-first. Audio is optional and can be generated later.
    # Dedup: skip papers the user already has an episode for.
    inserted_ids: list[uuid.UUID] = []
    skipped_existing = 0
    for episode in result["episodes"]:
        new_paper_id = paper_id_map.get(episode["paper_id"])
        if new_paper_id is None:
            continue

        existing = store_db.find_episode_for_paper(user_id, new_paper_id)
        if existing is not None:
            skipped_existing += 1
            logger.info("skipping duplicate episode for paper %s (existing=%s)", new_paper_id, existing)
            continue

        episode_record = dict(episode)
        episode_record["llm_provider"] = episode.get("llm_provider", "demo")

        episode_id = store_db.insert_episode(user_id, new_paper_id, episode_record)
        inserted_ids.append(episode_id)

        if synthesize_audio:
            try:
                synthesize_episode_audio(episode_id, script=episode["script"], title=episode["title"])
            except Exception as exc:
                logger.warning("audio synth failed for episode %s: %s", episode_id, exc)

    return {
        "generated_at": result["generated_at"],
        "episode_ids": [str(i) for i in inserted_ids],
        "result_count": len(inserted_ids),
        "skipped_existing": skipped_existing,
    }


def synthesize_episode_audio(
    episode_id: uuid.UUID,
    *,
    script: str,
    title: str,
) -> dict[str, str]:
    """Synthesize and persist audio for an already-generated script."""
    tts = TTSProvider()
    audio_bytes, mime, provider = tts.synthesize(script, title=title)
    cache_key = hashlib.sha1(f"{provider}:{script}".encode()).hexdigest()
    audio_store.store(key=cache_key, data=audio_bytes, content_type=mime)
    store_db.update_episode_audio(
        episode_id,
        audio_key=cache_key,
        audio_mime=mime,
        tts_provider=provider,
    )
    return {"audio_key": cache_key, "audio_mime": mime, "tts_provider": provider}
