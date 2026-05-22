"""Pipeline runner — turns a job row into actual database rows.

Used by:
  * /pipeline/run web handler (synchronous, dev mode)
  * pipeline.worker CLI (Fargate task entrypoint, prod mode)
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

from pipeline.orchestrator import build_demo_payload
from pipeline.synthesize.tts import TTSProvider

from . import audio_store, keys_repo, store_db

logger = logging.getLogger("neuropod.runner")


def run_for_user(
    user_id: uuid.UUID,
    *,
    topics: list[str],
    episode_count: int,
    window_days: int,
    require_user_keys: bool = True,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Run the pipeline for one user. Persists papers, chunks, episodes, audio."""

    keys = keys_repo.load_keys(user_id)
    if require_user_keys and not (keys.get("openai") or keys.get("anthropic") or keys.get("elevenlabs")):
        raise RuntimeError(
            "no provider keys configured — add at least one in /settings/keys"
        )

    feedback = store_db.feedback_for_user(user_id)
    prior = store_db.list_episodes(user_id, limit=24)

    result = build_demo_payload(
        topics=topics,
        num_episodes=episode_count,
        window_days=window_days,
        feedback_events=feedback,
        prior_episodes=prior,
        keys=keys,
        require_user_keys=require_user_keys,
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

    # episodes — synth audio per episode and store
    inserted_ids: list[uuid.UUID] = []
    tts = TTSProvider(keys=keys, require_user_keys=require_user_keys)
    for episode in result["episodes"]:
        new_paper_id = paper_id_map.get(episode["paper_id"])
        if new_paper_id is None:
            continue

        episode_record = dict(episode)
        episode_record["llm_provider"] = episode.get("llm_provider", "demo")

        episode_id = store_db.insert_episode(user_id, new_paper_id, episode_record)
        inserted_ids.append(episode_id)

        # Synthesize + store audio (best-effort — failure isn't fatal)
        try:
            audio_bytes, mime, _ = tts.synthesize(episode["script"], title=episode["title"])
            cache_key = hashlib.sha1(
                f"{tts.provider_name}:{episode['script']}".encode()
            ).hexdigest()
            audio_store.store(key=cache_key, data=audio_bytes, content_type=mime)
            store_db.update_episode_audio(episode_id, audio_key=cache_key, audio_mime=mime)
        except Exception as exc:
            logger.warning("audio synth failed for episode %s: %s", episode_id, exc)

    return {
        "generated_at": result["generated_at"],
        "episode_ids": [str(i) for i in inserted_ids],
        "result_count": len(inserted_ids),
    }
