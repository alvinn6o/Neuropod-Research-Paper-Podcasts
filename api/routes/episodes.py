from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from pipeline.synthesize.audio_post import post_process
from pipeline.synthesize.tts import TTSProvider

from ..audio_cache import AudioCache
from ..config import get_settings
from ..dependencies import get_store
from ..models import EpisodeListResponse, EpisodeResponse, PaperResponse
from ..storage import DemoStore

logger = logging.getLogger("neuropod.episodes")

router = APIRouter(prefix="/episodes", tags=["episodes"])

_cache_dir = Path(get_settings().store_path).parent / "audio_cache"
_audio_cache = AudioCache(_cache_dir, max_bytes=500 * 1024 * 1024)


def _audio_url(request: Request, episode_id: str) -> str:
    return str(request.url_for("stream_episode_audio", episode_id=episode_id))


def _serialize_episode(request: Request, episode: dict) -> EpisodeResponse:
    return EpisodeResponse(
        id=episode["id"],
        title=episode["title"],
        description=episode["description"],
        topic=episode["topic"],
        score=round(episode["score"], 4),
        duration_secs=episode["duration_secs"],
        tts_provider=episode["tts_provider"],
        qa_status=episode["qa_status"],
        created_at=episode["created_at"],
        audio_url=_audio_url(request, episode["id"]),
        script=episode.get("script"),
        paper=PaperResponse.model_validate(
            {
                **episode["paper"],
                "score": round(episode["paper"].get("score", 0.0), 4),
            }
        ),
    )


@router.get("", response_model=EpisodeListResponse)
def list_episodes(
    request: Request,
    topic: str | None = Query(default=None),
    limit: int = Query(default=24, ge=1, le=50),
    store: DemoStore = Depends(get_store),
) -> EpisodeListResponse:
    items = [_serialize_episode(request, item) for item in store.list_episodes(topic=topic, limit=limit)]
    return EpisodeListResponse(items=items, topics=store.get_topics())


@router.get("/{episode_id}", response_model=EpisodeResponse)
def get_episode(
    episode_id: str,
    request: Request,
    store: DemoStore = Depends(get_store),
) -> EpisodeResponse:
    episode = store.get_episode(episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    return _serialize_episode(request, episode)


@router.get("/{episode_id}/paper", response_model=PaperResponse)
def get_episode_paper(
    episode_id: str,
    store: DemoStore = Depends(get_store),
) -> PaperResponse:
    episode = store.get_episode(episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    return PaperResponse.model_validate(episode["paper"])


@router.get("/{episode_id}/audio", name="stream_episode_audio")
def stream_episode_audio(
    episode_id: str,
    store: DemoStore = Depends(get_store),
) -> StreamingResponse:
    episode = store.get_episode(episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    provider = TTSProvider()
    script = episode["script"]
    cache_key = hashlib.sha1(f"{provider.provider_name}:v2:{script}".encode()).hexdigest()

    cached = _audio_cache.get(cache_key)
    if cached:
        audio_bytes, media_type = cached
    else:
        try:
            audio_bytes, media_type, _ = provider.synthesize(script, title=episode["title"])
        except Exception as exc:
            logger.error("tts synthesize failed for episode %s: %s", episode_id, exc)
            raise HTTPException(status_code=502, detail=f"TTS provider failed: {exc}")
        try:
            audio_bytes = post_process(audio_bytes, media_type)
        except Exception as exc:
            logger.warning("audio post-process skipped: %s", exc)
        _audio_cache.put(cache_key, audio_bytes, media_type)

    headers = {
        "Content-Length": str(len(audio_bytes)),
        "Cache-Control": "public, max-age=86400",
        "Accept-Ranges": "bytes",
    }
    return StreamingResponse(io.BytesIO(audio_bytes), media_type=media_type, headers=headers)
