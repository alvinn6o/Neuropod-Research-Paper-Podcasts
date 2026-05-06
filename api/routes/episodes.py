from __future__ import annotations

import io
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, StreamingResponse

from .. import audio_store, store_db
from ..auth import AuthUser, CurrentUser
from ..models import EpisodeListResponse, EpisodeResponse, PaperResponse

logger = logging.getLogger("neuropod.episodes")

router = APIRouter(prefix="/episodes", tags=["episodes"])


def _audio_url(request: Request, episode_id: str) -> str:
    return str(request.url_for("stream_episode_audio", episode_id=episode_id))


def _serialize(request: Request, episode: dict) -> EpisodeResponse:
    paper = episode["paper"]
    return EpisodeResponse(
        id=episode["id"],
        title=episode["title"],
        description=episode["description"],
        topic=episode["topic"],
        score=round(episode["score"], 4),
        duration_secs=episode["duration_secs"],
        tts_provider=episode["tts_provider"],
        llm_provider=episode.get("llm_provider", "demo"),
        qa_status=episode["qa_status"],
        created_at=episode.get("created_at"),
        audio_url=_audio_url(request, episode["id"]),
        script=episode.get("script"),
        paper=PaperResponse.model_validate({
            **paper,
            "score": round(paper.get("score", 0.0), 4),
        }),
    )


@router.get("", response_model=EpisodeListResponse)
def list_episodes(
    request: Request,
    topic: str | None = Query(default=None),
    limit: int = Query(default=24, ge=1, le=50),
    user: AuthUser = CurrentUser,
) -> EpisodeListResponse:
    items = [_serialize(request, item) for item in store_db.list_episodes(user.id, topic=topic, limit=limit)]
    return EpisodeListResponse(items=items, topics=store_db.get_topics(user.id))


@router.get("/{episode_id}", response_model=EpisodeResponse)
def get_episode(episode_id: str, request: Request, user: AuthUser = CurrentUser) -> EpisodeResponse:
    import uuid as _u
    episode = store_db.get_episode(user.id, _u.UUID(episode_id))
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    return _serialize(request, episode)


@router.get("/{episode_id}/paper", response_model=PaperResponse)
def get_episode_paper(episode_id: str, user: AuthUser = CurrentUser) -> PaperResponse:
    import uuid as _u
    episode = store_db.get_episode(user.id, _u.UUID(episode_id))
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    return PaperResponse.model_validate(episode["paper"])


@router.get("/{episode_id}/audio", name="stream_episode_audio")
def stream_episode_audio(episode_id: str, user: AuthUser = CurrentUser):
    import uuid as _u
    episode = store_db.get_episode(user.id, _u.UUID(episode_id))
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    audio_key = episode.get("audio_key")
    if not audio_key:
        raise HTTPException(status_code=404, detail="Audio not yet generated")

    # If S3 is in use, prefer signed URL redirect (no proxying through Lambda).
    presigned = audio_store.url_for(audio_key)
    if presigned:
        return RedirectResponse(presigned, status_code=302)

    fetched = audio_store.fetch(audio_key)
    if not fetched:
        raise HTTPException(status_code=404, detail="Audio missing from storage")
    audio_bytes, mime = fetched
    return StreamingResponse(io.BytesIO(audio_bytes), media_type=mime, headers={
        "Content-Length": str(len(audio_bytes)),
        "Cache-Control": "public, max-age=86400",
        "Accept-Ranges": "bytes",
    })
