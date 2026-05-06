from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from pipeline.generate.embedder import get_embedder
from pipeline.generate.retriever import Retriever

from .. import keys_repo, store_db
from ..auth import AuthUser, CurrentUser
from ..config import get_settings
from ..models import AskRequest, AskResponse, CitationResponse

router = APIRouter(tags=["ask"])


@router.post("/episodes/{episode_id}/ask", response_model=AskResponse)
def ask_episode(episode_id: str, payload: AskRequest, user: AuthUser = CurrentUser) -> AskResponse:
    settings = get_settings()
    try:
        eid = uuid.UUID(episode_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid episode id")

    episode = store_db.get_episode(user.id, eid)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    chunks = store_db.get_chunks_for_paper(uuid.UUID(episode["paper"]["id"]))
    if not chunks:
        raise HTTPException(status_code=404, detail="Paper chunks not found")

    # Use the user's own keys for embeddings (so /ask vector search uses real embeds when available)
    keys = keys_repo.load_keys(user.id)
    embedder = get_embedder(keys=keys, require_user_keys=settings.require_user_keys)

    # Rate limit
    _, allowed = store_db.increment_rate_limit(user.id, "ask", daily_max=settings.daily_ask_limit)
    if not allowed:
        raise HTTPException(status_code=429, detail="daily question limit reached")

    retriever = Retriever(embedder=embedder)
    top_chunks = retriever.retrieve(chunks, payload.question, limit=4)
    answer = retriever.answer_question(episode["paper"], top_chunks, payload.question)
    citations = [
        CitationResponse(section=c["section"], excerpt=c["content"][:240].strip())
        for c in top_chunks
    ]
    return AskResponse(answer=answer, citations=citations)
