from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from pipeline.generate.embedder import get_embedder
from pipeline.generate.retriever import Retriever

from ..dependencies import get_store
from ..models import AskRequest, AskResponse, CitationResponse
from ..storage import DemoStore

router = APIRouter(tags=["ask"])

_embedder = get_embedder()
_retriever = Retriever(embedder=_embedder)


@router.post("/episodes/{episode_id}/ask", response_model=AskResponse)
def ask_episode(
    episode_id: str,
    payload: AskRequest,
    store: DemoStore = Depends(get_store),
) -> AskResponse:
    episode = store.get_episode(episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    chunks = store.get_chunks_for_paper(episode["paper"]["id"])
    if not chunks:
        raise HTTPException(status_code=404, detail="Paper chunks not found")

    top_chunks = _retriever.retrieve(chunks, payload.question, limit=4)
    answer = _retriever.answer_question(episode["paper"], top_chunks, payload.question)
    citations = [
        CitationResponse(section=chunk["section"], excerpt=chunk["content"][:240].strip())
        for chunk in top_chunks
    ]
    return AskResponse(answer=answer, citations=citations)
