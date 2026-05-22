from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import APIRouter, HTTPException

from pipeline._http import ProviderError, post_json
from pipeline.generate.embedder import get_embedder
from pipeline.generate.retriever import Retriever
from pipeline.provider_status import record_failure, record_success

from .. import store_db
from ..auth import AuthUser, CurrentUser
from ..config import get_settings
from ..models import AskRequest, AskResponse, CitationResponse

logger = logging.getLogger("neuropod.ask")

router = APIRouter(tags=["ask"])


ASK_SYSTEM_PROMPT = (
    "You are a research assistant answering questions about a specific paper. "
    "Use the provided PAPER METADATA and RETRIEVED CHUNKS to answer. "
    "If the question asks about authors, title, arXiv ID, publication date, or "
    "categories, prefer the metadata. For substantive questions, ground your "
    "answer in the retrieved chunks. If the answer isn't supported by either, "
    "say so plainly. Keep answers under 120 words. No markdown, no preamble."
)


def _llm_answer(question: str, paper: dict, chunks: list[dict]) -> str | None:
    """Try to answer via Anthropic, then OpenAI. Returns None if neither works."""
    chunks_block = "\n\n".join(
        f"[{c['section']}] {c['content']}" for c in chunks[:6]
    )
    metadata_block = (
        f"TITLE: {paper.get('title', '')}\n"
        f"AUTHORS: {', '.join(paper.get('authors', []))}\n"
        f"ARXIV_ID: {paper.get('arxiv_id', '')}\n"
        f"CATEGORIES: {', '.join(paper.get('categories', []))}\n"
        f"PUBLISHED: {paper.get('published_at', '')}\n"
        f"ABSTRACT: {paper.get('abstract', '')[:1200]}"
    )
    user_prompt = (
        f"PAPER METADATA:\n{metadata_block}\n\n"
        f"RETRIEVED CHUNKS:\n{chunks_block}\n\n"
        f"QUESTION: {question.strip()}"
    )

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        try:
            start = time.time()
            result = post_json(
                provider="anthropic",
                url="https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                },
                body={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 400,
                    "system": ASK_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
                timeout=30,
            )
            record_success("ask:anthropic", latency_ms=int((time.time() - start) * 1000))
            return result["content"][0]["text"].strip()
        except ProviderError as exc:
            record_failure("ask:anthropic", error=exc.detail, status=exc.status)
            logger.warning("ask via anthropic failed: %s", exc)

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        try:
            start = time.time()
            result = post_json(
                provider="openai",
                url="https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}",
                },
                body={
                    "model": "gpt-4o-mini",
                    "max_tokens": 400,
                    "messages": [
                        {"role": "system", "content": ASK_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=30,
            )
            record_success("ask:openai", latency_ms=int((time.time() - start) * 1000))
            return result["choices"][0]["message"]["content"].strip()
        except ProviderError as exc:
            record_failure("ask:openai", error=exc.detail, status=exc.status)
            logger.warning("ask via openai failed: %s", exc)

    return None


def _metadata_answer(question: str, paper: dict) -> str | None:
    """Cheap deterministic answers for common metadata questions when no LLM is configured."""
    q = question.lower()
    if any(term in q for term in ["author", "who wrote", "wrote the paper", "wrote this"]):
        authors = paper.get("authors") or []
        if not authors:
            return f"The paper '{paper.get('title', '')}' lists no authors in the metadata."
        if len(authors) == 1:
            return f"The paper was written by {authors[0]}."
        return f"The paper was written by {', '.join(authors[:-1])}, and {authors[-1]}."
    if any(term in q for term in ["title", "what is the paper called", "name of the paper"]):
        return f"The paper is titled '{paper.get('title', '')}'."
    if "arxiv" in q and ("id" in q or "number" in q):
        return f"The arXiv ID is {paper.get('arxiv_id', '')}."
    if "publish" in q or "when was" in q:
        return f"It was published on {paper.get('published_at', '')[:10] or 'unknown'}."
    if "categor" in q or "field" in q:
        cats = paper.get("categories") or []
        if cats:
            return f"It's categorized under {', '.join(cats)}."
    return None


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

    paper = episode["paper"]
    chunks = store_db.get_chunks_for_paper(uuid.UUID(paper["id"]))
    if not chunks:
        raise HTTPException(status_code=404, detail="Paper chunks not found")

    _, allowed = store_db.increment_rate_limit(user.id, "ask", daily_max=settings.daily_ask_limit)
    if not allowed:
        raise HTTPException(status_code=429, detail="daily question limit reached")

    embedder = get_embedder()
    retriever = Retriever(embedder=embedder)
    top_chunks = retriever.retrieve(chunks, payload.question, limit=4)

    # 1. LLM-grounded answer if any provider key is set
    answer = _llm_answer(payload.question, paper, top_chunks)
    # 2. Deterministic metadata answer for common questions (no API cost)
    if not answer:
        answer = _metadata_answer(payload.question, paper)
    # 3. Last-resort extractive template
    if not answer:
        answer = retriever.answer_question(paper, top_chunks, payload.question)

    citations = [
        CitationResponse(section=c["section"], excerpt=c["content"][:240].strip())
        for c in top_chunks
    ]
    return AskResponse(answer=answer, citations=citations)
