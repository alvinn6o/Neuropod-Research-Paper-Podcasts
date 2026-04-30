from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PaperResponse(BaseModel):
    id: str
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_at: str
    pdf_url: str
    citation_count: int
    score: float = 0.0


class EpisodeResponse(BaseModel):
    id: str
    title: str
    description: str
    topic: str
    score: float
    duration_secs: int
    tts_provider: str
    qa_status: str
    created_at: str
    audio_url: str
    paper: PaperResponse
    script: str | None = None


class EpisodeListResponse(BaseModel):
    items: list[EpisodeResponse]
    topics: list[str]


class TopicUpdateRequest(BaseModel):
    topics: list[str] = Field(default_factory=list)


class TopicResponse(BaseModel):
    topics: list[str]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class CitationResponse(BaseModel):
    section: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


class FeedbackRequest(BaseModel):
    episode_id: str
    event_type: Literal["play", "pause", "skip", "complete"]
    position_secs: int = 0


class FeedbackResponse(BaseModel):
    ok: bool = True
    total_events: int
