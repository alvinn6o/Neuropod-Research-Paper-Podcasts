from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ----- Papers / episodes -----

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
    llm_provider: str = "demo"
    qa_status: str
    created_at: Optional[str] = None
    audio_url: str
    paper: PaperResponse
    script: Optional[str] = None


class EpisodeListResponse(BaseModel):
    items: list[EpisodeResponse]
    topics: list[str]


# ----- Topics / feedback / ask -----

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


# ----- Auth + me -----

class StubLoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=200, pattern=r".+@.+\..+")


class AuthSession(BaseModel):
    token: str
    user: "MeResponse"


class MeResponse(BaseModel):
    id: str
    email: str
    feed_slug: str
    display_name: Optional[str] = None
    keys: dict[str, str] = Field(default_factory=dict)  # provider -> hint


class KeyUpdateRequest(BaseModel):
    provider: Literal["openai", "anthropic", "elevenlabs", "bedrock"]
    # For api-key providers (openai/anthropic/elevenlabs):
    api_key: Optional[str] = Field(default=None, min_length=8, max_length=400)
    # For bedrock (AWS SigV4):
    region: Optional[str] = Field(default=None, min_length=2, max_length=40)
    access_key: Optional[str] = Field(default=None, min_length=8, max_length=128)
    secret_key: Optional[str] = Field(default=None, min_length=8, max_length=128)
    session_token: Optional[str] = Field(default=None, max_length=4096)
    model_id: Optional[str] = Field(default=None, max_length=128)


# ----- Pipeline jobs -----

class JobResponse(BaseModel):
    id: str
    status: str
    window_days: int
    topics: list[str]
    episode_count: int
    created_at: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    error: Optional[str]
    result_count: int


AuthSession.model_rebuild()
