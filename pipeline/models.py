from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class PaperCandidate:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published_at: str
    pdf_url: str
    sections: dict[str, str]
    citation_count: int = 0
    citation_velocity: float = 0.0
    recency_score: float = 0.0
    user_affinity_score: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaperChunk:
    id: str
    paper_id: str
    section: str
    chunk_index: int
    content: str
    token_count: int
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeDraft:
    id: str
    paper_id: str
    title: str
    description: str
    topic: str
    score: float
    script: str
    qa_status: str
    qa_notes: str
    duration_secs: int
    tts_provider: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)
