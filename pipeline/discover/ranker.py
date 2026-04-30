from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone

from ..models import PaperCandidate


def rank_candidates(
    candidates: list[PaperCandidate],
    topics: list[str],
    top_k: int,
    window_days: int | None = None,
    affinity_scores: dict[str, float] | None = None,
) -> list[PaperCandidate]:
    from .affinity import boost as affinity_boost

    topic_terms = _topic_terms(topics)
    now = datetime.now(timezone.utc)
    if window_days is None:
        window_days = int(os.getenv("NEUROPOD_DISCOVERY_WINDOW_DAYS", "7"))
    half_life = max(window_days / 2, 1.0)

    for candidate in candidates:
        published = datetime.fromisoformat(candidate.published_at.replace("Z", "+00:00"))
        age_days = max((now - published).total_seconds() / 86400, 0.0)
        recency = math.exp(-(age_days / half_life))

        text = " ".join([candidate.title, candidate.abstract, *candidate.categories]).lower()
        affinity_hits = sum(1 for term in topic_terms if term in text)
        user_affinity = affinity_hits / max(len(topic_terms), 1)

        if affinity_scores:
            user_affinity = min(user_affinity + affinity_boost(text, affinity_scores), 1.0)

        trending = min(candidate.citation_velocity / 10, 1.0)
        candidate.recency_score = recency
        candidate.user_affinity_score = user_affinity
        candidate.score = (0.45 * recency) + (0.35 * trending) + (0.20 * user_affinity)

    return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]


def _topic_terms(topics: list[str]) -> set[str]:
    terms: set[str] = set()
    for topic in topics:
        for token in re.split(r"[^a-zA-Z0-9]+", topic.lower()):
            if len(token) >= 3:
                terms.add(token)
    return terms or {"research"}
