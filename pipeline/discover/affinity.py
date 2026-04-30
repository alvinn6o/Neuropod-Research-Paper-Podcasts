"""Compute per-topic affinity from feedback events. PLAN §5.1 weighted formula."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone


WEIGHTS = {"complete": 1.0, "play": 0.5, "skip": -0.4, "pause": 0.0}
HALF_LIFE_DAYS = 7


def _topic_terms(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(token) >= 3}


def compute_affinity(events: list[dict], episodes: list[dict]) -> dict[str, float]:
    """Returns {topic_keyword: score} weighted by event type and decayed by recency."""
    if not events or not episodes:
        return {}

    episode_topics: dict[str, str] = {ep["id"]: ep.get("topic", "") for ep in episodes}
    now = datetime.now(timezone.utc)
    scores: dict[str, float] = {}

    for event in events:
        weight = WEIGHTS.get(event.get("event_type", ""), 0.0)
        if not weight:
            continue
        episode_topic = episode_topics.get(event["episode_id"])
        if not episode_topic:
            continue
        try:
            ts = datetime.fromisoformat(event["created_at"].replace("Z", "+00:00"))
        except Exception:
            continue
        age_days = max((now - ts).total_seconds() / 86400, 0.0)
        decay = math.exp(-age_days / HALF_LIFE_DAYS)
        for term in _topic_terms(episode_topic):
            scores[term] = scores.get(term, 0.0) + weight * decay

    return scores


def boost(candidate_text: str, scores: dict[str, float]) -> float:
    """Affinity contribution for a candidate. Normalized to roughly [0, 1]."""
    if not scores:
        return 0.0
    candidate_terms = _topic_terms(candidate_text)
    raw = sum(scores.get(term, 0.0) for term in candidate_terms)
    if raw <= 0:
        return 0.0
    return min(raw / 4.0, 1.0)
