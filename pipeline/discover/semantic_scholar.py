from __future__ import annotations

from ..models import PaperCandidate


class SemanticScholarClient:
    """Demo metadata enricher that keeps the interface future-proof."""

    def enrich(self, candidates: list[PaperCandidate]) -> list[PaperCandidate]:
        for candidate in candidates:
            if candidate.citation_velocity <= 0:
                candidate.citation_velocity = max(candidate.citation_count / 5, 0.1)
        return candidates
