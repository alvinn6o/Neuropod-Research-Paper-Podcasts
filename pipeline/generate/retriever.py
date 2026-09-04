from __future__ import annotations

import logging
from collections import Counter
from math import sqrt
from typing import Any

logger = logging.getLogger("neuropod.retriever")

# Bump when scoring changes. Persisted with every retrieval trace so a stored
# trace can be attributed to the code that produced it.
RETRIEVER_VERSION = "sparse-dense-sectionprior-v1"


class Retriever:
    section_bonus = {
        "abstract": 0.18,
        "results": 0.16,
        "conclusion": 0.12,
        "methods": 0.08,
        "introduction": 0.06,
        "discussion": 0.10,
        "limitations": 0.10,
    }

    def __init__(self, embedder: Any | None = None) -> None:
        self.embedder = embedder

    def retrieve(self, chunks: list[dict], query: str, limit: int = 5) -> list[dict]:
        return [row["chunk"] for row in self.retrieve_scored(chunks, query, limit=limit)]

    def retrieve_scored(self, chunks: list[dict], query: str, limit: int = 5) -> list[dict]:
        """Rank chunks and return the score components, not just the winners.

        The components are what make a retrieval trace useful: they are the
        features a learned reranker would train on, and they are what lets you
        answer "did the section prior or the cosine put this chunk on top?".
        `retrieve()` keeps the old chunks-only contract for existing callers.
        """
        if not chunks:
            return []

        query_dense = self._embed(query) if self._dense_available(chunks) else None
        query_vector = None if query_dense else self._sparse_vector(query)

        scored: list[dict] = []
        for chunk in chunks:
            bonus = self.section_bonus.get(chunk["section"], 0.0)
            if query_dense:
                dense = self._dense_cosine(query_dense, chunk["embedding"])
                sparse = None
                base = dense
            else:
                dense = None
                sparse = self._sparse_cosine(query_vector, self._sparse_vector(chunk["content"]))
                base = sparse
            scored.append({
                "chunk": chunk,
                "dense_score": dense,
                "sparse_score": sparse,
                "section_bonus": bonus,
                # Note the scale mismatch this exposes: `bonus` is a hand-set
                # additive constant summed straight onto a cosine. Replacing it
                # with a learned reranker is Phase 2.4; recording the components
                # separately is what makes that measurable.
                "final_score": base + bonus,
            })

        scored.sort(key=lambda row: row["final_score"], reverse=True)
        for rank, row in enumerate(scored):
            row["rank"] = rank
        return scored[:limit]

    def _dense_available(self, chunks: list[dict]) -> bool:
        """Dense scoring requires every chunk embedded by the SAME model.

        A single index holding vectors from two models yields cosines that look
        like numbers but mean nothing. When the set is mixed we drop to sparse
        for the whole query, which is at least internally consistent.
        """
        if not all(chunk.get("embedding") for chunk in chunks):
            return False
        models = {c.get("embedding_model") for c in chunks if c.get("embedding_model")}
        if len(models) > 1:
            logger.warning("mixed embedding models in one index %s; using sparse retrieval", models)
            return False
        return True

    def answer_question(self, paper: dict, chunks: list[dict], question: str) -> str:
        if not chunks:
            return f"The paper {paper['title']} does not have indexed chunks yet."

        lead = (
            f"Based on {paper['title']}, the clearest answer is that "
            f"{self._sentence(chunks[0]['content'])}"
        )
        supporting = " ".join(
            f"In the {chunk['section']} section, the paper adds that {self._sentence(chunk['content'])}"
            for chunk in chunks[1:]
        )
        return f"{lead} {supporting} This response is grounded in the paper sections most related to: {question.strip()}."

    def _embed(self, text: str) -> list[float] | None:
        if not self.embedder or not hasattr(self.embedder, "embed_text"):
            return None
        try:
            # OpenAIEmbedder returns None when the query embedding fails, which
            # routes this query to the sparse path rather than comparing a
            # fallback vector against a different model's chunk vectors.
            return self.embedder.embed_text(text)
        except Exception:
            return None

    def _sentence(self, text: str) -> str:
        sentence = text.strip().split(".")[0].strip()
        return sentence if sentence.endswith(".") else f"{sentence}."

    def _sparse_vector(self, text: str) -> Counter[str]:
        tokens = [
            token
            for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
            if len(token) > 2
        ]
        return Counter(tokens)

    def _sparse_cosine(self, left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = set(left).intersection(right)
        numerator = sum(left[token] * right[token] for token in overlap)
        left_norm = sqrt(sum(value * value for value in left.values()))
        right_norm = sqrt(sum(value * value for value in right.values()))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _dense_cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = sqrt(sum(a * a for a in left))
        right_norm = sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)
