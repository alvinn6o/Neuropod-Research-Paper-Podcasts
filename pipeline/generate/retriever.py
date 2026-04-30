from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Any


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
        if not chunks:
            return []

        scored: list[tuple[float, dict]] = []
        query_dense = self._embed(query)

        if query_dense and all(chunk.get("embedding") for chunk in chunks):
            for chunk in chunks:
                similarity = self._dense_cosine(query_dense, chunk["embedding"])
                similarity += self.section_bonus.get(chunk["section"], 0.0)
                scored.append((similarity, chunk))
        else:
            query_vector = self._sparse_vector(query)
            for chunk in chunks:
                chunk_vector = self._sparse_vector(chunk["content"])
                similarity = self._sparse_cosine(query_vector, chunk_vector)
                similarity += self.section_bonus.get(chunk["section"], 0.0)
                scored.append((similarity, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:limit]]

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
