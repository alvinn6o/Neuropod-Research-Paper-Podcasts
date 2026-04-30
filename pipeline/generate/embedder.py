from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time

from .._http import ProviderError, post_json
from ..models import PaperChunk
from ..provider_status import record_failure, record_success

logger = logging.getLogger("neuropod.embed")


class HashEmbedder:
    """Deterministic hash-based fallback (48 dims). Used when no real key is set."""

    def __init__(self, dimensions: int = 48) -> None:
        self.dimensions = dimensions

    def embed_chunks(self, chunks: list[PaperChunk]) -> list[PaperChunk]:
        for chunk in chunks:
            chunk.embedding = self.embed_text(chunk.content)
        return chunks

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAIEmbedder:
    """Real OpenAI embeddings (text-embedding-3-small, 1536 dims)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", batch: int = 64) -> None:
        self.api_key = api_key
        self.model = model
        self.batch = batch

    def embed_chunks(self, chunks: list[PaperChunk]) -> list[PaperChunk]:
        if not chunks:
            return chunks
        for start in range(0, len(chunks), self.batch):
            window = chunks[start : start + self.batch]
            inputs = [c.content[:8000] for c in window]
            try:
                vectors = self._call(inputs)
            except ProviderError as exc:
                record_failure("embed:openai", error=exc.detail, status=exc.status)
                logger.warning("openai embeddings failed; falling back: %s", exc)
                fallback = HashEmbedder()
                for chunk in window:
                    chunk.embedding = fallback.embed_text(chunk.content)
                continue
            for chunk, vector in zip(window, vectors):
                chunk.embedding = vector
        return chunks

    def embed_text(self, text: str) -> list[float]:
        try:
            return self._call([text[:8000]])[0]
        except ProviderError as exc:
            record_failure("embed:openai", error=exc.detail, status=exc.status)
            return HashEmbedder().embed_text(text)

    def _call(self, inputs: list[str]) -> list[list[float]]:
        start = time.time()
        result = post_json(
            provider="openai-embed",
            url="https://api.openai.com/v1/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            body={"model": self.model, "input": inputs},
            timeout=60,
        )
        record_success("embed:openai", latency_ms=int((time.time() - start) * 1000))
        return [row["embedding"] for row in result["data"]]


def get_embedder() -> HashEmbedder | OpenAIEmbedder:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return HashEmbedder()
    if os.getenv("NEUROPOD_EMBEDDER", "auto").lower() == "demo":
        return HashEmbedder()
    return OpenAIEmbedder(api_key=key)
