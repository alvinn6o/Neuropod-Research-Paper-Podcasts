from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time

from .._http import ProviderError, post_json
from ..ingest.tokenizer import truncate_to_tokens
from ..models import PaperChunk
from ..provider_status import record_failure, record_success
from ..usage import LLMCall, record

logger = logging.getLogger("neuropod.embed")

# Every embedder in this process must agree on this, and it must match
# `vector(N)` in db/schema.sql. Asserted at the persistence boundary.
EMBEDDING_DIM = 1536

# text-embedding-3-* accept 8191 tokens. Previously inputs were sliced at 8000
# *characters*, which is ~2000 tokens — throwing away most of a long chunk for
# no reason — while still not actually guaranteeing the token limit was met.
MAX_INPUT_TOKENS = 8000


class EmbeddingError(RuntimeError):
    """Raised when a chunk set could not be embedded in a single vector space.

    Deliberately not a ProviderError: the caller must decide whether to skip the
    paper, not silently substitute vectors from a different model.
    """


class HashEmbedder:
    """Deterministic hashed bag-of-words. Offline mode only.

    This is NOT a semantic embedding — "car" and "automobile" land in unrelated
    buckets. It exists so the app runs with no API key, and so tests are
    deterministic. It must never be mixed into an index alongside a real model:
    cosine similarity between a hash vector and an OpenAI vector is meaningless.
    """

    model_id = "hash-bow-v1"

    def __init__(self, dimensions: int = EMBEDDING_DIM) -> None:
        self.dimensions = dimensions

    def embed_chunks(self, chunks: list[PaperChunk]) -> list[PaperChunk]:
        for chunk in chunks:
            chunk.embedding = self.embed_text(chunk.content)
            chunk.embedding_model = self.model_id
            chunk.embedding_dim = self.dimensions
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
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch: int = 64,
        dimensions: int = EMBEDDING_DIM,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.batch = batch
        self.dimensions = dimensions

    @property
    def model_id(self) -> str:
        return self.model

    def embed_chunks(self, chunks: list[PaperChunk]) -> list[PaperChunk]:
        """Embed every chunk with this model, or raise.

        Previously a failed batch fell back to HashEmbedder for that batch only,
        so one paper could end up holding OpenAI vectors and SHA256 bag-of-words
        vectors in the same 1536-dim column, with nothing recording which was
        which. Retrieval over that index is silently wrong. Failing the paper is
        the correct outcome; the orchestrator skips it and says so.
        """
        if not chunks:
            return chunks
        for start in range(0, len(chunks), self.batch):
            window = chunks[start : start + self.batch]
            inputs = [truncate_to_tokens(c.content, MAX_INPUT_TOKENS) for c in window]
            try:
                vectors = self._call(inputs)
            except ProviderError as exc:
                record_failure("embed:openai", error=exc.detail, status=exc.status)
                raise EmbeddingError(
                    f"embedding batch failed ({exc}); refusing to mix embedding spaces"
                ) from exc

            if len(vectors) != len(window):
                # A short response would previously leave surplus chunks with
                # embedding=[], which then poisons the pgvector insert.
                raise EmbeddingError(
                    f"expected {len(window)} vectors, got {len(vectors)}"
                )
            for chunk, vector in zip(window, vectors):
                if len(vector) != self.dimensions:
                    raise EmbeddingError(
                        f"model returned dim {len(vector)}, index expects {self.dimensions}"
                    )
                chunk.embedding = vector
                chunk.embedding_model = self.model_id
                chunk.embedding_dim = self.dimensions
        return chunks

    def embed_text(self, text: str) -> list[float] | None:
        """Embed a query. Returns None on failure.

        None means "no dense query vector available" — the retriever then uses
        its sparse path, which is a coherent decision. Returning a hash vector
        here would compare it against OpenAI chunk vectors, which is not.
        """
        try:
            return self._call([truncate_to_tokens(text, MAX_INPUT_TOKENS)])[0]
        except (ProviderError, EmbeddingError) as exc:
            status = getattr(exc, "status", None)
            record_failure("embed:openai", error=str(exc)[:300], status=status)
            logger.warning("query embedding failed, falling back to sparse: %s", exc)
            return None

    def _call(self, inputs: list[str]) -> list[list[float]]:
        start = time.time()
        try:
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
        except ProviderError as exc:
            record(LLMCall(
                purpose="embed", provider="openai", model=self.model,
                latency_ms=int((time.time() - start) * 1000),
                ok=False, status=exc.status, error=exc.detail[:300],
            ))
            raise

        latency_ms = int((time.time() - start) * 1000)
        usage = result.get("usage") or {}
        record(LLMCall(
            purpose="embed", provider="openai", model=self.model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            latency_ms=latency_ms,
        ))
        record_success("embed:openai", latency_ms=latency_ms)
        try:
            return [row["embedding"] for row in result["data"]]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(f"unexpected embeddings response shape: {str(result)[:200]}") from exc


def get_embedder() -> HashEmbedder | OpenAIEmbedder:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return HashEmbedder()
    if os.getenv("NEUROPOD_EMBEDDER", "auto").lower() == "demo":
        return HashEmbedder()
    return OpenAIEmbedder(api_key=key)
