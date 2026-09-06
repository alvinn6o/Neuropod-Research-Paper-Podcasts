from __future__ import annotations

import logging
import os
from collections import Counter
from math import sqrt
from typing import Any

from .bm25 import BM25Index, fuse_rank_maps, tie_aware_ranks

logger = logging.getLogger("neuropod.retriever")

# Bump when scoring changes. Persisted with every retrieval trace so a stored
# trace can be attributed to the code that produced it.
RETRIEVER_VERSION = "hybrid-bm25-dense-rrf-v2"

# How many chunks reach the prompt. Kept here because it is a retrieval
# parameter, not a scriptwriter one: it defines what "good retrieval" means for
# this product (hit@14 = did the relevant passage make it in).
PROMPT_CHUNK_LIMIT = 14


class Retriever:
    """Hybrid BM25 + dense retrieval, fused with Reciprocal Rank Fusion.

    Replaces raw term-frequency cosine plus a hand-set additive section prior.
    Both changes are measured on the 168-paper / 1,935-query benchmark:

      * proper BM25 (IDF, length normalization, stopwords) is +0.078 nDCG@10
        over the previous scoring, CI [+0.064, +0.092], p<0.001
      * the section prior is *harmful*: dense -> dense+prior costs 0.034
        nDCG@10, CI [-0.042, -0.025], p<0.001, and rrf -> rrf+prior costs
        0.011, p<0.001

    Lexical and dense retrieval fail differently, which is why both run. An
    embedding model cannot represent a term coined after it was trained;
    BM25 matches the string. Identifiers, figures and novel method names —
    "Mamba", "A100", "5.2x" — are exactly what this corpus is full of.

    RRF fuses them without score normalization: BM25 scores are unbounded and
    corpus-dependent while cosines are in [-1, 1], so any weighted sum needs a
    tuning step that RRF avoids entirely.
    """

    # Retained ONLY as the historical baseline that eval/ compares against, and
    # so the learned reranker's section one-hots can be read against the weights
    # they replace. NOT applied in the scoring path — see the class docstring.
    section_bonus = {
        "abstract": 0.18,
        "results": 0.16,
        "conclusion": 0.12,
        "methods": 0.08,
        "introduction": 0.06,
        "discussion": 0.10,
        "limitations": 0.10,
    }

    def __init__(self, embedder: Any | None = None, reranker: Any | None = None) -> None:
        self.embedder = embedder
        # Lazily resolved so importing the retriever never loads a model.
        self._reranker = reranker
        self._reranker_resolved = reranker is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, chunks: list[dict], query: str, limit: int = 5) -> list[dict]:
        return [row["chunk"] for row in self.retrieve_scored(chunks, query, limit=limit)]

    def retrieve_scored(self, chunks: list[dict], query: str, limit: int = 5) -> list[dict]:
        """Rank chunks for one query, returning score components.

        The components are what make a retrieval trace useful: they are the
        features a learned reranker trains on, and they answer "what put this
        chunk on top?".
        """
        return self.retrieve_multi(chunks, [query], limit=limit)

    def retrieve_multi(
        self, chunks: list[dict], queries: list[str], limit: int = 5
    ) -> list[dict]:
        """Rank chunks against several queries at once, fusing the results.

        Used for facet retrieval: the scriptwriter's prompt asks for the
        problem, the method, the quantitative results and the limitations, so
        each is searched for separately and the ranked lists are fused. One
        query built from the abstract asks "find chunks like the summary of the
        whole paper", which is diffuse by construction and biased toward the
        abstract chunk itself.
        """
        if not chunks:
            return []
        queries = [q for q in queries if q and q.strip()] or [""]

        index = BM25Index.build([(c["id"], c["content"]) for c in chunks])
        dense_ok = self._dense_available(chunks)

        rankings: list[dict[str, float]] = []
        bm25_best: dict[str, float] = {}
        dense_best: dict[str, float] = {}

        for query in queries:
            bm = index.score(query)
            for cid, score in bm.items():
                bm25_best[cid] = max(bm25_best.get(cid, 0.0), score)
            # Only chunks that actually matched enter the fusion. RRF scores by
            # rank, so a chunk with score 0.0 would otherwise earn credit purely
            # for being in the list — on a query with no lexical hits every
            # chunk would be "ranked" and the fused order would be arbitrary.
            rankings.append(tie_aware_ranks({c: v for c, v in bm.items() if v > 0.0}))

            if dense_ok:
                qv = self._embed(query)
                if qv:
                    dn = {
                        c["id"]: self._dense_cosine(qv, c["embedding"]) for c in chunks
                    }
                    for cid, score in dn.items():
                        dense_best[cid] = max(dense_best.get(cid, -1.0), score)
                    rankings.append(tie_aware_ranks({c: v for c, v in dn.items() if v > 0.0}))

        fused = fuse_rank_maps(rankings)
        by_id = {c["id"]: c for c in chunks}

        # Unmatched chunks keep a deterministic order (corpus order) below every
        # matched one, rather than whatever the dict happened to yield.
        ordered = sorted(
            chunks,
            key=lambda c: (-fused.get(c["id"], 0.0), c.get("chunk_index", 0), c["id"]),
        )

        scored = [
            {
                "chunk": chunk,
                "dense_score": dense_best.get(chunk["id"]) if dense_best else None,
                "sparse_score": bm25_best.get(chunk["id"], 0.0),
                # Recorded as 0.0 rather than dropped: the trace schema and the
                # eval baselines both still carry the column, and an explicit
                # zero says "considered and not applied".
                "section_bonus": 0.0,
                "final_score": fused.get(chunk["id"], 0.0),
            }
            for chunk in ordered
        ]

        scored = self._maybe_rerank(scored, queries[0])
        for rank, row in enumerate(scored):
            row["rank"] = rank
        return scored[:limit]

    # ------------------------------------------------------------------
    # Reranking (opt-in)
    # ------------------------------------------------------------------

    def _maybe_rerank(self, scored: list[dict], query: str) -> list[dict]:
        reranker = self._load_reranker()
        if reranker is None or len(scored) < 2:
            return scored
        try:
            return reranker.rerank(scored, query)
        except Exception as exc:
            # A reranker failure must not fail retrieval — the fused order is
            # already a usable result.
            logger.warning("reranker failed, falling back to fused order: %s", exc)
            return scored

    def _load_reranker(self):
        if self._reranker_resolved:
            return self._reranker
        self._reranker_resolved = True
        if os.getenv("NEUROPOD_RERANKER", "").strip().lower() not in {"1", "true", "on"}:
            self._reranker = None
            return None
        from .rerank import load_reranker

        self._reranker = load_reranker()
        return self._reranker

    # ------------------------------------------------------------------
    # Q&A helper
    # ------------------------------------------------------------------

    def answer_question(self, paper: dict, chunks: list[dict], question: str) -> str:
        """Last-resort extractive answer when no LLM provider is reachable.

        Deliberately reads as a degraded template rather than a real answer:
        it stitches together leading sentences from the top chunks. It exists so
        the endpoint returns something grounded instead of erroring.
        """
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dense_available(self, chunks: list[dict]) -> bool:
        """Dense scoring requires every chunk embedded by the SAME model.

        A single index holding vectors from two models yields cosines that look
        like numbers but mean nothing.
        """
        if not self.embedder or not all(chunk.get("embedding") for chunk in chunks):
            return False
        models = {c.get("embedding_model") for c in chunks if c.get("embedding_model")}
        if len(models) > 1:
            logger.warning("mixed embedding models in one index %s; lexical only", models)
            return False
        return True

    def _embed(self, text: str) -> list[float] | None:
        if not self.embedder or not hasattr(self.embedder, "embed_text"):
            return None
        try:
            # OpenAIEmbedder returns None when the query embedding fails, which
            # keeps this query lexical rather than comparing a fallback vector
            # against a different model's chunk vectors.
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
