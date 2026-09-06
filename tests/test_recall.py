"""Recall@k benchmark for the retriever, run on a real paper fixture.

The fixture is Mamba (arXiv:2312.00752) — chunked, embedded, and the
chunks + embeddings are checked into tests/fixtures so this test runs
in CI without external calls.

Methodology:
  * 12 hand-authored queries with gold_substrings — see mamba_queries.json
  * A query is a hit at rank-k if ANY of the top-k retrieved chunks
    contains ANY of the gold_substrings (case-insensitive)
  * Reports recall@1, recall@5, recall@10 + MRR

These numbers tell you HOW WELL retrieval works; the deterministic
unit tests in test_evals.py tell you whether the scaffolding is sound.

To regenerate the fixtures with real OpenAI embeddings (better numbers):
  OPENAI_API_KEY=... NEUROPOD_EMBEDDER=openai python -m eval.precompute_fixtures
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pipeline.generate.retriever import Retriever

FIXTURES = Path(__file__).parent / "fixtures"
CHUNKS_PATH = FIXTURES / "mamba_chunks.json"
QUERIES_PATH = FIXTURES / "mamba_queries.json"
META_PATH = FIXTURES / "mamba_meta.json"

# Which embedder produced the checked-in vectors. Printed with every metric so a
# number can never be quoted without the model that produced it — the README
# previously reported these as the system's retrieval quality when they measure
# the SHA256 bag-of-words fallback, not the shipping OpenAI path.
EMBEDDER_BACKEND = (
    json.loads(META_PATH.read_text()).get("embedder_backend", "unknown")
    if META_PATH.exists()
    else "unknown"
)


def _load_chunks() -> list[dict]:
    return json.loads(CHUNKS_PATH.read_text())


def _load_queries() -> list[dict]:
    return json.loads(QUERIES_PATH.read_text())["queries"]


def _is_hit(chunks: list[dict], gold_substrings: list[str]) -> bool:
    haystack = " ".join(c["content"].lower() for c in chunks)
    return any(s.lower() in haystack for s in gold_substrings)


def _first_hit_rank(chunks: list[dict], gold_substrings: list[str]) -> int | None:
    """1-indexed position of the first hit, or None."""
    for i, chunk in enumerate(chunks, start=1):
        text = chunk["content"].lower()
        if any(s.lower() in text for s in gold_substrings):
            return i
    return None


@pytest.fixture(scope="module")
def chunks():
    return _load_chunks()


@pytest.fixture(scope="module")
def queries():
    return _load_queries()


@pytest.fixture(scope="module")
def retriever(chunks):
    # Pass an embedder of the same family used to build the fixture so dense
    # mode kicks in when embeddings are present.
    if chunks and chunks[0].get("embedding"):
        meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
        backend = meta.get("embedder_backend", "HashEmbedder")
        if backend == "OpenAIEmbedder" and os.getenv("OPENAI_API_KEY"):
            from pipeline.generate.embedder import OpenAIEmbedder
            embedder = OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY", ""))
        else:
            from pipeline.generate.embedder import HashEmbedder
            embedder = HashEmbedder(dimensions=len(chunks[0]["embedding"]))
        return Retriever(embedder=embedder)
    return Retriever()


def test_fixtures_present():
    assert CHUNKS_PATH.exists(), "run `python -m eval.precompute_fixtures` first"
    assert QUERIES_PATH.exists()
    chunks = _load_chunks()
    queries = _load_queries()
    assert len(chunks) >= 30, f"only {len(chunks)} chunks — pdf parse may be degraded"
    assert len(queries) >= 10, "need at least 10 queries for meaningful recall stats"


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    Reported alongside every recall number because the point estimate on its own
    is misleading at this sample size: recall@1 = 7/12 carries an interval of
    roughly [0.32, 0.81]. An interval that wide cannot distinguish a real 10-point
    retrieval improvement from noise, which means any A/B comparison run against
    this fixture is unfalsifiable. Fixing that needs more papers and more
    queries, not a better retriever — see Phase 1 of the roadmap.
    """
    if n == 0:
        return (0.0, 0.0)
    p_hat = hits / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@pytest.mark.parametrize("k", [1, 5, 10])
def test_recall_at_k(retriever, chunks, queries, k):
    hits = 0
    for q in queries:
        top = retriever.retrieve(chunks, q["query"], limit=k)
        if _is_hit(top, q["gold_substrings"]):
            hits += 1
    recall = hits / len(queries)

    # Print so CI logs show real numbers — with the interval, not just the point
    lo, hi = wilson_interval(hits, len(queries))
    print(
        f"\n  recall@{k}: {hits}/{len(queries)} = {recall:.1%} "
        f"[95% CI {lo:.1%}-{hi:.1%}]  (embedder={EMBEDDER_BACKEND}, n={len(queries)})"
    )

    # Honest, embedder-agnostic floor. Hash embeddings yield ~50-70%;
    # OpenAI embeddings should clear ~85%+ at k=10.
    floor = {1: 0.10, 5: 0.40, 10: 0.55}[k]
    assert recall >= floor, (
        f"recall@{k}={recall:.2%} below floor {floor:.0%}. "
        f"Either retrieval regressed or the fixture is stale."
    )


def test_mrr(retriever, chunks, queries):
    reciprocal_ranks = []
    for q in queries:
        # Retrieve more than we'd ever need so MRR isn't capped artificially
        top = retriever.retrieve(chunks, q["query"], limit=20)
        rank = _first_hit_rank(top, q["gold_substrings"])
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"\n  MRR over {len(queries)} queries = {mrr:.3f}  (embedder={EMBEDDER_BACKEND})")
    assert mrr > 0.15, f"MRR={mrr:.3f} — first relevant chunk is ranked too low on average"


def test_section_diversity_in_top_5(retriever, chunks, queries):
    """Top-5 should pull from at least 2 different sections on average."""
    distinct_section_counts = []
    for q in queries:
        top = retriever.retrieve(chunks, q["query"], limit=5)
        distinct_section_counts.append(len({c["section"] for c in top}))
    avg = sum(distinct_section_counts) / len(distinct_section_counts)
    print(f"\n  avg distinct sections in top-5 = {avg:.2f}")
    assert avg >= 1.5, "top-5 is too concentrated in one section — retrieval not spreading enough"
