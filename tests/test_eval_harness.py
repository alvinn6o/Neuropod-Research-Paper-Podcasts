"""Evaluation-harness correctness, and the retrieval regression gate.

Two different jobs in one file:

  * unit tests for the metric and BM25 implementations — a benchmark whose
    metrics are wrong is worse than no benchmark, because it produces confident
    numbers;
  * a gate that fails when retrieval regresses against the frozen corpus.

The gate is the point of Phase 1. Before it, `eval/ragas_eval.py` printed three
floats to stdout on 3 live-fetched papers and nothing could fail.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.metrics import (
    bootstrap_ci,
    hit_at_k,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
    reciprocal_rank,
    wilson_interval,
)
from pipeline.generate.bm25 import BM25Index, reciprocal_rank_fusion, tokenize

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "eval" / "baselines.json"
CORPUS_CHUNKS = ROOT / "eval" / "corpus" / "chunks.jsonl"
CORPUS_QUERIES = ROOT / "eval" / "corpus" / "queries_ict.jsonl"

needs_corpus = pytest.mark.skipif(
    not (CORPUS_CHUNKS.exists() and CORPUS_QUERIES.exists()),
    reason="frozen corpus not built (python -m eval.corpus_build build)",
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_ndcg_is_1_for_ideal_ranking_and_lower_otherwise():
    qrels = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], qrels, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a", "c"], qrels, 3) < 1.0
    assert ndcg_at_k(["c", "b", "a"], qrels, 3) < ndcg_at_k(["b", "a", "c"], qrels, 3)


def test_ndcg_distinguishes_grades_where_binary_recall_cannot():
    """The reason for graded labels: both rankings have the same hit@1."""
    qrels = {"best": 2, "ok": 1}
    good = ndcg_at_k(["best", "ok"], qrels, 2)
    worse = ndcg_at_k(["ok", "best"], qrels, 2)
    assert good > worse
    assert hit_at_k(["best", "ok"], qrels, 1) == hit_at_k(["ok", "best"], qrels, 1) == 1.0


def test_ndcg_is_zero_when_nothing_relevant_exists():
    assert ndcg_at_k(["a"], {"a": 0}, 1) == 0.0


def test_recall_and_hit_differ():
    """`test_recall.py` calls hit@k 'recall' — they are not the same metric."""
    qrels = {"a": 1, "b": 1}
    assert recall_at_k(["a", "x"], qrels, 2) == 0.5
    assert hit_at_k(["a", "x"], qrels, 2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a"], {"a": 1}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a": 1}) == 0.0


def test_wilson_narrows_as_n_grows():
    lo_small, hi_small = wilson_interval(7, 12)
    lo_big, hi_big = wilson_interval(700, 1200)
    assert (hi_small - lo_small) > 4 * (hi_big - lo_big)


def test_bootstrap_ci_is_deterministic_under_a_seed():
    vals = [0.1 * i for i in range(20)]
    assert bootstrap_ci(vals, seed=3) == bootstrap_ci(vals, seed=3)


def test_paired_bootstrap_detects_a_real_shift_and_ignores_noise():
    better = paired_bootstrap([0.4] * 60, [0.5] * 60)
    assert better.delta == pytest.approx(0.1)
    assert better.significant

    same = paired_bootstrap([0.4] * 60, [0.4] * 60)
    assert not same.significant


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1], [0.1, 0.2])


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def test_tokenizer_preserves_identifiers_and_numbers():
    assert tokenize("GPT-4 hits 3.2x on h_t") == ["gpt-4", "hits", "3.2x", "h_t"]


def test_stopwords_carry_no_signal():
    idx = BM25Index.build([("a", "the model of the data"), ("b", "unrelated content here")])
    assert all(score == 0.0 for score in idx.score("the of the").values())


def test_idf_prefers_rare_terms():
    docs = [(str(i), "common term here") for i in range(9)]
    docs.append(("rare", "common term rare_token"))
    idx = BM25Index.build(docs)
    assert idx.idf("rare_token") > idx.idf("common")


def test_bm25_normalizes_for_length():
    """A short exact match should beat a long chunk padded with filler."""
    idx = BM25Index.build([
        ("short", "selective scan throughput"),
        ("long", "selective scan throughput " + "filler content padding words " * 40),
    ])
    ranked = [cid for cid, _ in idx.top_k("selective scan throughput", 2)]
    assert ranked[0] == "short"


def test_rrf_rewards_agreement_across_rankings():
    """A doc ranked highly by both systems must beat one ranked highly by only
    one, even when that one system ranks it first."""
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]]))
    assert max(fused, key=fused.get) == "a", "agreed-on doc should win"
    assert fused["b"] == fused["c"], "ranks 2+3 and 3+2 must tie"

    # A doc first in one list but absent from the other loses to a doc that is
    # second in both — the property that makes RRF robust to one bad system.
    fused2 = dict(reciprocal_rank_fusion([["x", "steady"], ["steady"]]))
    assert fused2["steady"] > fused2["x"]


# ---------------------------------------------------------------------------
# The regression gate
# ---------------------------------------------------------------------------

@needs_corpus
@pytest.mark.parametrize("config", ["current", "bm25", "dense"])
def test_retrieval_has_not_regressed(config):
    """Fail if nDCG@10 drops more than `tolerance` below the frozen baseline.

    Not a floor plucked from intuition — the baseline is a measured value on a
    pinned corpus, and the tolerance is wide enough to absorb the bootstrap's
    own variance while still catching a real regression.
    """
    from eval.harness import run
    from eval.metrics import evaluate_run

    baselines = json.loads(BASELINES.read_text())
    expected = baselines["configs"][config]["ndcg@10"]
    tolerance = baselines["tolerance"]

    result = run([config])
    actual = evaluate_run(result["runs"][config], result["qrels"])["ndcg@10"]

    print(f"\n  {config}: nDCG@10 = {actual.value:.4f} "
          f"[{actual.ci_low:.4f},{actual.ci_high:.4f}] "
          f"baseline {expected:.4f} (tolerance {tolerance})")

    assert actual.value >= expected - tolerance, (
        f"{config} nDCG@10 regressed: {actual.value:.4f} vs baseline {expected:.4f}. "
        f"If this change is intended, update eval/baselines.json in this PR and "
        f"include the paired-bootstrap delta in the commit message."
    )


@needs_corpus
def test_corpus_is_large_enough_to_be_falsifiable():
    """The failure the old fixture had: n=12 gives a CI ~50 points wide.

    A benchmark that cannot resolve a 10-point change cannot gate anything.
    """
    queries = [json.loads(l) for l in CORPUS_QUERIES.read_text().splitlines() if l.strip()]
    assert len(queries) >= 300, f"only {len(queries)} queries"
    assert len({q["paper_id"] for q in queries}) >= 30

    lo, hi = wilson_interval(int(0.3 * len(queries)), len(queries))
    assert (hi - lo) < 0.12, "interval still too wide to detect a 10-point change"


# ---------------------------------------------------------------------------
# Leak guard
# ---------------------------------------------------------------------------

@needs_corpus
def test_chunk_length_carries_no_signal_about_which_chunk_is_gold():
    """A query-independent ranker must not beat random ordering.

    Regression test for a real leak. ICT redacts the query sentence from its
    gold chunk; the chunker caps chunks at 110 words, so redacting only the gold
    left 88.9% of non-gold chunks sitting exactly at the cap and 0% of gold
    chunks there. Ranking purely by "shorter than the cap" — never looking at
    the query — scored nDCG@10 = 0.369 against BM25's 0.224, and a GBDT with a
    length feature reached 0.667 by learning the artifact and nothing else.

    `queries.redact_pool` removes one sentence from every candidate, which
    equalizes the distribution. This test fails if that guarantee breaks.
    """
    import random

    from eval.metrics import ndcg_at_k
    from eval.train_reranker import load_dataset, qrels_for, rank_with, split_by_paper
    from pipeline.generate.features import FEATURE_NAMES

    rows = load_dataset()
    _, _, test, _ = split_by_paper(rows)
    qrels = qrels_for(test)
    i_len = FEATURE_NAMES.index("chunk_tokens")

    by_length = rank_with(lambda X: -X[:, i_len], test)
    length_ndcg = sum(ndcg_at_k(by_length[q], qrels[q], 10) for q in qrels) / len(qrels)

    rng = random.Random(0)
    random_ndcg = sum(
        ndcg_at_k(
            rng.sample([c.chunk_id for c in r["candidates"]], k=min(10, len(r["candidates"]))),
            qrels[r["query_id"]], 10,
        )
        for r in test
    ) / len(test)

    print(f"\n  nDCG@10 by length alone = {length_ndcg:.4f} vs random {random_ndcg:.4f}")
    assert length_ndcg < random_ndcg * 1.5, (
        f"chunk length predicts relevance (nDCG@10={length_ndcg:.4f} vs random "
        f"{random_ndcg:.4f}). The redaction is leaking again — check redact_pool."
    )


@needs_corpus
def test_gold_and_non_gold_chunk_lengths_are_comparable():
    """The mechanism behind the leak, asserted directly."""
    from eval.train_reranker import load_dataset

    rows = load_dataset()
    gold, other = [], []
    for r in rows:
        for c in r["candidates"]:
            (gold if c.chunk_id == r["gold"] else other).append(len(c.content.split()))

    at_cap_gold = sum(1 for n in gold if n >= 110) / len(gold)
    at_cap_other = sum(1 for n in other if n >= 110) / len(other)
    print(f"\n  at-cap: gold={at_cap_gold:.1%} non-gold={at_cap_other:.1%}")
    assert abs(at_cap_gold - at_cap_other) < 0.10, (
        "gold and non-gold chunks differ in how often they hit the 110-word cap; "
        "that alone identifies the gold chunk"
    )
