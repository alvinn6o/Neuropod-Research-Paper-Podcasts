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
@pytest.mark.parametrize("config", ["shipped", "legacy", "bm25"])
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

    rows = load_dataset(verbose=False)
    _, test, _ = split_by_paper(rows)
    qrels = qrels_for(test)
    i_len = FEATURE_NAMES.index("chunk_tokens")

    # BOTH directions. An earlier version of this test only ranked shortest-
    # first and passed while longest-first scored 0.168 against random's 0.130 —
    # gold chunks had drifted 6.4 words longer than the rest. A one-sided test
    # for a two-sided property is not a test.
    shortest = rank_with(lambda X: -X[:, i_len], test)
    longest = rank_with(lambda X: X[:, i_len], test)
    length_ndcg = max(
        sum(ndcg_at_k(shortest[q], qrels[q], 10) for q in qrels) / len(qrels),
        sum(ndcg_at_k(longest[q], qrels[q], 10) for q in qrels) / len(qrels),
    )

    rng = random.Random(0)
    random_ndcg = sum(
        ndcg_at_k(
            rng.sample([c.chunk_id for c in r["candidates"]], k=min(10, len(r["candidates"]))),
            qrels[r["query_id"]], 10,
        )
        for r in test
    ) / len(test)

    print(f"\n  nDCG@10 by length (worst direction) = {length_ndcg:.4f} vs random {random_ndcg:.4f}")
    assert length_ndcg < random_ndcg * 1.25, (
        f"chunk length predicts relevance (nDCG@10={length_ndcg:.4f} vs random "
        f"{random_ndcg:.4f}). The redaction is leaking again — check redact_pool."
    )


@needs_corpus
def test_gold_and_non_gold_chunk_lengths_are_comparable():
    """The mechanism behind the leak, asserted directly."""
    from eval.train_reranker import load_dataset

    rows = load_dataset(verbose=False)
    gold, other = [], []
    for r in rows:
        for c in r["candidates"]:
            (gold if c.chunk_id == r["gold"] else other).append(len(c.content.split()))

    at_cap_gold = sum(1 for n in gold if n >= 110) / len(gold)
    at_cap_other = sum(1 for n in other if n >= 110) / len(other)
    mean_gap = abs(sum(gold) / len(gold) - sum(other) / len(other))
    print(f"\n  at-cap: gold={at_cap_gold:.1%} non-gold={at_cap_other:.1%}  mean gap={mean_gap:.1f}w")
    assert abs(at_cap_gold - at_cap_other) < 0.10, (
        "gold and non-gold chunks differ in how often they hit the 110-word cap; "
        "that alone identifies the gold chunk"
    )
    # The mean gap is the residual the redaction cannot fully remove: gold loses
    # its query sentence, others lose a length-matched sentence of their own.
    assert mean_gap < 8.0, f"gold chunks drift {mean_gap:.1f} words from the rest"


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

def test_features_are_deterministic_and_correctly_shaped():
    from pipeline.generate.features import FEATURE_NAMES, build_features

    chunks = [
        {"id": "a", "section": "results", "content": "We measure 5.2x throughput on A100 GPUs.", "chunk_index": 0},
        {"id": "b", "section": "introduction", "content": "Sequence models are widely studied.", "chunk_index": 1},
    ]
    first = build_features(chunks, "what throughput on A100?", dense_scores={"a": 0.7, "b": 0.1})
    second = build_features(chunks, "what throughput on A100?", dense_scores={"a": 0.7, "b": 0.1})
    assert [c.features for c in first] == [c.features for c in second]
    assert all(len(c.features) == len(FEATURE_NAMES) for c in first)
    assert all(all(isinstance(v, float) for v in c.features) for c in first)


def test_features_put_the_relevant_chunk_ahead_on_lexical_signals():
    from pipeline.generate.features import FEATURE_NAMES, build_features

    chunks = [
        {"id": "hit", "section": "results", "content": "Throughput reached 5.2x on A100 GPUs.", "chunk_index": 0},
        {"id": "miss", "section": "results", "content": "Unrelated discussion of dataset licensing.", "chunk_index": 1},
    ]
    feats = {c.chunk_id: dict(zip(FEATURE_NAMES, c.features))
             for c in build_features(chunks, "throughput on A100")}
    assert feats["hit"]["bm25"] > feats["miss"]["bm25"]
    assert feats["hit"]["number_match"] > feats["miss"]["number_match"]
    assert feats["hit"]["max_sent_overlap"] > feats["miss"]["max_sent_overlap"]


def test_section_one_hots_cover_the_hand_set_prior():
    """The learned model must be able to express the heuristic it replaces,
    otherwise 'learned beats hand-set' compares two different hypothesis spaces."""
    from pipeline.generate.features import SECTIONS
    from pipeline.generate.retriever import Retriever

    assert set(Retriever.section_bonus).issubset(set(SECTIONS))


@needs_corpus
def test_cv_folds_never_share_a_paper():
    """Grouping is the load-bearing discipline: queries from one paper share a
    chunk pool, so a query-level split lets a model memorize chunks it is then
    scored on."""
    from eval.train_reranker import load_dataset

    from eval.train_reranker import split_by_paper

    rows = load_dataset(verbose=False)
    papers = sorted({r["paper_id"] for r in rows})
    n_folds = 5
    folds = [set(papers[i::n_folds]) for i in range(n_folds)]

    for i, a in enumerate(folds):
        for b in folds[i + 1:]:
            assert not (a & b), "a paper appears in two folds"
    assert set().union(*folds) == set(papers)

    cv_pool, holdout_rows, holdout_papers = split_by_paper(rows)
    assert not ({r["paper_id"] for r in cv_pool} & holdout_papers)
    assert {r["paper_id"] for r in holdout_rows} == holdout_papers


def test_lightgbm_absence_degrades_instead_of_crashing(monkeypatch):
    """LightGBM is the one dependency pip can install correctly that still
    fails to import — it links a system OpenMP runtime."""
    import builtins

    from eval import train_reranker

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise OSError("dlopen failed: libomp.dylib not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert train_reranker._lightgbm() is None


# ---------------------------------------------------------------------------
# Shipped retriever behaviour (items #1 and #2)
# ---------------------------------------------------------------------------

def test_facet_queries_do_not_contain_paper_topic_terms():
    """Retrieval is scoped to one paper, so every candidate is already on-topic.

    Topical terms therefore carry no discriminating signal between chunks and
    would only dilute the facet cue. The queries exist to separate *sections of
    one paper*, not to find the paper.
    """
    from pipeline.generate.facets import build_facet_queries

    title, abstract = "Mamba: Linear-Time Sequence Modeling", "We introduce selective state spaces."
    queries = build_facet_queries(title, abstract)
    assert len(queries) == 4
    joined = " ".join(queries).lower()
    for term in ("mamba", "selective", "state spaces"):
        assert term not in joined


def test_zero_scoring_chunks_do_not_earn_fusion_credit():
    """RRF scores by rank, so a chunk that matched nothing must not be ranked.

    Otherwise a query with no lexical hits still produces a full ranking and the
    fused order is arbitrary.
    """
    from pipeline.generate.retriever import Retriever

    chunks = [
        {"id": "match", "section": "results", "content": "throughput on A100 gpus", "chunk_index": 0},
        {"id": "nomatch", "section": "methods", "content": "entirely unrelated prose", "chunk_index": 1},
    ]
    scored = Retriever().retrieve_scored(chunks, "throughput A100", limit=2)
    assert scored[0]["chunk"]["id"] == "match"
    assert scored[0]["final_score"] > 0.0
    assert scored[1]["final_score"] == 0.0


def test_ranking_is_invariant_to_input_order():
    """Tie-aware ranks make fusion independent of how the caller ordered chunks.

    Without averaged ranks for ties, two identically-scoring chunks are split by
    sort order, so the final ranking depends on input order rather than data.
    """
    from pipeline.generate.retriever import Retriever

    chunks = [
        {"id": "a", "section": "results", "content": "language models are widely studied", "chunk_index": 0},
        {"id": "b", "section": "introduction", "content": "language models are widely studied", "chunk_index": 1},
    ]
    forward = Retriever().retrieve_scored(chunks, "language models", limit=2)
    backward = Retriever().retrieve_scored(list(reversed(chunks)), "language models", limit=2)
    assert [r["final_score"] for r in forward] == [r["final_score"] for r in backward]


def test_reranker_is_off_unless_explicitly_enabled(monkeypatch):
    """A serving path that changes behaviour based on which files happen to
    exist is worse than one that requires an explicit switch."""
    from pipeline.generate.retriever import Retriever

    monkeypatch.delenv("NEUROPOD_RERANKER", raising=False)
    r = Retriever()
    assert r._load_reranker() is None


def test_missing_reranker_model_does_not_break_retrieval(monkeypatch, tmp_path):
    from pipeline.generate import rerank
    from pipeline.generate.retriever import Retriever

    monkeypatch.setenv("NEUROPOD_RERANKER", "on")
    monkeypatch.setattr(rerank, "LGBM_PATH", tmp_path / "absent.txt")
    monkeypatch.setattr(rerank, "LINEAR_PATH", tmp_path / "absent.json")

    chunks = [
        {"id": "a", "section": "results", "content": "alpha beta gamma", "chunk_index": 0},
        {"id": "b", "section": "methods", "content": "delta epsilon", "chunk_index": 1},
    ]
    out = Retriever().retrieve_scored(chunks, "alpha", limit=2)
    assert [r["chunk"]["id"] for r in out] == ["a", "b"]


@needs_corpus
def test_facet_queries_improve_section_coverage():
    """The production-query change, measured.

    The ICT benchmark cannot evaluate this: it supplies its own queries and so
    never exercises the query the pipeline actually sends. Coverage does.
    """
    import json as _json
    from collections import defaultdict

    from eval import queries as q_mod
    from eval.coverage import coverage_for
    from pipeline.generate.embedder import HashEmbedder
    from pipeline.generate.facets import build_facet_queries
    from pipeline.generate.retriever import PROMPT_CHUNK_LIMIT, Retriever

    manifest = {
        p["arxiv_id"]: p
        for p in _json.loads((ROOT / "eval" / "corpus" / "manifest.json").read_text())["papers"]
    }
    by_paper = defaultdict(list)
    for c in q_mod.load_chunks():
        by_paper[c["paper_id"]].append(c)

    emb = HashEmbedder()
    retriever = Retriever(embedder=emb)
    old_scores, new_scores = [], []
    for paper_id, chunks in sorted(by_paper.items())[:40]:
        meta = manifest.get(paper_id, {})
        for c in chunks:
            c["embedding"] = emb.embed_text(c["content"])
            c["embedding_model"] = emb.model_id
        available = {c["section"] for c in chunks}
        for store, qs in (
            (old_scores, [f"{meta.get('title','')} {meta.get('abstract','')}"]),
            (new_scores, build_facet_queries(meta.get("title", ""), meta.get("abstract", ""))),
        ):
            got = [r["chunk"] for r in retriever.retrieve_multi(chunks, qs, limit=PROMPT_CHUNK_LIMIT)]
            store.append(coverage_for(got, available)[0])

    old_mean = sum(old_scores) / len(old_scores)
    new_mean = sum(new_scores) / len(new_scores)
    print(f"\n  section coverage@14: title+abstract={old_mean:.3f} facets={new_mean:.3f}")
    assert new_mean > old_mean, "facet queries must not reduce section coverage"


# ---------------------------------------------------------------------------
# Cross-encoder harness
# ---------------------------------------------------------------------------

def test_cross_encoder_scoring_runs_without_a_downloaded_model():
    """Exercises the plumbing with a locally-built tiny BERT.

    The real model lives on HuggingFace, which is unreachable from CI and from
    the dev sandbox (403). Rather than skip the code path entirely, this builds
    a two-layer BERT from config — random weights, so the scores are
    meaningless, but batching, truncation and logit extraction are real.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    import tempfile
    from pathlib import Path as _Path

    from eval.cross_encoder import score_pairs

    tmp = _Path(tempfile.mkdtemp())
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok{i}" for i in range(200)]
    (tmp / "vocab.txt").write_text("\n".join(vocab))
    tok = transformers.BertTokenizerFast(vocab_file=str(tmp / "vocab.txt"))

    for num_labels in (1, 2):
        cfg = transformers.BertConfig(
            vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=64, num_labels=num_labels,
        )
        model = transformers.BertForSequenceClassification(cfg)
        model.eval()
        scores = score_pairs(tok, model, "a query", ["passage one", "passage two", "three"])
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)


def test_cross_encoder_module_does_not_use_sentence_transformers():
    """CVE-2026-68770 (CVSS 9.8) is a trust-gate bypass in sentence-transformers'
    `import_module_class`: an `os.path.exists` clause satisfies the gate
    regardless of `trust_remote_code=False`, so a malicious `modeling_*.py` in a
    model directory executes at import. We load via `transformers` directly."""
    source = (ROOT / "eval" / "cross_encoder.py").read_text()
    assert "sentence_transformers" not in source
    assert "trust_remote_code=False" in source
    assert "use_safetensors=True" in source


# ---------------------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------------------

TUNING = ROOT / "eval" / "tuning_results.json"
needs_tuning = pytest.mark.skipif(not TUNING.exists(), reason="tuning not run")


def test_nested_folds_never_share_a_paper():
    """Grouping has to hold at BOTH levels. An inner fold that shares a paper
    with its outer test set leaks the answer into hyperparameter selection."""
    from eval.tune import folds_by_paper, split_on

    rows = [{"paper_id": f"p{i}", "query_id": f"q{i}"} for i in range(20)]
    for held in folds_by_paper(rows, 5):
        outer_train, outer_test = split_on(rows, held)
        outer_test_papers = {r["paper_id"] for r in outer_test}
        assert not ({r["paper_id"] for r in outer_train} & outer_test_papers)
        for inner_held in folds_by_paper(outer_train, 3):
            inner_train, inner_test = split_on(outer_train, inner_held)
            assert not ({r["paper_id"] for r in inner_train} & {r["paper_id"] for r in inner_test})
            assert not ({r["paper_id"] for r in inner_test} & outer_test_papers)


def test_search_space_contains_the_incumbent_config():
    """The sweep must be able to re-select the hand-picked settings.

    If the incumbent is outside the search space, 'tuning improved things' can
    just mean 'the arbitrary alternative was worse'.
    """
    from eval.tune import BASELINE_CONFIG, GRID, sample_configs

    for key, value in BASELINE_CONFIG.items():
        assert value in GRID[key], f"{key}={value} is not in the search grid"
    assert sample_configs(6)[0] == BASELINE_CONFIG


@needs_tuning
def test_nested_estimate_is_not_above_the_non_nested_one():
    """The direction of selection bias, asserted.

    Choosing a winner on the same folds you report inflates the score. The
    nested estimate should therefore sit at or below the non-nested one; if it
    were meaningfully above, the nesting is wired wrong.
    """
    r = json.loads(TUNING.read_text())
    assert r["nested_ndcg"] <= r["nonnested_ndcg"] + 0.005, (
        f"nested ({r['nested_ndcg']:.4f}) exceeds non-nested "
        f"({r['nonnested_ndcg']:.4f}) — check the fold wiring"
    )
    assert r["selection_bias"] >= 0


@needs_tuning
def test_tuning_gain_is_reported_against_a_held_out_set():
    """A CV gain is a hypothesis; the holdout is the test of it.

    Here the CV gain was +0.017 and the holdout gain +0.002 (p=0.768). The
    point of this test is that both numbers exist and the holdout one is the
    reported conclusion.
    """
    r = json.loads(TUNING.read_text())
    assert "holdout" in r and {"hand-picked", "tuned"} <= set(r["holdout"])
    assert "holdout_paired" in r
    cv_gain = r["nested_ndcg"] - r["hand_picked_cv"]
    holdout_gain = r["holdout"]["tuned"] - r["holdout"]["hand-picked"]
    print(f"\n  CV gain {cv_gain:+.4f} vs holdout gain {holdout_gain:+.4f} "
          f"(p={r['holdout_paired']['p_value']:.3f})")
