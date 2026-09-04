"""Retrieval ablation harness.

Runs several retrieval configurations over the frozen corpus and reports each
metric with a confidence interval, plus a paired bootstrap against the shipping
baseline. The deliverable is the comparison table, not any single number.

Retrieval is scoped to one paper's chunks, matching production
(`Retriever.retrieve` is called with a single paper's chunk list). Evaluating
cross-corpus retrieval here would measure a system that does not exist.

Usage:
    python -m eval.harness                    # all configs, ICT queries
    python -m eval.harness --baseline current --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import queries as q_mod
from eval.metrics import (
    MetricSummary,
    evaluate_run,
    ndcg_at_k,
    paired_bootstrap,
    per_query_scores,
    reciprocal_rank,
)
from pipeline.generate.bm25 import BM25Index, reciprocal_rank_fusion
from pipeline.generate.embedder import HashEmbedder
from pipeline.generate.retriever import Retriever

CORPUS_DIR = ROOT / "eval" / "corpus"
RESULTS = CORPUS_DIR / "results.json"

TOP_N = 20  # ranking depth kept for metric computation


# ---------------------------------------------------------------------------
# Retrieval configurations. Each takes (chunks, query) and returns ranked ids.
# ---------------------------------------------------------------------------

def _dense_scores(chunks: list[dict], query: str, embedder: HashEmbedder) -> dict[str, float]:
    qv = embedder.embed_text(query)
    out = {}
    for c in chunks:
        v = c.get("_emb")
        out[c["id"]] = sum(a * b for a, b in zip(qv, v)) if v else 0.0
    return out


def config_current(chunks, query, embedder):
    """The shipping retriever: raw term-frequency cosine + additive section prior."""
    return [r["chunk"]["id"] for r in Retriever().retrieve_scored(chunks, query, limit=TOP_N)]


def config_dense_prior(chunks, query, embedder):
    """Shipping dense path: hash-embedding cosine + the same additive prior."""
    r = Retriever(embedder=embedder)
    for c in chunks:
        c["embedding"] = c.get("_emb") or []
    scored = r.retrieve_scored(chunks, query, limit=TOP_N)
    return [row["chunk"]["id"] for row in scored]


def config_bm25(chunks, query, embedder):
    idx = BM25Index.build([(c["id"], c["content"]) for c in chunks])
    return [cid for cid, _ in idx.top_k(query, TOP_N)]


def config_dense_only(chunks, query, embedder):
    scores = _dense_scores(chunks, query, embedder)
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N]]


def config_rrf(chunks, query, embedder):
    """RRF over BM25 and dense. No score normalization, no tuned weights."""
    idx = BM25Index.build([(c["id"], c["content"]) for c in chunks])
    bm = [cid for cid, _ in idx.top_k(query, TOP_N)]
    dn = config_dense_only(chunks, query, embedder)
    return [cid for cid, _ in reciprocal_rank_fusion([bm, dn])[:TOP_N]]


def config_rrf_prior(chunks, query, embedder):
    """RRF, then the hand-set section prior applied to the fused score.

    Included to test the prior on a scale where it is not obviously dominant:
    RRF scores are ~1/60, so a +0.18 additive prior would swamp them entirely.
    Scaled to the fused range instead, which is the charitable version.
    """
    idx = BM25Index.build([(c["id"], c["content"]) for c in chunks])
    bm = [cid for cid, _ in idx.top_k(query, TOP_N)]
    dn = config_dense_only(chunks, query, embedder)
    fused = dict(reciprocal_rank_fusion([bm, dn]))
    section = {c["id"]: c["section"] for c in chunks}
    span = max(fused.values()) - min(fused.values()) if fused else 0.0
    boosted = {
        cid: s + Retriever.section_bonus.get(section.get(cid, ""), 0.0) * span
        for cid, s in fused.items()
    }
    return [cid for cid, _ in sorted(boosted.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N]]


CONFIGS: dict[str, Callable] = {
    "current": config_current,
    "bm25": config_bm25,
    "dense": config_dense_only,
    "dense+prior": config_dense_prior,
    "rrf": config_rrf,
    "rrf+prior": config_rrf_prior,
}


def run(config_names: list[str], *, per_paper_limit: int | None = None) -> dict:
    chunks = q_mod.load_chunks()
    queries = q_mod.read(q_mod.ICT_QUERIES)
    if not queries:
        raise SystemExit("no queries — run `python -m eval.queries --mode ict`")

    embedder = HashEmbedder()
    # Deterministic and cached once: recomputing per query would dominate runtime.
    emb_cache = {c["id"]: embedder.embed_text(c["content"]) for c in chunks}

    by_paper: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_paper[c["paper_id"]].append(c)

    redacted = q_mod.redact_gold(chunks, queries)  # gold only, for the length check
    runs: dict[str, dict[str, list[str]]] = {name: {} for name in config_names}
    qrels: dict[str, dict[str, int]] = {}
    query_meta: dict[str, dict] = {}

    for query in queries:
        gold = redacted.get(query.query_id)
        if gold is None or len(gold["content"].split()) < 5:
            # The gold chunk was (almost) entirely the query sentence, so after
            # redaction nothing is left to retrieve. Excluded rather than
            # counted as a miss — it is not a retrieval failure.
            continue
        # One sentence removed from EVERY chunk, not just gold. Redacting only
        # the gold chunk makes it identifiable by length alone (see
        # queries.redact_pool), which is a leak worth more nDCG than BM25.
        paper_chunks = q_mod.redact_pool(by_paper[query.paper_id], query)
        for c in paper_chunks:
            # Every chunk's text changed, so no embedding can come from cache.
            c["_emb"] = embedder.embed_text(c["content"])

        qrels[query.query_id] = {query.gold_chunk_id: 2}
        query_meta[query.query_id] = {"section": query.section, "paper_id": query.paper_id}
        for name in config_names:
            runs[name][query.query_id] = CONFIGS[name](paper_chunks, query.query, embedder)

    return {"runs": runs, "qrels": qrels, "meta": query_meta,
            "n_queries": len(qrels), "n_papers": len(by_paper)}


def report(result: dict, baseline: str) -> dict:
    runs, qrels, meta = result["runs"], result["qrels"], result["meta"]
    out: dict = {"n_queries": result["n_queries"], "n_papers": result["n_papers"], "configs": {}}

    summaries = {name: evaluate_run(run, qrels) for name, run in runs.items()}

    print(f"\nCorpus: {result['n_papers']} papers, {result['n_queries']} ICT queries")
    print("=" * 92)
    print(f"{'config':<14} {'nDCG@10':>22} {'hit@1':>20} {'MRR':>18}")
    print("-" * 92)
    for name in runs:
        s = summaries[name]
        def fmt(m: MetricSummary) -> str:
            return f"{m.value:.3f} [{m.ci_low:.3f},{m.ci_high:.3f}]"
        print(f"{name:<14} {fmt(s['ndcg@10']):>22} {fmt(s['hit@1']):>20} {fmt(s['mrr']):>18}")
        out["configs"][name] = {k: vars(v) for k, v in s.items()}

    # Paired comparisons against the shipping baseline.
    print("\nPaired bootstrap vs '%s' (nDCG@10, 1000 resamples)" % baseline)
    print("-" * 92)
    base_ids, base_scores = per_query_scores(
        runs[baseline], qrels, lambda r, rel: ndcg_at_k(r, rel, 10)
    )
    out["paired_vs_baseline"] = {}
    for name in runs:
        if name == baseline:
            continue
        ids, scores = per_query_scores(runs[name], qrels, lambda r, rel: ndcg_at_k(r, rel, 10))
        assert ids == base_ids
        pr = paired_bootstrap(base_scores, scores)
        verdict = "significant" if pr.significant else "not significant"
        sign = "+" if pr.delta >= 0 else ""
        print(f"  {name:<14} delta={sign}{pr.delta:.4f}  "
              f"95% CI [{pr.ci_low:+.4f},{pr.ci_high:+.4f}]  p={pr.p_value:.3f}  {verdict}")
        out["paired_vs_baseline"][name] = vars(pr)

    # Per-section slice on the best config: an aggregate hides where it fails.
    best = max(runs, key=lambda n: summaries[n]["ndcg@10"].value)
    by_section: dict[str, list[float]] = defaultdict(list)
    for qid, rel in qrels.items():
        by_section[meta[qid]["section"]].append(ndcg_at_k(runs[best][qid], rel, 10))
    print(f"\nnDCG@10 by section for best config ('{best}')")
    print("-" * 92)
    out["by_section"] = {}
    for section, vals in sorted(by_section.items(), key=lambda kv: -len(kv[1])):
        mean = sum(vals) / len(vals)
        out["by_section"][section] = {"ndcg@10": mean, "n": len(vals)}
        print(f"  {section:<16} {mean:.3f}   n={len(vals)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--baseline", default="current")
    ap.add_argument("--json", default=str(RESULTS))
    args = ap.parse_args()

    res = run(args.configs)
    summary = report(res, args.baseline)
    Path(args.json).write_text(json.dumps(summary, indent=2))
    print(f"\nwritten -> {args.json}")
