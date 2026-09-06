"""Cross-encoder reranking: the quality/latency frontier.

Answers "why not just use a transformer?" with a measurement instead of an
opinion.

The distinction that matters is not transformer-vs-trees, it is **bi-encoder vs
cross-encoder**. BM25 and dense retrieval score a query and a passage
*independently* and compare the results, which is why they can pre-compute an
index. A cross-encoder feeds `[CLS] query [SEP] passage [SEP]` through one
network so attention runs across both — it can see that "5.2x throughput"
answers "how much faster", which no independent scoring can. The cost is that
nothing is pre-computable: every (query, passage) pair is a forward pass.

That is the whole trade, and it is why real systems cascade rather than choose:
cheap retrieval narrows thousands to ~50, the expensive model reorders those.

Security note: this deliberately does NOT use `sentence-transformers`.
CVE-2026-68770 (CVSS 9.8, published 2026-07-31) is a logic flaw in its
`import_module_class`, where an `os.path.exists(...)` clause satisfies the trust
gate regardless of `trust_remote_code=False` — so a malicious `modeling_*.py`
inside a downloaded model directory executes at import. We load through
`transformers` directly with `trust_remote_code=False` and safetensors, which
does not go near that code path.

    python -m eval.cross_encoder --papers 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import queries as q_mod
from eval.metrics import ndcg_at_k, paired_bootstrap, summarize
from eval.train_reranker import (
    fit_models,
    load_dataset,
    ndcg_of,
    qrels_for,
    rank_with,
    split_by_paper,
)
from pipeline.generate.features import FEATURE_NAMES

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RESULTS = ROOT / "eval" / "cross_encoder_results.json"

# Rerank depth. The cascade's whole point: the cross-encoder never sees the
# full candidate pool, only what cheap retrieval already narrowed to.
RERANK_DEPTH = 50


def load_model():
    """Pretrained, not fine-tuned.

    Fine-tuning on 1,935 ICT queries risks learning the quirks of an artificial
    query distribution, and the frontier — not a peak score — is what this is
    for. Zero-shot is also the honest comparison: it is what you get for free.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, trust_remote_code=False, use_safetensors=True
    )
    model.eval()
    torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4) // 2))
    return tok, model


def score_pairs(tok, model, query: str, passages: list[str], batch: int = 16) -> list[float]:
    import torch

    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(passages), batch):
            enc = tok(
                [query] * len(passages[i : i + batch]),
                passages[i : i + batch],
                padding=True, truncation=True, max_length=320, return_tensors="pt",
            )
            logits = model(**enc).logits
            out.extend(logits[:, 0].tolist() if logits.shape[-1] == 1
                       else logits[:, -1].tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=30,
                    help="holdout papers to score (cross-encoder is slow on CPU)")
    ap.add_argument("--depth", type=int, default=RERANK_DEPTH)
    args = ap.parse_args()

    print("building features...")
    rows = load_dataset(verbose=False)
    dev, hold, _ = split_by_paper(rows)

    papers = sorted({r["paper_id"] for r in hold})[: args.papers]
    subset = [r for r in hold if r["paper_id"] in set(papers)]
    print(f"  {len(subset)} queries over {len(papers)} held-out papers")

    print("training the feature-based models on the same split...")
    models, _ = fit_models(dev, use_lambdamart=True)
    qrels = qrels_for(subset)

    bm_i = FEATURE_NAMES.index("bm25")
    runs, timings = {}, {}

    for name, scorer in (
        ("bm25", lambda X: X[:, bm_i]),
        ("lambdamart", models["lambdamart"]),
    ):
        t0 = time.perf_counter()
        runs[name] = rank_with(scorer, subset)
        timings[name] = (time.perf_counter() - t0) / len(subset) * 1000

    print(f"loading {MODEL_NAME} ...")
    tok, model = load_model()

    # The cascade: BM25 gives the candidates, the cross-encoder reorders them.
    print(f"scoring {len(subset)} queries x top-{args.depth} pairs on CPU...")
    ce_run: dict[str, list[str]] = {}
    t0 = time.perf_counter()
    for i, r in enumerate(subset):
        if i and i % 100 == 0:
            print(f"    {i}/{len(subset)}", flush=True)
        X = np.asarray([c.features for c in r["candidates"]], dtype=np.float64)
        order = np.argsort(-X[:, bm_i])[: args.depth]
        cands = [r["candidates"][j] for j in order]
        scores = score_pairs(tok, model, r["query_text"], [c.content for c in cands])
        ranked = [c.chunk_id for _, c in sorted(zip(scores, cands), key=lambda kv: -kv[0])]
        ce_run[r["query_id"]] = ranked
    timings["cross-encoder"] = (time.perf_counter() - t0) / len(subset) * 1000
    runs["cross-encoder"] = ce_run

    print(f"\nQuality / latency frontier — {len(subset)} queries, {len(papers)} held-out papers")
    print("=" * 84)
    print(f"  {'model':<16}{'nDCG@10':>26}{'ms/query':>12}{'vs BM25 quality':>18}")
    print("  " + "-" * 80)
    per_query = {n: ndcg_of(run, qrels) for n, run in runs.items()}
    base = float(np.mean(per_query["bm25"]))
    out: dict = {"n_queries": len(subset), "n_papers": len(papers), "depth": args.depth,
                 "model": MODEL_NAME, "results": {}}
    for name in ("bm25", "lambdamart", "cross-encoder"):
        s = summarize(name, per_query[name])
        delta = s.value - base
        print(f"  {name:<16}{s.value:>10.4f} [{s.ci_low:.3f},{s.ci_high:.3f}]"
              f"{timings[name]:>12.1f}{delta:>+18.4f}")
        out["results"][name] = {"ndcg@10": s.value, "ci_low": s.ci_low,
                                "ci_high": s.ci_high, "ms_per_query": timings[name]}

    print("\npaired bootstrap")
    print("  " + "-" * 80)
    for a, b in (("bm25", "cross-encoder"), ("bm25", "lambdamart"),
                 ("lambdamart", "cross-encoder")):
        pr = paired_bootstrap(per_query[a], per_query[b])
        print(f"  {a:<14} -> {b:<16} delta={pr.delta:+.4f} "
              f"CI[{pr.ci_low:+.4f},{pr.ci_high:+.4f}] p={pr.p_value:.3f} "
              f"{'significant' if pr.significant else 'n.s.'}")
        out.setdefault("paired", {})[f"{a}->{b}"] = vars(pr)

    ce_ms, lm_ms = timings["cross-encoder"], timings["lambdamart"]
    print(f"\n  Cost of the cross-encoder: {ce_ms / max(lm_ms, 1e-9):.0f}x the latency of")
    print(f"  LambdaMART ({ce_ms:.0f} ms vs {lm_ms:.1f} ms per query, CPU, depth {args.depth}).")

    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwritten -> {RESULTS}")


if __name__ == "__main__":
    main()
