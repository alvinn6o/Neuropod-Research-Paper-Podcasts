"""Train a learned reranker to replace the hand-set section prior.

Motivation is measured, not assumed. `Retriever.section_bonus` adds a constant
per section (abstract 0.18, results 0.16, ...) straight onto a cosine score, and
the Phase 1 ablation showed it is significantly *harmful*: dense -> dense+prior
costs 0.027 nDCG@10, CI [-0.043, -0.012], p<0.001. Those weights were never fit
to anything. This fits them.

Splits are grouped by PAPER, never by query. Queries from one paper share the
same chunk pool, so a query-level split leaks the answer: a model could memorize
"chunk 17 of paper X is a good answer" from the train fold and be scored on it
in test. Grouping by paper is the difference between a real held-out number and
a flattering one.

The test fold is scored once, at the end. Model selection happens on dev.

Usage:
    python -m eval.train_reranker
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import queries as q_mod
from eval.metrics import ndcg_at_k, paired_bootstrap, summarize
from pipeline.generate.bm25 import BM25Index
from pipeline.generate.embedder import HashEmbedder
from pipeline.generate.features import FEATURE_NAMES, SECTIONS, build_features
from pipeline.generate.retriever import Retriever

MODEL_PATH = ROOT / "eval" / "reranker.json"
RESULTS_PATH = ROOT / "eval" / "reranker_results.json"

# Papers, not queries. Sorted then sliced so the split is reproducible.
TRAIN_FRAC, DEV_FRAC = 0.60, 0.20


def load_dataset():
    chunks = q_mod.load_chunks()
    queries = q_mod.read(q_mod.ICT_QUERIES)
    if not queries:
        raise SystemExit("no queries — run `python -m eval.queries --mode ict`")

    embedder = HashEmbedder()
    by_paper = defaultdict(list)
    for c in chunks:
        by_paper[c["paper_id"]].append(c)
    redacted = q_mod.redact_gold(chunks, queries)

    rows = []
    for query in queries:
        gold = redacted.get(query.query_id)
        if gold is None or len(gold["content"].split()) < 5:
            continue
        paper_chunks = q_mod.redact_pool(by_paper[query.paper_id], query)
        qv = embedder.embed_text(query.query)
        dense = {
            c["id"]: float(np.dot(qv, embedder.embed_text(c["content"])))
            for c in paper_chunks
        }

        cands = build_features(paper_chunks, query.query, dense_scores=dense)
        rows.append({
            "query_id": query.query_id,
            "paper_id": query.paper_id,
            "query": query.query,
            "gold": query.gold_chunk_id,
            "candidates": cands,
        })
    return rows


def split_by_paper(rows):
    papers = sorted({r["paper_id"] for r in rows})
    n = len(papers)
    n_train = int(n * TRAIN_FRAC)
    n_dev = int(n * DEV_FRAC)
    train_p = set(papers[:n_train])
    dev_p = set(papers[n_train:n_train + n_dev])
    test_p = set(papers[n_train + n_dev:])
    assert not (train_p & dev_p) and not (train_p & test_p) and not (dev_p & test_p)
    return (
        [r for r in rows if r["paper_id"] in train_p],
        [r for r in rows if r["paper_id"] in dev_p],
        [r for r in rows if r["paper_id"] in test_p],
        (train_p, dev_p, test_p),
    )


def to_matrix(rows, *, negatives_per_query: int | None = None, seed: int = 0):
    """Flatten to (X, y, group_ids).

    `negatives_per_query=None` keeps every candidate; an int keeps the gold plus
    that many highest-BM25 non-gold chunks.

    Subsampling has a sharp cost that is easy to miss: the model then trains on
    a different candidate distribution than it is served, since evaluation
    always uses the full pool. Measured on dev, top-20 hard negatives scores
    *below random* while the full pool does not — the model never saw the easy
    negatives it is asked to rank at serving time and happily scores them high.
    This is train/serve skew, not a modelling failure.
    """
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for r in rows:
        cands = r["candidates"]
        gold_idx = [i for i, c in enumerate(cands) if c.chunk_id == r["gold"]]
        if not gold_idx:
            continue
        keep = set(gold_idx)
        if negatives_per_query is None:
            keep.update(range(len(cands)))          # every candidate
        else:
            neg = [i for i in range(len(cands)) if i not in keep]
            # Hard negatives: the highest-BM25 non-gold chunks. Random negatives
            # are trivially separable and teach the model almost nothing.
            bm = FEATURE_NAMES.index("bm25")
            neg.sort(key=lambda i: cands[i].features[bm], reverse=True)
            keep.update(neg[:negatives_per_query])
        for i in sorted(keep):
            X.append(cands[i].features)
            y.append(1 if i in gold_idx else 0)
            groups.append(r["query_id"])
    return np.asarray(X, dtype=np.float64), np.asarray(y), groups


def rank_with(scorer, rows, limit: int = 20):
    """Score every candidate and return {query_id: ranked chunk ids}."""
    run = {}
    for r in rows:
        X = np.asarray([c.features for c in r["candidates"]], dtype=np.float64)
        scores = scorer(X)
        order = np.argsort(-scores)
        run[r["query_id"]] = [r["candidates"][i].chunk_id for i in order[:limit]]
    return run


def baseline_runs(rows, limit: int = 20):
    """Reference systems, recomputed from the same candidate objects."""
    bm25_i = FEATURE_NAMES.index("bm25")
    dense_i = FEATURE_NAMES.index("dense")
    runs = {"bm25": {}, "dense+prior": {}}
    for r in rows:
        cands = r["candidates"]
        bm = np.asarray([c.features[bm25_i] for c in cands])
        runs["bm25"][r["query_id"]] = [
            cands[i].chunk_id for i in np.argsort(-bm)[:limit]
        ]
        prior = np.asarray([
            c.features[dense_i] + Retriever.section_bonus.get(c.section, 0.0) for c in cands
        ])
        runs["dense+prior"][r["query_id"]] = [
            cands[i].chunk_id for i in np.argsort(-prior)[:limit]
        ]
    return runs


def qrels_for(rows):
    return {r["query_id"]: {r["gold"]: 2} for r in rows}


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("building features...")
    rows = load_dataset()
    train, dev, test, (tp, dp, sp) = split_by_paper(rows)
    print(f"  {len(rows)} queries over {len({r['paper_id'] for r in rows})} papers")
    print(f"  train {len(train)}q/{len(tp)}p   dev {len(dev)}q/{len(dp)}p   test {len(test)}q/{len(sp)}p")
    print("  split is by PAPER — no paper appears in two folds")

    # Full candidate pool, not subsampled negatives. Selected on dev: top-20
    # hard negatives scored 0.264 and the full pool 0.443 (gbdt), because
    # subsampling trains on a different distribution than serving uses.
    Xtr, ytr, _ = to_matrix(train, negatives_per_query=None, seed=0)
    print(f"  train matrix {Xtr.shape}, positives {int(ytr.sum())} ({ytr.mean():.1%})")

    scaler = StandardScaler().fit(Xtr)

    linear = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    linear.fit(scaler.transform(Xtr), ytr)

    gbdt = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0, random_state=0,
    )
    gbdt.fit(Xtr, ytr)

    models = {
        "linear": lambda X: linear.decision_function(scaler.transform(X)),
        "gbdt": lambda X: gbdt.predict_proba(X)[:, 1],
    }

    # ---- dev: choose the model here, not on test ----
    print("\ndev (model selection)")
    print("-" * 70)
    dev_qrels = qrels_for(dev)
    dev_runs = {**baseline_runs(dev), **{n: rank_with(f, dev) for n, f in models.items()}}
    dev_scores = {}
    for name, run in dev_runs.items():
        vals = [ndcg_at_k(run[q], dev_qrels[q], 10) for q in sorted(dev_qrels)]
        dev_scores[name] = sum(vals) / len(vals)
        print(f"  {name:<14} nDCG@10 = {dev_scores[name]:.4f}")
    best = max(models, key=lambda n: dev_scores[n])
    print(f"  -> selected '{best}' on dev")

    # ---- test: touched once ----
    print("\ntest (held-out papers, scored once)")
    print("=" * 70)
    test_qrels = qrels_for(test)
    test_runs = {**baseline_runs(test), **{n: rank_with(f, test) for n, f in models.items()}}
    per_query = {}
    for name, run in test_runs.items():
        vals = [ndcg_at_k(run[q], test_qrels[q], 10) for q in sorted(test_qrels)]
        per_query[name] = vals
        s = summarize(f"{name} nDCG@10", vals)
        print(f"  {s}")

    print("\npaired bootstrap vs bm25 (test)")
    print("-" * 70)
    results = {"dev": dev_scores, "test": {}, "paired": {}}
    for name in test_runs:
        if name == "bm25":
            continue
        pr = paired_bootstrap(per_query["bm25"], per_query[name])
        verdict = "significant" if pr.significant else "not significant"
        print(f"  {name:<14} delta={pr.delta:+.4f} CI[{pr.ci_low:+.4f},{pr.ci_high:+.4f}] "
              f"p={pr.p_value:.3f}  {verdict}")
        results["paired"][name] = vars(pr)
    for name, vals in per_query.items():
        results["test"][name] = {"ndcg@10": sum(vals) / len(vals), "n": len(vals)}

    # ---- what the linear model learned about sections ----
    print("\nlearned section weights vs the hand-set prior")
    print("-" * 70)
    print(f"  {'section':<14} {'hand-set':>10} {'learned':>10}")
    coefs = dict(zip(FEATURE_NAMES, linear.coef_[0]))
    section_report = {}
    for s in SECTIONS:
        hand = Retriever.section_bonus.get(s, 0.0)
        learned = coefs[f"section_{s}"]
        section_report[s] = {"hand_set": hand, "learned": float(learned)}
        print(f"  {s:<14} {hand:>10.3f} {learned:>10.3f}")
    results["section_weights"] = section_report

    print("\ntop features by |coefficient| (linear)")
    print("-" * 70)
    ranked = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:8]
    for name, w in ranked:
        print(f"  {name:<22} {w:+.3f}")
    results["coefficients"] = {k: float(v) for k, v in coefs.items()}

    MODEL_PATH.write_text(json.dumps({
        "model": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "coef": [float(c) for c in linear.coef_[0]],
        "intercept": float(linear.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "trained_on": {"papers": sorted(tp), "queries": len(train)},
    }, indent=2))
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nmodel  -> {MODEL_PATH}")
    print(f"results-> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
