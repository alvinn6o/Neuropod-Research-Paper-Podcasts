"""Train and evaluate a learned reranker.

Motivation is measured. `Retriever.section_bonus` adds a hand-set constant per
section straight onto a cosine score, and the ablation shows it is significantly
harmful (dense -> dense+prior = -0.041 nDCG@10, p<0.001). Those weights were
never fit to anything. This fits them.

Two disciplines matter more here than the model choice:

**Grouping by paper, never by query.** Queries from one paper share a chunk
pool, so a query-level split lets a model memorize "chunk 17 of paper X is a
good answer" in train and be rewarded for it in test.

**Cross-validation, not one split.** With tens of papers a single split has
enormous variance — an earlier version showed dev 0.443 vs test 0.298, a gap
larger than the margin being claimed. GroupKFold over papers gives a mean and a
spread across folds, so "better" is a claim about the corpus rather than about
one lucky partition. A final held-out set is still kept and scored once.

Usage:
    python -m eval.train_reranker
    python -m eval.train_reranker --no-lambdamart   # skip LightGBM
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import queries as q_mod
from eval.metrics import ndcg_at_k, paired_bootstrap, summarize
from pipeline.generate.embedder import HashEmbedder
from pipeline.generate.features import FEATURE_NAMES, SECTIONS, build_features
from pipeline.generate.retriever import Retriever

MODEL_PATH = ROOT / "eval" / "reranker.json"
RESULTS_PATH = ROOT / "eval" / "reranker_results.json"

HOLDOUT_FRAC = 0.20
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(verbose: bool = True):
    chunks = q_mod.load_chunks()
    queries = q_mod.read(q_mod.ICT_QUERIES)
    if not queries:
        raise SystemExit("no queries — run `python -m eval.queries --mode ict`")

    embedder = HashEmbedder()
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_paper[c["paper_id"]].append(c)

    # Redaction is seeded per chunk, so each chunk has one canonical redacted
    # form and its embedding is computed once rather than once per query.
    pool_cache: dict[str, list[dict]] = {}
    emb_cache: dict[str, np.ndarray] = {}
    for paper_id, paper_chunks in by_paper.items():
        dummy = q_mod.EvalQuery("", paper_id, "", "", "ict", "")
        pool = q_mod.redact_pool(paper_chunks, dummy)
        pool_cache[paper_id] = pool
        for c in pool:
            emb_cache[c["id"]] = np.asarray(embedder.embed_text(c["content"]))

    rows = []
    for i, query in enumerate(queries):
        if verbose and i and i % 500 == 0:
            print(f"    {i}/{len(queries)} queries featurized", flush=True)
        pool = pool_cache.get(query.paper_id)
        if not pool:
            continue
        # Only the gold chunk differs from the cached pool: it loses the query
        # sentence specifically rather than its canonical one.
        gold_txt = None
        paper_chunks = []
        for c in pool:
            if c["id"] == query.gold_chunk_id:
                base = next(x for x in by_paper[query.paper_id] if x["id"] == c["id"])
                gold_txt = q_mod._strip_sentence(base["content"], query.query)
                paper_chunks.append({**c, "content": gold_txt})
            else:
                paper_chunks.append(c)
        if gold_txt is None or len(gold_txt.split()) < 5:
            continue

        qv = np.asarray(embedder.embed_text(query.query))
        dense = {}
        for c in paper_chunks:
            v = (np.asarray(embedder.embed_text(c["content"]))
                 if c["id"] == query.gold_chunk_id else emb_cache[c["id"]])
            dense[c["id"]] = float(qv @ v)

        rows.append({
            "query_id": query.query_id,
            "paper_id": query.paper_id,
            "gold": query.gold_chunk_id,
            "candidates": build_features(paper_chunks, query.query, dense_scores=dense),
        })
    return rows


def to_matrix(rows):
    """Flatten to (X, y, group_sizes, query_ids). Full candidate pool.

    No negative subsampling: measured on dev, top-20 hard negatives scored 0.264
    against 0.443 for the full pool. Subsampling trains on a distribution the
    model is never served — at inference it ranks every candidate, including
    easy negatives it never saw, and scores them confidently. Train/serve skew.
    """
    X, y, groups, qids = [], [], [], []
    for r in rows:
        n = 0
        for c in r["candidates"]:
            X.append(c.features)
            y.append(1 if c.chunk_id == r["gold"] else 0)
            n += 1
        groups.append(n)
        qids.append(r["query_id"])
    return np.asarray(X, dtype=np.float64), np.asarray(y), groups, qids


def qrels_for(rows):
    return {r["query_id"]: {r["gold"]: 2} for r in rows}


def split_by_paper(rows, holdout_frac: float = HOLDOUT_FRAC):
    """(cv_pool, holdout, holdout_paper_ids), partitioned by PAPER.

    One helper so main() and the tests cannot disagree about what the holdout
    is — a split defined twice is a split that eventually differs.
    """
    papers = sorted({r["paper_id"] for r in rows})
    n_hold = max(1, int(len(papers) * holdout_frac))
    holdout = set(papers[-n_hold:])
    return (
        [r for r in rows if r["paper_id"] not in holdout],
        [r for r in rows if r["paper_id"] in holdout],
        holdout,
    )


def rank_with(scorer, rows, limit: int = 20):
    run = {}
    for r in rows:
        X = np.asarray([c.features for c in r["candidates"]], dtype=np.float64)
        order = np.argsort(-scorer(X))
        run[r["query_id"]] = [r["candidates"][i].chunk_id for i in order[:limit]]
    return run


def baseline_runs(rows, limit: int = 20):
    bm_i, dn_i = FEATURE_NAMES.index("bm25"), FEATURE_NAMES.index("dense")
    runs = {"bm25": {}, "dense+prior": {}}
    for r in rows:
        cands = r["candidates"]
        bm = np.asarray([c.features[bm_i] for c in cands])
        runs["bm25"][r["query_id"]] = [cands[i].chunk_id for i in np.argsort(-bm)[:limit]]
        pr = np.asarray([c.features[dn_i] + Retriever.section_bonus.get(c.section, 0.0)
                         for c in cands])
        runs["dense+prior"][r["query_id"]] = [cands[i].chunk_id for i in np.argsort(-pr)[:limit]]
    return runs


def ndcg_of(run, qrels, k: int = 10) -> list[float]:
    return [ndcg_at_k(run[q], qrels[q], k) for q in sorted(qrels) if q in run]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _lightgbm():
    """Import LightGBM, or return None with an actionable message.

    LightGBM links against a system OpenMP runtime. Linux wheels bundle it, but
    on macOS it needs `brew install libomp` — so this is the one dependency that
    can be installed correctly by pip and still fail to import. Degrading to the
    sklearn models keeps the pipeline runnable rather than erroring on a machine
    that just hasn't run brew yet.
    """
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        print("  lightgbm not installed — skipping LambdaMART")
    except OSError as exc:
        print(f"  lightgbm present but its OpenMP runtime is missing ({exc.__class__.__name__}).")
        print("  macOS: brew install libomp    Debian/Ubuntu: apt-get install libgomp1")
    return None


def fit_models(train_rows, *, use_lambdamart: bool):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y, groups, _ = to_matrix(train_rows)
    scaler = StandardScaler().fit(X)

    linear = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    linear.fit(scaler.transform(X), y)

    gbdt = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0, random_state=0,
    ).fit(X, y)

    models = {
        "linear": lambda Z: linear.decision_function(scaler.transform(Z)),
        "gbdt": lambda Z: gbdt.predict_proba(Z)[:, 1],
    }
    extra = {"linear_obj": linear, "scaler": scaler}

    if use_lambdamart:
        lgb = _lightgbm()
        if lgb is None:
            return models, extra
        # LambdaMART: optimizes NDCG directly over query groups, rather than
        # classifying each candidate independently and hoping the induced order
        # is good. The group structure is the whole point — it is the only one
        # of these three that knows candidates compete within a query.
        ranker = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg",
            n_estimators=300, learning_rate=0.06, num_leaves=15,
            min_child_samples=20, reg_lambda=1.0, random_state=0, verbose=-1,
        )
        ranker.fit(X, y, group=groups)
        models["lambdamart"] = lambda Z: ranker.predict(Z)
        extra["lambdamart_obj"] = ranker
    return models, extra


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def cross_validate(rows, *, use_lambdamart: bool, n_folds: int = N_FOLDS):
    """GroupKFold over papers. Returns {model: [fold means]}."""
    papers = sorted({r["paper_id"] for r in rows})
    folds = [set(papers[i::n_folds]) for i in range(n_folds)]
    scores: dict[str, list[float]] = defaultdict(list)

    for i, held in enumerate(folds, start=1):
        tr = [r for r in rows if r["paper_id"] not in held]
        te = [r for r in rows if r["paper_id"] in held]
        if not te or not tr:
            continue
        qrels = qrels_for(te)
        models, _ = fit_models(tr, use_lambdamart=use_lambdamart)
        runs = {**baseline_runs(te), **{n: rank_with(f, te) for n, f in models.items()}}
        for name, run in runs.items():
            scores[name].append(float(np.mean(ndcg_of(run, qrels))))
        print(f"    fold {i}/{n_folds}: {len(tr)} train / {len(te)} test queries", flush=True)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lambdamart", action="store_true")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()
    use_lm = not args.no_lambdamart

    print("building features...")
    rows = load_dataset()
    papers = sorted({r["paper_id"] for r in rows})
    dev_rows, hold_rows, holdout_papers = split_by_paper(rows)
    n_hold = len(holdout_papers)

    print(f"  {len(rows)} queries over {len(papers)} papers")
    print(f"  cv pool: {len(dev_rows)}q/{len(papers)-n_hold}p    holdout: {len(hold_rows)}q/{n_hold}p")
    print(f"  grouping is by PAPER throughout — no paper spans two folds")

    print(f"\ncross-validation ({args.folds}-fold, grouped by paper)")
    print("=" * 78)
    cv = cross_validate(dev_rows, use_lambdamart=use_lm, n_folds=args.folds)
    print(f"\n  {'model':<14} {'mean nDCG@10':>14} {'std':>8}   per-fold")
    print("  " + "-" * 74)
    for name, vals in sorted(cv.items(), key=lambda kv: -np.mean(kv[1])):
        folds = " ".join(f"{v:.3f}" for v in vals)
        print(f"  {name:<14} {np.mean(vals):>14.4f} {np.std(vals):>8.4f}   {folds}")

    learned = [n for n in cv if n not in {"bm25", "dense+prior"}]
    best = max(learned, key=lambda n: float(np.mean(cv[n])))
    print(f"\n  -> selected '{best}' by CV mean")

    print("\nheld-out papers (scored once)")
    print("=" * 78)
    models, extra = fit_models(dev_rows, use_lambdamart=use_lm)
    qrels = qrels_for(hold_rows)
    runs = {**baseline_runs(hold_rows), **{n: rank_with(f, hold_rows) for n, f in models.items()}}
    per_query = {n: ndcg_of(run, qrels) for n, run in runs.items()}
    for name in sorted(per_query, key=lambda n: -float(np.mean(per_query[n]))):
        print(f"  {summarize(f'{name} nDCG@10', per_query[name])}")

    print("\npaired bootstrap vs bm25 (holdout)")
    print("-" * 78)
    results = {"cv": {k: v for k, v in cv.items()}, "holdout": {}, "paired": {}, "selected": best}
    for name in per_query:
        if name == "bm25":
            continue
        pr = paired_bootstrap(per_query["bm25"], per_query[name])
        print(f"  {name:<14} delta={pr.delta:+.4f} CI[{pr.ci_low:+.4f},{pr.ci_high:+.4f}] "
              f"p={pr.p_value:.3f}  {'significant' if pr.significant else 'not significant'}")
        results["paired"][name] = vars(pr)
    for name, vals in per_query.items():
        results["holdout"][name] = {"ndcg@10": float(np.mean(vals)), "n": len(vals)}

    linear = extra["linear_obj"]
    coefs = dict(zip(FEATURE_NAMES, linear.coef_[0]))
    print("\nlearned section weights vs the hand-set prior")
    print("-" * 78)
    print(f"  {'section':<14} {'hand-set':>10} {'learned':>10}")
    results["section_weights"] = {}
    for s in SECTIONS:
        hand = Retriever.section_bonus.get(s, 0.0)
        got = float(coefs[f"section_{s}"])
        results["section_weights"][s] = {"hand_set": hand, "learned": got}
        flag = "  <- sign flip" if hand > 0 and got < 0 else ""
        print(f"  {s:<14} {hand:>10.3f} {got:>10.3f}{flag}")

    print("\ntop features by |coefficient| (linear)")
    print("-" * 78)
    for name, w in sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:10]:
        print(f"  {name:<24} {w:+.3f}")
    results["coefficients"] = {k: float(v) for k, v in coefs.items()}

    MODEL_PATH.write_text(json.dumps({
        "model": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "coef": [float(c) for c in linear.coef_[0]],
        "intercept": float(linear.intercept_[0]),
        "scaler_mean": extra["scaler"].mean_.tolist(),
        "scaler_scale": extra["scaler"].scale_.tolist(),
        "n_train_queries": len(dev_rows),
        "n_train_papers": len(papers) - n_hold,
    }, indent=2))
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nmodel  -> {MODEL_PATH}")
    print(f"results-> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
