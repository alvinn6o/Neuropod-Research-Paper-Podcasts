"""Nested cross-validation for the reranker's hyperparameters.

Nothing in this project has ever been tuned; every LightGBM setting was picked
by hand. This closes that gap, and does it in the way that does not lie.

**Why nested.** The obvious approach — run CV, try many configurations, report
the best CV score — reports a number that is optimistically biased. You chose
the configuration *because* it scored well on those folds, so its score on those
folds includes the luck that made it win. With enough configurations you can
report a good number for a model that is no better than the default.

Nested CV separates the two jobs. An **inner** CV over the training folds picks
the hyperparameters; the **outer** fold, which the selection never saw, reports
the score. The outer estimate answers "how well does *this whole procedure*,
including its tuning, generalise?" — which is the question that matters.

Both numbers are printed here on purpose. The gap between them is the
selection bias, made visible rather than assumed away.

Grouping is by paper at both levels: queries from one paper share a chunk pool,
so a query-level split would leak.

    python -m eval.tune --configs 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import ndcg_at_k, paired_bootstrap, summarize
from eval.train_reranker import (
    load_dataset,
    ndcg_of,
    qrels_for,
    rank_with,
    split_by_paper,
    to_matrix,
)

RESULTS = ROOT / "eval" / "tuning_results.json"

# The hand-picked configuration everything so far was measured with. It is in
# the search space, so the sweep can confirm or overturn it rather than being
# free to look good by comparison with something arbitrary.
BASELINE_CONFIG = {
    "n_estimators": 300, "learning_rate": 0.06, "num_leaves": 15,
    "min_child_samples": 20, "reg_lambda": 1.0,
}

GRID = {
    "n_estimators": [150, 300, 600],
    "learning_rate": [0.03, 0.06, 0.12],
    "num_leaves": [7, 15, 31],
    "min_child_samples": [10, 20, 40],
    "reg_lambda": [0.0, 1.0, 10.0],
}


def sample_configs(n: int, seed: int = 0) -> list[dict]:
    """Random search, with the hand-picked config always included.

    Random search rather than exhaustive grid: with 5 hyperparameters the full
    grid is 243 fits per inner fold, and random search reaches a comparable
    optimum in far fewer trials when only a couple of dimensions actually
    matter (Bergstra & Bengio, 2012).
    """
    rng = np.random.default_rng(seed)
    seen = {json.dumps(BASELINE_CONFIG, sort_keys=True)}
    configs = [dict(BASELINE_CONFIG)]
    while len(configs) < n:
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in GRID.items()}
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


def fit_and_score(config: dict, train_rows, test_rows) -> float:
    import lightgbm as lgb

    X, y, groups, _ = to_matrix(train_rows)
    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        random_state=0, verbose=-1, **config,
    )
    ranker.fit(X, y, group=groups)
    run = rank_with(lambda Z: ranker.predict(Z), test_rows)
    return float(np.mean(ndcg_of(run, qrels_for(test_rows))))


def folds_by_paper(rows, k: int) -> list[set[str]]:
    papers = sorted({r["paper_id"] for r in rows})
    return [set(papers[i::k]) for i in range(k)]


def split_on(rows, held: set[str]):
    return ([r for r in rows if r["paper_id"] not in held],
            [r for r in rows if r["paper_id"] in held])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, default=12)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    args = ap.parse_args()

    print("building features...")
    rows = load_dataset(verbose=False)
    dev, hold, _ = split_by_paper(rows)
    configs = sample_configs(args.configs)
    print(f"  {len(dev)} dev queries over {len({r['paper_id'] for r in dev})} papers")
    print(f"  {len(configs)} configurations, {args.outer} outer x {args.inner} inner folds")
    print(f"  ~{len(configs) * args.outer * args.inner + args.outer} fits\n")

    t0 = time.perf_counter()
    outer_folds = folds_by_paper(dev, args.outer)

    # ---- Nested: selection happens inside, scoring happens outside ----
    nested_scores: list[float] = []
    chosen: list[dict] = []
    for oi, held in enumerate(outer_folds, start=1):
        outer_train, outer_test = split_on(dev, held)
        inner_folds = folds_by_paper(outer_train, args.inner)

        inner_means = []
        for cfg in configs:
            scores = []
            for inner_held in inner_folds:
                itr, ite = split_on(outer_train, inner_held)
                if itr and ite:
                    scores.append(fit_and_score(cfg, itr, ite))
            inner_means.append(float(np.mean(scores)) if scores else 0.0)

        best = configs[int(np.argmax(inner_means))]
        chosen.append(best)
        score = fit_and_score(best, outer_train, outer_test)
        nested_scores.append(score)
        print(f"  outer fold {oi}/{args.outer}: inner-best nDCG@10={max(inner_means):.4f}"
              f"  ->  outer nDCG@10={score:.4f}")

    # ---- Non-nested: select and report on the SAME folds ----
    print("\n  non-nested pass (selection and scoring on the same folds)...")
    nonnested_per_config = []
    for cfg in configs:
        scores = [fit_and_score(cfg, *split_on(dev, held)) for held in outer_folds]
        nonnested_per_config.append(float(np.mean(scores)))
    nonnested_best_idx = int(np.argmax(nonnested_per_config))
    nonnested = nonnested_per_config[nonnested_best_idx]
    baseline_cv = nonnested_per_config[0]   # index 0 is the hand-picked config

    nested = float(np.mean(nested_scores))
    elapsed = time.perf_counter() - t0

    print(f"\nResults  ({elapsed/60:.1f} min)")
    print("=" * 78)
    print(f"  hand-picked config, plain CV        nDCG@10 = {baseline_cv:.4f}")
    print(f"  best config, NON-nested CV          nDCG@10 = {nonnested:.4f}   <- optimistic")
    print(f"  tuning procedure, NESTED CV         nDCG@10 = {nested:.4f}   <- honest")
    print(f"\n  selection bias (non-nested - nested) = {nonnested - nested:+.4f}")
    print(f"  real gain over hand-picked           = {nested - baseline_cv:+.4f}")
    print("\n  The first gap is the price of choosing a winner on the same folds you")
    print("  report. The second is what tuning actually bought.")

    print("\nWhich configuration each outer fold chose")
    print("-" * 78)
    keys = list(GRID)
    print("  fold " + "".join(f"{k:>19}" for k in keys))
    for i, cfg in enumerate(chosen, start=1):
        print(f"  {i:<5}" + "".join(f"{cfg[k]:>19}" for k in keys))
    stability = {k: Counter(c[k] for c in chosen).most_common(1)[0] for k in keys}
    print("\n  most-selected value per hyperparameter (and how many of "
          f"{len(chosen)} folds agreed):")
    for k, (val, count) in stability.items():
        flag = "" if count > len(chosen) // 2 else "   <- unstable, folds disagree"
        print(f"    {k:<20} {val!r:>8}  {count}/{len(chosen)}{flag}")
    print("\n  Folds disagreeing is itself a result: a hyperparameter the data cannot")
    print("  pin down is one the model is insensitive to, and tuning it is noise.")

    # ---- Holdout, once, with the majority configuration ----
    final_cfg = {k: stability[k][0] for k in keys}
    print(f"\nHeld-out papers, scored once with the majority config")
    print("=" * 78)
    print(f"  {final_cfg}")
    import lightgbm as lgb
    qrels = qrels_for(hold)
    per_query = {}
    for name, cfg in (("hand-picked", BASELINE_CONFIG), ("tuned", final_cfg)):
        X, y, groups, _ = to_matrix(dev)
        r = lgb.LGBMRanker(objective="lambdarank", metric="ndcg",
                           random_state=0, verbose=-1, **cfg)
        r.fit(X, y, group=groups)
        per_query[name] = ndcg_of(rank_with(lambda Z: r.predict(Z), hold), qrels)
        print(f"  {summarize(name, per_query[name])}")
    pr = paired_bootstrap(per_query["hand-picked"], per_query["tuned"])
    print(f"\n  hand-picked -> tuned: delta={pr.delta:+.4f} "
          f"CI[{pr.ci_low:+.4f},{pr.ci_high:+.4f}] p={pr.p_value:.3f} "
          f"{'significant' if pr.significant else 'NOT significant'}")

    RESULTS.write_text(json.dumps({
        "nested_ndcg": nested, "nonnested_ndcg": nonnested,
        "hand_picked_cv": baseline_cv, "selection_bias": nonnested - nested,
        "outer_scores": nested_scores, "chosen_per_fold": chosen,
        "final_config": final_cfg,
        "holdout": {k: float(np.mean(v)) for k, v in per_query.items()},
        "holdout_paired": vars(pr), "minutes": elapsed / 60,
    }, indent=2))
    print(f"\nwritten -> {RESULTS}")


if __name__ == "__main__":
    main()
