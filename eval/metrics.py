"""Retrieval metrics with uncertainty.

The point of this module is that a metric without an interval is not a result.
The existing fixture reports recall@1 = 58.3% on n=12, which carries a 95%
interval of roughly [32%, 81%] — wide enough that a genuine 10-point retrieval
improvement is indistinguishable from noise. Every number here therefore comes
with a confidence interval, and A-vs-B comparisons use a *paired* bootstrap over
the shared query set rather than comparing two independent intervals (which is
both less powerful and a common way to miss a real effect).

Graded relevance is supported throughout (0 = irrelevant, 1 = related,
2 = directly answers) because nDCG over graded labels distinguishes "put the
best chunk first" from "put a merely related chunk first", and binary recall
cannot.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


# ---------------------------------------------------------------------------
# Point metrics. Each takes the ranked chunk ids and a {chunk_id: grade} map.
# ---------------------------------------------------------------------------

def recall_at_k(ranked: Sequence[str], relevant: dict[str, int], k: int) -> float:
    """Fraction of relevant items retrieved in the top k.

    Binarized at grade >= 1. Note this is recall over the *judged* set, so it is
    only meaningful when the candidate pool was built by pooling — otherwise
    unjudged-but-relevant chunks silently count against you.
    """
    positives = {cid for cid, grade in relevant.items() if grade >= 1}
    if not positives:
        return 0.0
    hits = len(positives.intersection(ranked[:k]))
    return hits / len(positives)


def hit_at_k(ranked: Sequence[str], relevant: dict[str, int], k: int) -> float:
    """1.0 if any relevant item is in the top k. This is what the existing
    `test_recall.py` actually measures, despite the name — kept separate so the
    two are never confused when comparing old numbers to new."""
    positives = {cid for cid, grade in relevant.items() if grade >= 1}
    return 1.0 if positives.intersection(ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: dict[str, int]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if relevant.get(cid, 0) >= 1:
            return 1.0 / i
    return 0.0


def dcg(grades: Iterable[float]) -> float:
    # Standard exponential-gain DCG: (2^rel - 1) / log2(rank + 1).
    return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(grades, start=1))


def ndcg_at_k(ranked: Sequence[str], relevant: dict[str, int], k: int) -> float:
    gains = [float(relevant.get(cid, 0)) for cid in ranked[:k]]
    ideal = sorted((float(g) for g in relevant.values()), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(gains) / ideal_dcg


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves at small n and extreme proportions — exactly the regime this project
    is in.
    """
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI over per-query scores.

    Used for mean metrics like nDCG where Wilson does not apply. Seeded so a CI
    is reproducible across runs — an unseeded interval that wobbles between runs
    cannot be used as a CI gate.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(int((1 - alpha / 2) * resamples), resamples - 1)]
    return (lo, hi)


@dataclass
class PairedResult:
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int

    @property
    def significant(self) -> bool:
        """Significant when the interval excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedResult:
    """Paired bootstrap on per-query differences.

    Pairing matters: the same queries are run through both systems, so the
    query-difficulty variance that dominates an unpaired comparison cancels.
    Comparing two independent CIs and asking whether they overlap is a weaker
    and more error-prone test than this.

    p is two-sided, computed as the fraction of resampled mean differences whose
    sign opposes the observed difference (times two).
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired bootstrap needs equal-length score vectors")
    if not baseline:
        return PairedResult(0.0, 0.0, 0.0, 1.0, 0)

    diffs = [c - b for b, c in zip(baseline, candidate)]
    observed = sum(diffs) / len(diffs)

    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(int((1 - alpha / 2) * resamples), resamples - 1)]

    if observed > 0:
        opposing = sum(1 for m in means if m <= 0)
    elif observed < 0:
        opposing = sum(1 for m in means if m >= 0)
    else:
        opposing = resamples // 2
    p = min(1.0, 2 * opposing / resamples)
    return PairedResult(observed, lo, hi, p, n)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class MetricSummary:
    name: str
    value: float
    ci_low: float
    ci_high: float
    n: int

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.value:.4f} "
            f"[95% CI {self.ci_low:.4f}-{self.ci_high:.4f}] n={self.n}"
        )


def summarize(name: str, per_query: Sequence[float], *, binary: bool = False, seed: int = 0) -> MetricSummary:
    """Mean of per-query scores with an interval.

    Binary metrics (hit@k) use Wilson; continuous ones (nDCG, MRR) bootstrap.
    """
    if not per_query:
        return MetricSummary(name, 0.0, 0.0, 0.0, 0)
    mean = sum(per_query) / len(per_query)
    if binary:
        lo, hi = wilson_interval(int(round(sum(per_query))), len(per_query))
    else:
        lo, hi = bootstrap_ci(per_query, seed=seed)
    return MetricSummary(name, mean, lo, hi, len(per_query))


def evaluate_run(
    run: dict[str, Sequence[str]],
    qrels: dict[str, dict[str, int]],
    *,
    ks: Sequence[int] = (1, 5, 10),
    seed: int = 0,
) -> dict[str, MetricSummary]:
    """Score a full run. `run` maps query_id -> ranked chunk ids."""
    query_ids = [qid for qid in qrels if qid in run]
    out: dict[str, MetricSummary] = {}
    for k in ks:
        out[f"ndcg@{k}"] = summarize(
            f"ndcg@{k}", [ndcg_at_k(run[q], qrels[q], k) for q in query_ids], seed=seed
        )
        out[f"hit@{k}"] = summarize(
            f"hit@{k}", [hit_at_k(run[q], qrels[q], k) for q in query_ids], binary=True
        )
        out[f"recall@{k}"] = summarize(
            f"recall@{k}", [recall_at_k(run[q], qrels[q], k) for q in query_ids], seed=seed
        )
    out["mrr"] = summarize("mrr", [reciprocal_rank(run[q], qrels[q]) for q in query_ids], seed=seed)
    return out


def per_query_scores(
    run: dict[str, Sequence[str]],
    qrels: dict[str, dict[str, int]],
    metric: Callable[[Sequence[str], dict[str, int]], float],
) -> tuple[list[str], list[float]]:
    """Aligned (query_ids, scores) for paired comparisons."""
    qids = sorted(qid for qid in qrels if qid in run)
    return qids, [metric(run[q], qrels[q]) for q in qids]
