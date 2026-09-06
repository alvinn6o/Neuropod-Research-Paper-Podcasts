"""Task A evaluation: which papers should become episodes?

A different problem from Task B (which chunk of a paper reaches the prompt):
different unit, different labels, different baseline. This is the one that
answers "how do we know these are the best papers?" — a question the project
previously had no way to answer at all.

Metrics are @5, not @10: the pipeline generates 3-5 episodes per run
(`MAX_EPISODES_PER_RUN = 5`), so quality below rank 5 is invisible to the user
and measuring it would flatter the system.

Baselines first, and the production heuristic is measured as *itself* — by
calling `rank_candidates`, not by a reimplementation — so the number is about
the code that ships.

    python -m eval.recommend
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import ndcg_at_k, paired_bootstrap, summarize
from eval.topics import TOPICS, TfIdf, load_papers
from pipeline.discover.ranker import rank_candidates
from pipeline.models import PaperCandidate

QRELS = ROOT / "eval" / "corpus" / "topic_qrels.json"
POOLS = ROOT / "eval" / "corpus" / "topic_pools.json"
RESULTS = ROOT / "eval" / "recommend_results.json"

K = 5


def load_qrels() -> dict[str, dict[str, int]]:
    if not QRELS.exists():
        raise SystemExit("no labels — run `python -m eval.annotate ingest ...` first")
    return json.loads(QRELS.read_text())


def as_candidates(papers) -> list[PaperCandidate]:
    return [
        PaperCandidate(
            arxiv_id=p.arxiv_id, title=p.title, abstract=p.abstract,
            authors=[], categories=p.categories, published_at=p.published_at,
            pdf_url="", sections={},
        )
        for p in papers
    ]


# ---------------------------------------------------------------------------
# Rankers
# ---------------------------------------------------------------------------

def rank_production(papers, topic_desc: str) -> list[str]:
    """The shipping heuristic, called directly rather than reimplemented.

    `0.45*recency + 0.35*trending + 0.20*affinity`. Measured on this corpus,
    two of the three terms are numerically inert and the 0.20-weighted affinity
    term does all the work — see `decompose()`.
    """
    cands = as_candidates(papers)
    ranked = rank_candidates(cands, topic_desc.split(", "), top_k=len(cands))
    return [c.arxiv_id for c in ranked]


def rank_recency(papers, topic_desc: str) -> list[str]:
    """What the production heuristic collapses to once trending is constant and
    affinity is weak. Included to show how much of it is really just a date sort."""
    return [p.arxiv_id for p in sorted(papers, key=lambda p: p.published_at, reverse=True)]


def rank_tfidf(papers, topic_desc: str, tfidf: TfIdf, index: dict[str, int]) -> list[str]:
    """TF-IDF cosine between the topic description and title+abstract.

    The non-learned bar. Classical lexical matching is often startlingly hard to
    beat, and a learned model that cannot clear it is not worth serving.
    """
    scores = tfidf.query(topic_desc)
    return [p.arxiv_id for p in sorted(papers, key=lambda p: -scores[index[p.arxiv_id]])]


def rank_random(papers, topic_desc: str, seed: int = 0) -> list[str]:
    import random
    ids = [p.arxiv_id for p in papers]
    random.Random(seed).shuffle(ids)
    return ids


def decompose() -> None:
    """Score the heuristic's three terms separately.

    The aggregate hides which part is working. Here it turns out that the
    0.20-weighted affinity term reproduces the full heuristic exactly, while
    0.80 of the weight sits on terms that do not vary between papers.
    """
    import math
    from datetime import datetime, timezone

    from pipeline.discover.ranker import _topic_terms

    qrels = load_qrels()
    papers = load_papers()
    now = datetime.now(timezone.utc)

    print("\nDecomposing 0.45*recency + 0.35*trending + 0.20*affinity")
    print("=" * 78)
    aff_scores, rec_scores = [], []
    for topic, desc in TOPICS.items():
        terms = _topic_terms(desc.split(", "))

        def affinity(p):
            text = " ".join([p.title, p.abstract, *p.categories]).lower()
            return sum(1 for t in terms if t in text) / max(len(terms), 1)

        by_aff = sorted(papers, key=lambda p: -affinity(p))
        aff_scores.append(ndcg_at_k([p.arxiv_id for p in by_aff], qrels[topic], K))
        by_rec = sorted(papers, key=lambda p: p.published_at, reverse=True)
        rec_scores.append(ndcg_at_k([p.arxiv_id for p in by_rec], qrels[topic], K))

    ages = [(now - datetime.fromisoformat(p.published_at.replace("Z", "+00:00"))).days
            for p in papers]
    half_life = 3.5  # window_days / 2, the production default
    max_recency = math.exp(-min(ages) / half_life)

    print(f"  affinity term alone   nDCG@{K} = {sum(aff_scores)/len(aff_scores):.3f}")
    print(f"  recency  term alone   nDCG@{K} = {sum(rec_scores)/len(rec_scores):.3f}")
    print(f"  trending term alone   nDCG@{K} = constant for every paper (arXiv citations are 0)")
    print()
    print(f"  Corpus age: {min(ages)}-{max(ages)} days. With half-life {half_life}d,")
    print(f"  the largest recency score in the corpus is {max_recency:.2e} — it underflows")
    print(f"  to zero for every paper, so 0.45 of the weight multiplies a constant.")
    print()
    print("  CAVEAT, and it matters: production ranks a 7-day discovery window, where")
    print("  recency DOES vary. This corpus spans years, so it is out of distribution")
    print("  for that term. What this measures is ranking within a diverse pool, not")
    print("  the production setting. A Task A corpus drawn from one 7-day window would")
    print("  be needed to evaluate recency honestly.")


def main() -> None:
    qrels = load_qrels()
    pools = json.loads(POOLS.read_text())["pools"]
    all_papers = load_papers()
    by_id = {p.arxiv_id: p for p in all_papers}
    tfidf = TfIdf([p.text for p in all_papers])
    index = {p.arxiv_id: i for i, p in enumerate(all_papers)}

    rankers = {
        "random": lambda ps, d: rank_random(ps, d),
        "production": rank_production,
        "recency-only": rank_recency,
        "tfidf": lambda ps, d: rank_tfidf(ps, d, tfidf, index),
    }

    def evaluate(candidate_set: str):
        """candidate_set: 'pool' (judged only) or 'corpus' (all 168, unjudged=0)."""
        per_topic: dict[str, dict[str, float]] = defaultdict(dict)
        per_ranker: dict[str, list[float]] = defaultdict(list)
        prec: dict[str, list[float]] = defaultdict(list)
        for topic, description in TOPICS.items():
            papers = ([by_id[pid] for pid in pools[topic]] if candidate_set == "pool"
                      else all_papers)
            rel = qrels[topic]
            for name, fn in rankers.items():
                ranked = fn(papers, description)
                per_topic[topic][name] = ndcg_at_k(ranked, rel, K)
                per_ranker[name].append(per_topic[topic][name])
                prec[name].append(sum(1 for p in ranked[:K] if rel.get(p, 0) >= 1) / K)
        return per_topic, per_ranker, prec

    def report(title: str, caveat: str, per_topic, per_ranker, prec) -> dict:
        print(f"\n{title}")
        print("=" * 78)
        print(f"  {caveat}")
        header = f"  {'ranker':<14}" + "".join(f"{t:>10}" for t in TOPICS) + f"{'mean':>9}{'P@5':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        block: dict = {}
        for name in rankers:
            row = "".join(f"{per_topic[t][name]:>10.3f}" for t in TOPICS)
            mean = sum(per_ranker[name]) / len(per_ranker[name])
            p5 = sum(prec[name]) / len(prec[name])
            print(f"  {name:<14}{row}{mean:>9.3f}{p5:>8.1%}")
            block[name] = {"ndcg@5_mean": mean, "p@5": p5,
                           "per_topic": {t: per_topic[t][name] for t in TOPICS}}
        return block

    print(f"Task A — paper recommendation, {len(TOPICS)} topics")
    print(f"nDCG@{K} (the pipeline generates 3-5 episodes, so @5 is what a user sees)")

    corpus_pt, corpus_pr, corpus_p5 = evaluate("corpus")
    pool_pt, per_ranker, prec = evaluate("pool")
    per_topic = pool_pt
    out: dict = {"k": K, "corpus": {}, "pool": {}}
    out["corpus"] = report(
        "A. Ranking the FULL corpus (168 papers, unjudged treated as grade 0)",
        "The end-to-end task. Every system sees the same 168 candidates.",
        corpus_pt, corpus_pr, corpus_p5)
    out["pool"] = report(
        "B. Ranking the judged pool only (60 papers per topic)",
        "Reranking quality on a candidate set TF-IDF helped build — reads high for it.",
        pool_pt, per_ranker, prec)

    print(f"\npaired bootstrap vs production, full corpus (n={len(TOPICS)} topics — very small)")
    print("  " + "-" * 74)
    for name in rankers:
        if name == "production":
            continue
        r = paired_bootstrap(corpus_pr["production"], corpus_pr[name])
        print(f"  {name:<14} delta={r.delta:+.4f} CI[{r.ci_low:+.4f},{r.ci_high:+.4f}] "
              f"p={r.p_value:.3f}  {'significant' if r.significant else 'not significant'}")
        out["corpus"][name]["vs_production"] = vars(r)

    print("\n  NOTE: n=5 topics. These intervals are wide by construction and the")
    print("  bootstrap is resampling five numbers. Treat the ordering as the")
    print("  finding and the magnitudes as indicative — the fix is more topics,")
    print("  not more resamples.")

    decompose()

    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwritten -> {RESULTS}")


if __name__ == "__main__":
    main()
