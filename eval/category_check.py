"""A third annotator that requires no domain expertise: arXiv's own categories.

The human-vs-LLM kappa came out at 0.374, and the annotator reports the cause
is their own limited familiarity with several subfields rather than a defect in
the LLM labels. That is plausible and common — relevance judgment across all of
cs.LG/cs.CL/cs.CV/stat.ML is genuinely expert work — but it does not rescue the
LLM labels. Kappa measures agreement, not correctness. If one annotator is
noisy, low agreement says the *check* failed, not that the other side passed.

So this adds a signal that depends on neither of us: **the arXiv category the
paper's own authors assigned it.** It is independent of the TF-IDF pooling and
of both annotators, needs no judgment to apply, and is reproducible by anyone.

It is NOT ground truth, and the ways it is wrong are worth naming:

  * categories are coarse. cs.CV covers "image generation" and "3D pose
    estimation" alike, so category agreement will call a pose paper on-topic
    for a diffusion-flavoured profile.
  * authors self-assign and often cross-list generously.
  * several profiles here ("learning theory") span categories rather than
    matching one.

What it can do is bound the disagreement. If the LLM labels agree strongly with
author-assigned categories, the labels are at least tracking something external
and stable. If they do not, that is a real problem independent of who annotated.

    python -m eval.category_check
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.annotate import cohens_kappa
from eval.topics import TOPICS, load_papers

QRELS_LLM = ROOT / "eval" / "corpus" / "topic_qrels_llm.json"
QRELS_HUMAN = ROOT / "eval" / "corpus" / "topic_qrels_human.json"
POOLS = ROOT / "eval" / "corpus" / "topic_pools.json"
RESULTS = ROOT / "eval" / "category_agreement.json"

# arXiv categories that make a paper plausibly on-topic. Written from arXiv's
# published taxonomy, not from the corpus, so it cannot be tuned to flatter a
# result. 'theory' deliberately spans several, which is exactly why category
# agreement will be weakest there.
TOPIC_CATEGORIES: dict[str, set[str]] = {
    "llm": {"cs.CL"},
    "vision": {"cs.CV", "eess.IV"},
    "rl": {"cs.RO"},                       # arXiv has no RL category; cs.LG is too broad
    "graph": {"cs.SI"},                    # likewise: GNNs live under cs.LG
    "theory": {"stat.ML", "math.ST", "stat.ME", "math.OC", "cs.IT", "math.NA"},
}


def category_label(paper, topic: str) -> int:
    """1 if any of the paper's categories matches the topic's set."""
    allowed = TOPIC_CATEGORIES[topic]
    cats = set(paper.categories) | ({paper.primary_category} if paper.primary_category else set())
    return 1 if cats & allowed else 0


def main() -> None:
    papers = {p.arxiv_id: p for p in load_papers()}
    llm = json.loads(QRELS_LLM.read_text())
    human = json.loads(QRELS_HUMAN.read_text()) if QRELS_HUMAN.exists() else {}

    print("Agreement with author-assigned arXiv categories (binary: relevant vs not)")
    print("=" * 78)
    print(f"  {'topic':<10}{'n':>5}{'LLM vs cat':>13}{'human vs cat':>15}{'coverage':>11}")
    print("  " + "-" * 74)

    out: dict = {"topic_categories": {k: sorted(v) for k, v in TOPIC_CATEGORIES.items()},
                 "per_topic": {}}
    all_llm, all_cat = [], []
    for topic in TOPICS:
        pids = sorted(llm[topic])
        cat = [category_label(papers[pid], topic) for pid in pids]
        lab = [1 if llm[topic][pid] >= 1 else 0 for pid in pids]
        k_llm = cohens_kappa(lab, cat)
        all_llm += lab
        all_cat += cat

        shared = sorted(set(llm[topic]) & set(human.get(topic, {})))
        if len(shared) > 5:
            h = [1 if human[topic][pid] >= 1 else 0 for pid in shared]
            c = [category_label(papers[pid], topic) for pid in shared]
            k_hum = f"{cohens_kappa(h, c):>15.3f}"
        else:
            k_hum = f"{'-':>15}"

        # How many pooled papers the category rule even fires on. A topic whose
        # categories barely appear cannot be checked this way.
        coverage = sum(cat) / len(cat)
        print(f"  {topic:<10}{len(pids):>5}{k_llm:>13.3f}{k_hum}{coverage:>10.0%}")
        out["per_topic"][topic] = {"kappa_llm_vs_category": k_llm, "coverage": coverage}

    overall = cohens_kappa(all_llm, all_cat)
    print("  " + "-" * 74)
    print(f"  {'OVERALL':<10}{len(all_llm):>5}{overall:>13.3f}")
    out["overall_llm_vs_category"] = overall

    print("\n  Read this as a floor, not a score. Categories are coarse and")
    print("  self-assigned; a topic profile narrower than its category (a")
    print("  diffusion-only reader inside all of cs.CV) will disagree by design.")
    print("  'rl' and 'graph' have no arXiv category of their own — both live")
    print("  under cs.LG — so their rows measure almost nothing and say so via")
    print("  low coverage.")

    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwritten -> {RESULTS}")


if __name__ == "__main__":
    main()
