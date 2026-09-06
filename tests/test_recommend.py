"""Task A: paper recommendation labels, pooling, and baselines.

Task A ranks *papers* (which become episodes); Task B ranks *chunks* (which
reach the prompt). Different unit, different labels, different baseline — the
tests are kept apart for the same reason.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POOLS = ROOT / "eval" / "corpus" / "topic_pools.json"
QRELS_LLM = ROOT / "eval" / "corpus" / "topic_qrels_llm.json"
QRELS = ROOT / "eval" / "corpus" / "topic_qrels.json"

needs_labels = pytest.mark.skipif(
    not (POOLS.exists() and QRELS.exists()),
    reason="topic labels not built (python -m eval.topics; eval.annotate ingest)",
)


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------

def test_kappa_is_1_for_identical_and_0_for_chance():
    from eval.annotate import cohens_kappa

    assert cohens_kappa([0, 1, 2, 0, 1], [0, 1, 2, 0, 1]) == 1.0
    assert cohens_kappa([0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]) == pytest.approx(0.0)


def test_kappa_punishes_agreement_that_is_just_a_common_label():
    """The reason kappa exists. Two annotators who both mostly say 0 agree often
    while sharing little actual judgment; raw agreement cannot tell the
    difference and kappa can."""
    from eval.annotate import cohens_kappa

    a = [0] * 8 + [1, 2]
    b = [0] * 8 + [2, 1]
    raw = sum(x == y for x, y in zip(a, b)) / len(a)
    assert raw == 0.8
    assert cohens_kappa(a, b) < 0.5


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

@needs_labels
def test_pool_has_a_random_arm_that_finds_positives_tfidf_misses():
    """Pooling's whole purpose, asserted.

    Labelling only what the current system already surfaces makes its misses
    invisible — anything it never retrieves is never judged, so it cannot be
    scored against. The random arm is what makes the benchmark able to say the
    baseline was wrong.
    """
    from eval.topics import TOP_K, TOPICS, TfIdf, load_papers

    pools = json.loads(POOLS.read_text())["pools"]
    labels = json.loads(QRELS.read_text())
    papers = load_papers()
    tfidf = TfIdf([p.text for p in papers])

    found_outside_tfidf = 0
    for topic, description in TOPICS.items():
        scores = tfidf.query(description)
        ranked = sorted(range(len(papers)), key=lambda i: -scores[i])
        top_ids = {papers[i].arxiv_id for i in ranked[:TOP_K]}
        for pid in set(pools[topic]) - top_ids:
            if labels[topic].get(pid, 0) >= 1:
                found_outside_tfidf += 1

    assert found_outside_tfidf >= 10, (
        f"only {found_outside_tfidf} positives came from the random arm — the pool "
        "may be too TF-IDF-shaped to detect its misses"
    )


@needs_labels
def test_every_pooled_paper_is_labelled_exactly_once():
    pools = json.loads(POOLS.read_text())["pools"]
    labels = json.loads(QRELS.read_text())
    for topic, pool in pools.items():
        assert set(labels[topic]) == set(pool), f"{topic}: label set != pool"


@needs_labels
def test_labels_are_graded_and_not_degenerate():
    """A label set that is 95% one class measures almost nothing."""
    from collections import Counter

    labels = json.loads(QRELS.read_text())
    for topic, judgments in labels.items():
        dist = Counter(judgments.values())
        assert set(dist) <= {0, 1, 2}
        positive_rate = (dist[1] + dist[2]) / len(judgments)
        assert 0.10 < positive_rate < 0.90, (
            f"{topic}: positive rate {positive_rate:.0%} is too skewed to be informative"
        )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

@needs_labels
def test_baselines_beat_random_and_the_heuristic_is_not_just_recency():
    """Two claims at once.

    The first is a sanity floor. The second corrects an assumption: the
    production heuristic weights recency at 0.45, so it *looks* like a date
    sort — but ranking by date alone scores far worse, because the 0.20
    affinity term is carrying it.
    """
    from eval.metrics import ndcg_at_k
    from eval.recommend import rank_production, rank_random, rank_recency
    from eval.topics import TOPICS, load_papers

    qrels = json.loads(QRELS.read_text())
    papers = load_papers()

    def mean_ndcg(fn):
        return sum(
            ndcg_at_k(fn(papers, desc), qrels[topic], 5) for topic, desc in TOPICS.items()
        ) / len(TOPICS)

    production = mean_ndcg(rank_production)
    recency = mean_ndcg(rank_recency)
    random_ = mean_ndcg(lambda ps, d: rank_random(ps, d))

    assert production > random_ * 2, "the heuristic must clearly beat random"
    assert production > recency + 0.2, (
        f"production ({production:.3f}) is barely above recency-only ({recency:.3f}); "
        "the affinity term should be doing the work"
    )


@needs_labels
def test_recency_term_is_inert_on_this_corpus():
    """A stated limitation, pinned by a test so it cannot be forgotten.

    Production ranks a 7-day discovery window; this corpus spans years. With a
    3.5-day half-life, exp(-age/half_life) underflows to ~0 for every paper, so
    0.45 of the heuristic's weight multiplies a constant here. Any conclusion
    about the recency term from this corpus is out of distribution.
    """
    import math
    from datetime import datetime, timezone

    from eval.topics import load_papers

    now = datetime.now(timezone.utc)
    ages = [
        (now - datetime.fromisoformat(p.published_at.replace("Z", "+00:00"))).days
        for p in load_papers()
    ]
    largest = math.exp(-min(ages) / 3.5)
    assert largest < 1e-20, (
        "corpus is now fresh enough for the recency term to vary — the caveat in "
        "eval/recommend.py should be revisited"
    )


# ---------------------------------------------------------------------------
# Annotator reliability
# ---------------------------------------------------------------------------

PASS2 = ROOT / "eval" / "corpus" / "topic_qrels_llm_pass2.json"

needs_pass2 = pytest.mark.skipif(not PASS2.exists(), reason="no second annotation pass")


@needs_pass2
def test_annotator_is_self_consistent():
    """Test-retest reliability of the LLM annotator.

    This is NOT validation. It is the same annotator judging the same papers a
    second time under a different procedure (full abstracts, one at a time,
    without consulting pass 1). It bounds reliability from above: an annotator
    who cannot reproduce their own judgments certainly cannot be trusted
    against someone else's. Human-vs-LLM kappa still requires a human.
    """
    from eval.annotate import cohens_kappa

    p1 = json.loads((ROOT / "eval" / "corpus" / "topic_qrels_llm.json").read_text())
    p2 = json.loads(PASS2.read_text())
    a = [p1[t][pid] for t in p2 for pid in p2[t]]
    b = [p2[t][pid] for t in p2 for pid in p2[t]]

    kappa = cohens_kappa(a, b)
    print(f"\n  self-consistency kappa = {kappa:.3f} over n={len(a)}")
    assert kappa > 0.7, (
        f"the annotator disagrees with itself (kappa={kappa:.3f}); the rubric is "
        "too ambiguous to produce a usable label set"
    )


@needs_pass2
def test_disagreements_concentrate_in_the_adjacent_grade():
    """Where the rubric is actually weak.

    Every disagreement between the two passes involved grade 1 ('adjacent'),
    while the 0-vs-2 boundary was stable. That localises the ambiguity: the
    definition of 'the topic is a real but secondary part of the paper' is the
    part a second annotator would most likely read differently. Binarizing
    (relevant vs not) raises kappa, which is the same fact from another angle.
    """
    from eval.annotate import cohens_kappa

    p1 = json.loads((ROOT / "eval" / "corpus" / "topic_qrels_llm.json").read_text())
    p2 = json.loads(PASS2.read_text())

    disagreements = [
        (p1[t][pid], p2[t][pid]) for t in p2 for pid in p2[t] if p1[t][pid] != p2[t][pid]
    ]
    assert disagreements, "no disagreements at all is suspicious, not reassuring"
    assert all(1 in pair for pair in disagreements), (
        "a disagreement now spans the 0-vs-2 boundary, not just the adjacent band — "
        "the rubric has a new failure mode"
    )

    a = [p1[t][pid] for t in p2 for pid in p2[t]]
    b = [p2[t][pid] for t in p2 for pid in p2[t]]
    binary = cohens_kappa([1 if x >= 1 else 0 for x in a], [1 if y >= 1 else 0 for y in b])
    assert binary >= cohens_kappa(a, b)


def test_saving_one_topic_does_not_clobber_another(tmp_path, monkeypatch):
    """Regression test for a real data-loss bug.

    A review session loaded the whole label file at start and wrote its whole
    in-memory copy at the end. Anything that changed on disk meanwhile was
    silently reverted: a session in another topic lost its work, and a topic
    dropped mid-session came back. Both actually happened during labelling.
    """
    from eval import annotate

    path = tmp_path / "labels.json"
    monkeypatch.setitem(annotate.LABELS, "human", path)

    annotate.save_labels("human", {"llm": {"a": 2}, "vision": {"b": 1}})
    # A second session, started before the next change, saves only its topic.
    annotate.save_topic("human", "graph", {"c": 0})

    saved = json.loads(path.read_text())
    assert saved["llm"] == {"a": 2}, "another topic's judgments were clobbered"
    assert saved["vision"] == {"b": 1}
    assert saved["graph"] == {"c": 0}


def test_dropping_a_topic_records_a_reason(tmp_path, monkeypatch):
    """A dropped topic becomes single-annotator, and that limitation has to
    travel with the data rather than live in a commit message."""
    from eval import annotate

    labels = tmp_path / "labels.json"
    excl = tmp_path / "exclusions.json"
    monkeypatch.setitem(annotate.LABELS, "human", labels)
    monkeypatch.setattr(annotate, "EXCLUSIONS", excl)
    monkeypatch.setattr(annotate, "MERGED", tmp_path / "merged.json")

    annotate.save_labels("human", {"theory": {"a": 1}, "llm": {"b": 2}})
    args = type("A", (), {"topic": "theory", "role": "human", "reason": "no expertise"})()
    annotate.cmd_drop(args)

    assert "theory" not in json.loads(labels.read_text())
    recorded = json.loads(excl.read_text())["human"]["theory"]
    assert recorded["dropped"] == 1 and recorded["reason"] == "no expertise"


def test_category_check_needs_no_annotator_judgment():
    """The third signal exists because both annotators are fallible: the LLM is
    systematically generous, and the human lacks expertise in several
    subfields. arXiv categories are author-assigned, so they depend on neither.
    """
    from eval.category_check import TOPIC_CATEGORIES, category_label
    from eval.topics import TOPICS

    assert set(TOPIC_CATEGORIES) == set(TOPICS)

    paper = type("P", (), {"categories": ["cs.CV", "cs.LG"], "primary_category": "cs.CV"})()
    assert category_label(paper, "vision") == 1
    assert category_label(paper, "llm") == 0
