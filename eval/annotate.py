"""Annotation tooling for Task A relevance labels.

Three annotator roles, kept in separate files so they can be compared rather
than merged:

  llm    — an LLM judge reading title + abstract
  human  — the project owner, correcting a sample
  merged — human where present, LLM elsewhere; this is what models train on

**Why keep them separate.** The headline claim of any LLM-labelled dataset is
that the judge agrees with a human. That claim is only checkable if both sets
survive. Cohen's kappa between them is reported before any model number is
quoted; below about 0.6 the label *definition* is the problem, not the model.

    python -m eval.annotate worksheet        # emit papers to judge
    python -m eval.annotate review --topic llm --n 20   # human spot-check
    python -m eval.annotate agreement        # kappa between annotators
    python -m eval.annotate stats
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.topics import TOPICS, load_papers

CORPUS = ROOT / "eval" / "corpus"
POOL_PATH = CORPUS / "topic_pools.json"
LABELS = {
    # First pass: bulk judgment from title + category + abstract snippet.
    "llm": CORPUS / "topic_qrels_llm.json",
    # Independent second pass by the same annotator, full abstracts, one paper
    # at a time, without consulting pass 1. Kappa against `llm` measures
    # SELF-CONSISTENCY (test-retest reliability), NOT agreement with a human.
    # It is an upper bound: an annotator who cannot reproduce their own
    # judgments certainly cannot be trusted against someone else's.
    "llm_pass2": CORPUS / "topic_qrels_llm_pass2.json",
    # The project owner. This is the only role that can produce the
    # human-vs-LLM kappa the label set actually needs.
    "human": CORPUS / "topic_qrels_human.json",
}
MERGED = CORPUS / "topic_qrels.json"
# Which topics a role deliberately did NOT judge, and why. Kept next to the
# labels so the gap travels with the data rather than living in a commit
# message nobody reads.
EXCLUSIONS = CORPUS / "topic_qrels_exclusions.json"

# The rubric. Written down because "relevant" is not self-evident, and an
# annotation guideline that lives only in someone's head cannot be audited,
# handed over, or agreed with.
GUIDELINE = """
2 — ON TOPIC. The paper's main contribution is squarely within the topic. A
    subscriber to this topic would want an episode about it.

1 — ADJACENT. The topic is a real but secondary part of the paper: applied to
    this domain, evaluated on it, or a neighbouring subfield. Worth surfacing
    only when little else is available.

0 — OFF TOPIC. A subscriber would consider this a mistake, even if it shares
    generic machine-learning vocabulary.

Judge the *paper*, not the wording. Shared jargon is not relevance: nearly every
paper here says "model", "training" and "data".
"""


def load_pools() -> dict[str, list[str]]:
    if not POOL_PATH.exists():
        raise SystemExit("run `python -m eval.topics` first")
    return json.loads(POOL_PATH.read_text())["pools"]


def load_labels(role: str) -> dict[str, dict[str, int]]:
    path = LABELS[role]
    return json.loads(path.read_text()) if path.exists() else {}


def save_labels(role: str, data: dict) -> None:
    LABELS[role].write_text(json.dumps(data, indent=2, sort_keys=True))


def cmd_worksheet(args) -> None:
    """Emit the pooled papers as a judgeable worksheet."""
    pools = load_pools()
    papers = {p.arxiv_id: p for p in load_papers()}
    done = load_labels("llm")
    topics = [args.topic] if args.topic else list(TOPICS)

    for topic in topics:
        pending = [pid for pid in pools[topic] if pid not in done.get(topic, {})]
        if not pending:
            print(f"# {topic}: complete ({len(pools[topic])} labelled)")
            continue
        print(f"\n{'=' * 78}\n# TOPIC {topic}: {TOPICS[topic]}\n# {len(pending)} unlabelled\n{'=' * 78}")
        for pid in pending[: args.limit]:
            p = papers[pid]
            abstract = " ".join(p.abstract.split())[: args.chars]
            print(f"\n[{pid}] ({p.primary_category})\nTITLE: {p.title}\nABSTRACT: {abstract}")


def cmd_ingest(args) -> None:
    """Merge a JSON blob of {topic: {arxiv_id: grade}} into a role's labels."""
    incoming = json.loads(Path(args.path).read_text())
    data = load_labels(args.role)
    added = 0
    for topic, judgments in incoming.items():
        bucket = data.setdefault(topic, {})
        for pid, grade in judgments.items():
            if grade not in (0, 1, 2):
                raise SystemExit(f"grade must be 0/1/2, got {grade!r} for {pid}")
            bucket[pid] = grade
            added += 1
    save_labels(args.role, data)
    print(f"{added} judgments -> {LABELS[args.role].name}")
    _rebuild_merged()


def cmd_review(args) -> None:
    """Interactive human spot-check of the LLM labels."""
    llm = load_labels("llm")
    human = load_labels("human")
    papers = {p.arxiv_id: p for p in load_papers()}
    topic = args.topic

    already = human.get(topic, {})
    if args.redo:
        # Clearing rather than re-prompting: a corrected judgment made while
        # looking at the old one is not independent, and independence is the
        # only reason these labels are worth collecting.
        print(f"--redo: discarding {len(already)} existing judgments for '{topic}'")
        human[topic] = {}
        already = human[topic]

    candidates = [pid for pid in llm.get(topic, {}) if pid not in already]
    if not candidates:
        print(f"{topic}: all {len(already)} already judged. "
              f"Re-do them with:  make annotate TOPIC={topic} REDO=1")
        return
    if already:
        print(f"resuming '{topic}': {len(already)} already judged, "
              f"{len(candidates)} remaining\n")

    print(GUIDELINE)
    print("=" * 78)
    print(f"TOPIC: {topic} — {TOPICS[topic]}")
    print("=" * 78)
    if args.show_llm:
        print("\n!! --show-llm is ON. The LLM's grade is displayed BEFORE you answer,")
        print("!! which anchors you and inflates agreement. Do not use this for kappa.\n")
    else:
        print("\nBlind: the LLM's grade is hidden until after you answer, so your")
        print("judgment is independent. That is what makes kappa mean anything.\n")
    print("Enter 0, 1 or 2.  's' skips this paper.  'q' saves and quits.\n")

    bucket = human.setdefault(topic, {})
    total = min(args.n, len(candidates))
    agreed = 0
    scored = 0
    for i, pid in enumerate(candidates[:total], start=1):
        p = papers[pid]
        suggested = llm[topic][pid]
        abstract = " ".join(p.abstract.split())
        truncated = abstract[: args.chars]
        print("\n" + "=" * 78)
        print(f"[{i}/{total}]  {pid}   arXiv: {p.primary_category}")
        print("=" * 78)
        for line in textwrap.wrap(p.title, width=74):
            print(f"  {line}")
        print()
        # Wrapped rather than dumped as one long line: a terminal soft-wrap
        # breaks mid-word and makes a 900-character abstract genuinely hard to
        # read, which is a bad way to collect careful judgments.
        for line in textwrap.wrap(truncated, width=74):
            print(f"    {line}")
        if len(abstract) > args.chars:
            print(f"    [... {len(abstract) - args.chars} more chars; "
                  f"raise with --chars]")
        print(f"\n  TOPIC: {TOPICS[topic]}")
        print("  2 = squarely on topic   1 = adjacent/secondary   0 = off topic")
        if args.show_llm:
            print(f"\n  (LLM said: {suggested})")
        while True:
            try:
                raw = input("\n  your grade [0/1/2, s, q] > ").strip().lower()
            except EOFError:
                raw = "q"
            if raw in {"0", "1", "2", "s", "q"}:
                break
            print("  -> enter 0, 1, 2, s or q")
        if raw == "q":
            break
        if raw == "s":
            continue
        bucket[pid] = int(raw)
        scored += 1
        if int(raw) == suggested:
            agreed += 1
            print(f"  = LLM also said {suggested}")
        else:
            print(f"  x LLM said {suggested}, you said {raw}  <- disagreement, this is the signal")

    save_labels("human", human)
    print("\n" + "=" * 78)
    print(f"saved {scored} judgments this session "
          f"({sum(len(v) for v in human.values())} total across all topics)")
    if scored:
        print(f"raw agreement this session: {agreed}/{scored} = {agreed/scored:.0%}")
        print("\nRaw agreement is NOT the number that matters — run `agreement` for kappa,")
        print("which corrects for how often two annotators would agree by chance.")
    _rebuild_merged()


def _rebuild_merged() -> None:
    """Human labels win where they exist; LLM pass 1 fills the rest.

    Pass 2 deliberately does NOT feed the merged set — merging an annotator's
    two attempts would destroy the disagreement signal that makes the
    self-consistency number meaningful.
    """
    llm, human = load_labels("llm"), load_labels("human")
    merged: dict[str, dict[str, int]] = {}
    for topic in set(llm) | set(human):
        merged[topic] = {**llm.get(topic, {}), **human.get(topic, {})}
    MERGED.write_text(json.dumps(merged, indent=2, sort_keys=True))


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Agreement corrected for chance.

    Raw agreement is misleading when one grade dominates: if 80% of papers are
    off-topic, two annotators who both guess 0 every time agree 80% of the time
    while sharing no judgment at all. Kappa subtracts that expectation.
    """
    if not a:
        return 0.0
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def cmd_agreement(args) -> None:
    llm, human = load_labels("llm"), load_labels(args.against)
    pairs_a, pairs_b = [], []
    per_topic = {}
    for topic in sorted(set(llm) & set(human)):
        shared = sorted(set(llm[topic]) & set(human[topic]))
        if not shared:
            continue
        a = [llm[topic][p] for p in shared]
        b = [human[topic][p] for p in shared]
        per_topic[topic] = (cohens_kappa(a, b), len(shared),
                            sum(x == y for x, y in zip(a, b)) / len(shared))
        pairs_a += a
        pairs_b += b

    if not pairs_a:
        print(f"No overlapping judgments between 'llm' and '{args.against}' yet.")
        return

    kind = ("SELF-CONSISTENCY (same annotator, second pass) — an upper bound, "
            "not validation" if args.against == "llm_pass2"
            else "HUMAN vs LLM agreement")
    print(f"\n{kind}\n")

    print(f"{'topic':<10} {'n':>5} {'raw agree':>11} {'kappa':>8}")
    print("-" * 38)
    for topic, (k, n, raw) in per_topic.items():
        print(f"{topic:<10} {n:>5} {raw:>10.1%} {k:>8.3f}")
    unjudged = sorted(set(TOPICS) - set(per_topic))
    overall = cohens_kappa(pairs_a, pairs_b)
    raw = sum(x == y for x, y in zip(pairs_a, pairs_b)) / len(pairs_a)
    print("-" * 38)
    print(f"{'OVERALL':<10} {len(pairs_a):>5} {raw:>10.1%} {overall:>8.3f}")
    print(f"\n  interpretation: {_kappa_reading(overall)}")
    if unjudged:
        print(f"\n  NOT covered by this kappa: {', '.join(unjudged)} — single-annotator.")
        if EXCLUSIONS.exists():
            notes = json.loads(EXCLUSIONS.read_text()).get(args.against if args.against != "llm_pass2" else "human", {})
            for topic, info in sorted(notes.items()):
                print(f"    {topic}: dropped ({info['reason']})")
    if overall < 0.6:
        print("  ** Below 0.6 — fix the label DEFINITION before trusting any model number. **")


def _kappa_reading(k: float) -> str:
    if k < 0.20: return "slight — the two annotators are barely related"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate — usable but the rubric is ambiguous somewhere"
    if k < 0.80: return "substantial — the standard bar for a usable label set"
    return "almost perfect"


def cmd_drop(args) -> None:
    """Discard one annotator role's judgments for a topic, with a reason.

    A legitimate annotation decision, not a cleanup: an annotator without the
    domain knowledge to judge a topic produces labels that look like data and
    are not. Dropping them is correct — silently keeping them is what would
    corrupt the label set.

    The reason is recorded because the consequence outlives the deletion: that
    topic's labels stay single-annotator, so kappa does not cover it and any
    Task A number involving it carries a weaker claim.
    """
    data = load_labels(args.role)
    removed = len(data.get(args.topic, {}))
    if not removed:
        print(f"{args.role} has no judgments for '{args.topic}'")
        return
    data.pop(args.topic, None)
    save_labels(args.role, data)

    notes = json.loads(EXCLUSIONS.read_text()) if EXCLUSIONS.exists() else {}
    notes.setdefault(args.role, {})[args.topic] = {
        "dropped": removed,
        "reason": args.reason,
    }
    EXCLUSIONS.write_text(json.dumps(notes, indent=2, sort_keys=True))

    print(f"dropped {removed} '{args.topic}' judgments from '{args.role}'")
    print(f"  reason recorded in {EXCLUSIONS.name}: {args.reason}")
    print(f"\n  '{args.topic}' is now single-annotator (LLM only). Kappa will not")
    print(f"  cover it, and that limitation should be stated wherever Task A")
    print(f"  numbers for '{args.topic}' are reported.")
    _rebuild_merged()


def cmd_stats(args) -> None:
    for role in ("llm", "human"):
        data = load_labels(role)
        if not data:
            print(f"{role}: none")
            continue
        total = sum(len(v) for v in data.values())
        dist = Counter(g for v in data.values() for g in v.values())
        print(f"{role}: {total} judgments  grades={dict(sorted(dist.items()))}")
        for topic in sorted(data):
            d = Counter(data[topic].values())
            pos = d[2] + d[1]
            print(f"   {topic:<8} n={len(data[topic]):<4} "
                  f"2={d[2]:<3} 1={d[1]:<3} 0={d[0]:<3} positive-rate={pos/len(data[topic]):.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worksheet"); w.add_argument("--topic"); w.add_argument("--limit", type=int, default=100); w.add_argument("--chars", type=int, default=700); w.set_defaults(fn=cmd_worksheet)
    i = sub.add_parser("ingest"); i.add_argument("path"); i.add_argument("--role", default="llm", choices=list(LABELS)); i.set_defaults(fn=cmd_ingest)
    r = sub.add_parser("review")
    r.add_argument("--topic", required=True, choices=list(TOPICS))
    r.add_argument("--n", type=int, default=20)
    r.add_argument("--chars", type=int, default=1100)
    r.add_argument("--redo", action="store_true",
                   help="discard existing human judgments for this topic and start over")
    r.add_argument("--show-llm", action="store_true",
                   help="show the LLM grade BEFORE you answer (anchors you; not for kappa)")
    r.set_defaults(fn=cmd_review)
    a = sub.add_parser("agreement"); a.add_argument("--against", default="human", choices=["human", "llm_pass2"]); a.set_defaults(fn=cmd_agreement)
    d = sub.add_parser("drop")
    d.add_argument("--topic", required=True, choices=list(TOPICS))
    d.add_argument("--role", default="human", choices=list(LABELS))
    d.add_argument("--reason", required=True,
                   help="why these judgments are being discarded (recorded, not optional)")
    d.set_defaults(fn=cmd_drop)
    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats)
    args = ap.parse_args()
    args.fn(args)
