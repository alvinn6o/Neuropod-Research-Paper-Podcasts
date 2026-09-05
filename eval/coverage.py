"""Section coverage: does retrieval give the scriptwriter what the prompt asks for?

The ICT benchmark measures *pinpointing* — find the one chunk a sentence came
from. That is the right task for comparing rankers, but it cannot evaluate the
production query at all, because ICT supplies its own queries and never
exercises the one the pipeline actually sends.

The product needs something different: *coverage*. The prompt instructs the
model to explain the problem, the method, the quantitative results and the
limitations. If the top 14 chunks contain nothing from the results section, the
model cannot report results without inventing them — the most damaging failure
this system has.

So the metric is: of the sections the prompt needs, how many are represented in
the chunks that reach it? Deterministic, needs no labels, no API calls, and it
measures the query rather than the ranker.

Limits, stated: section labels come from the PDF header regexes, so a paper
whose headers were not recognised contributes little. And section *presence* is
a coarse proxy for whether the content is any good. It is a necessary condition,
not a sufficient one.

    python -m eval.coverage
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import queries as q_mod
from eval.metrics import summarize
from pipeline.generate.embedder import HashEmbedder
from pipeline.generate.facets import build_facet_queries
from pipeline.generate.retriever import PROMPT_CHUNK_LIMIT, Retriever

# What the scriptwriter prompt actually asks the model to cover. `body` counts
# for every facet: it is the fallback label for a paper whose section headers
# were not recognised, so its chunks could serve any role.
NEEDED_SECTIONS = {
    "problem": {"abstract", "introduction", "background", "body"},
    "method": {"methods", "body"},
    "results": {"results", "experiments", "body"},
    "limitations": {"limitations", "discussion", "conclusion", "body"},
}

RESULTS_PATH = ROOT / "eval" / "coverage_results.json"


def _load_papers() -> dict[str, list[dict]]:
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for chunk in q_mod.load_chunks():
        by_paper[chunk["paper_id"]].append(chunk)
    return by_paper


def coverage_for(retrieved: list[dict], available: set[str]) -> tuple[float, dict[str, bool]]:
    """Fraction of *achievable* facets covered.

    Normalized by what the paper actually contains: a paper with no limitations
    section cannot have its limitations covered, and scoring that as a miss
    would measure PDF parsing rather than retrieval.
    """
    got = {c["section"] for c in retrieved}
    hits, achievable = {}, 0
    for facet, sections in NEEDED_SECTIONS.items():
        if not (sections & available):
            continue
        achievable += 1
        hits[facet] = bool(sections & got)
    if not achievable:
        return 0.0, hits
    return sum(hits.values()) / achievable, hits


def run() -> dict:
    by_paper = _load_papers()
    embedder = HashEmbedder()
    retriever = Retriever(embedder=embedder)

    strategies = {
        # What shipped before this change.
        "title+abstract": lambda t, a: [f"{t} {a}"],
        "facets": lambda t, a: build_facet_queries(t, a),
        "facets+topical": lambda t, a: build_facet_queries(t, a) + [f"{t} {a}"],
    }

    manifest = {p["arxiv_id"]: p for p in
                json.loads((ROOT / "eval" / "corpus" / "manifest.json").read_text())["papers"]}

    per_strategy: dict[str, list[float]] = defaultdict(list)
    facet_hits: dict[str, Counter] = defaultdict(Counter)
    facet_total: dict[str, Counter] = defaultdict(Counter)

    for paper_id, chunks in sorted(by_paper.items()):
        meta = manifest.get(paper_id, {})
        title, abstract = meta.get("title", ""), meta.get("abstract", "")
        for chunk in chunks:
            chunk["embedding"] = embedder.embed_text(chunk["content"])
            chunk["embedding_model"] = embedder.model_id
        available = {c["section"] for c in chunks}

        for name, build in strategies.items():
            retrieved = retriever.retrieve(chunks, "", limit=PROMPT_CHUNK_LIMIT) \
                if False else [
                    r["chunk"] for r in retriever.retrieve_multi(
                        chunks, build(title, abstract), limit=PROMPT_CHUNK_LIMIT)
                ]
            score, hits = coverage_for(retrieved, available)
            per_strategy[name].append(score)
            for facet, ok in hits.items():
                facet_total[name][facet] += 1
                facet_hits[name][facet] += int(ok)

    print(f"Section coverage@{PROMPT_CHUNK_LIMIT} over {len(by_paper)} papers")
    print("=" * 74)
    out: dict = {"n_papers": len(by_paper), "strategies": {}}
    for name in strategies:
        s = summarize(f"{name}", per_strategy[name])
        print(f"  {s}")
        out["strategies"][name] = {
            "coverage": s.value, "ci_low": s.ci_low, "ci_high": s.ci_high,
            "per_facet": {
                f: facet_hits[name][f] / facet_total[name][f]
                for f in facet_total[name]
            },
        }

    print(f"\nper-facet hit rate (was the facet's section represented in the top {PROMPT_CHUNK_LIMIT}?)")
    print("-" * 74)
    facets = list(NEEDED_SECTIONS)
    print(f"  {'strategy':<18}" + "".join(f"{f:>14}" for f in facets))
    for name in strategies:
        row = "".join(
            f"{facet_hits[name][f] / facet_total[name][f]:>13.1%} " if facet_total[name][f] else f"{'n/a':>14}"
            for f in facets
        )
        print(f"  {name:<18}{row}")

    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwritten -> {RESULTS_PATH}")
    return out


if __name__ == "__main__":
    run()
