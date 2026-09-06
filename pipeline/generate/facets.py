"""Facet queries for retrieval.

The scriptwriter's prompt asks the model to cover five specific things: the
problem, the method, the quantitative results, the limitations, and the
implications. Retrieval should therefore search for each of those, not once for
"the paper in general".

What it replaced (`orchestrator.py`): the query was `title + abstract`. Two
problems with that. An abstract summarises the whole paper, so as a query it is
diffuse — it matches every chunk a little and no chunk specifically. And the
abstract is itself in the index, so the query partly retrieves its own source.

Note what the facet cues deliberately do NOT contain: the paper's title or
topic. Retrieval is scoped to a single paper's chunks, so every candidate is
already on-topic and topical terms carry no discriminating signal — they would
only dilute the facet cue. The query's whole job here is to separate
*sections of one paper*, not to find the paper.

These are keyword-style rather than natural-language questions because BM25 does
the heavy lifting: it matches terms, and section prose reliably contains the
vocabulary of its own function ("we propose", "compared to the baseline",
"a limitation of our approach").
"""
from __future__ import annotations

# Ordered; the keys are recorded in retrieval traces so a chunk can be
# attributed to the facet that surfaced it.
FACET_CUES: dict[str, str] = {
    "problem": (
        "problem motivation challenge why this matters shortcoming of prior work "
        "existing approaches fail difficulty"
    ),
    "method": (
        "method approach architecture mechanism algorithm model design we propose "
        "our framework implementation training procedure"
    ),
    "results": (
        "results experiments evaluation accuracy improvement benchmark performance "
        "compared baseline outperforms ablation measured achieves"
    ),
    "limitations": (
        "limitations failure cases weaknesses does not generalize future work "
        "remains open caveat assumption threats to validity"
    ),
}


def build_facet_queries(title: str = "", abstract: str = "") -> list[str]:
    """The queries used to retrieve context for one paper.

    `title` and `abstract` are accepted but deliberately unused for the facet
    cues themselves — see the module docstring. They are kept in the signature
    so a caller can pass the paper through and an experiment can add a topical
    query back without changing every call site.
    """
    return list(FACET_CUES.values())


def facet_names() -> list[str]:
    return list(FACET_CUES)
