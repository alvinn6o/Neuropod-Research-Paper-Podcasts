"""Topic profiles and pooled candidate selection for Task A.

Task A is paper-level recommendation: given a user's topic, rank ~50 arXiv
candidates so the ones worth an episode come first. It is a different problem
from Task B (which chunk of one paper reaches the prompt) — different unit,
different labels, different baseline.

**Candidate pooling.** For each topic the labelled pool is the union of the
top-K by TF-IDF and a random sample of the rest. Labelling only what the current
system already likes is the classic way to build a benchmark that flatters the
system that built it: anything it never surfaces is never judged, so its misses
are invisible. This is TREC-style pooling, scaled down.

The random arm also keeps the label distribution honest. A pool of pure
TF-IDF top hits would be almost all positives, and a classifier trained on it
would never see a hard negative.
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "eval" / "corpus" / "manifest.json"
POOL_PATH = ROOT / "eval" / "corpus" / "topic_pools.json"

# Five profiles a real user of this product might set. Chosen after surveying
# the corpus so each has enough plausible positives to be worth labelling —
# a topic with three relevant papers measures nothing.
TOPICS: dict[str, str] = {
    "llm": "large language models, in-context learning, prompting, instruction tuning, transformers for text",
    "vision": "computer vision, image generation, diffusion models, object detection, multimodal vision-language",
    "rl": "reinforcement learning, bandits, exploration, policy optimization, sequential decision making",
    "graph": "graph neural networks, graph representation learning, node classification, network structure",
    "theory": "learning theory, generalization bounds, statistical estimation, convergence analysis, optimization theory",
}

TOP_K = 40          # highest TF-IDF scorers
RANDOM_K = 20       # random draw from the remainder — the anti-bias arm
SEED = 11

_TOKEN = re.compile(r"[a-z][a-z0-9-]{2,}")
_STOP = frozenset("""
and are for from that this with the not but has have been their które which
where when what how our its into over under more than very can may also such
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    categories: list[str]
    primary_category: str
    published_at: str

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}"


def load_papers() -> list[Paper]:
    raw = json.loads(MANIFEST.read_text())["papers"]
    return [
        Paper(
            arxiv_id=p["arxiv_id"], title=p["title"], abstract=p["abstract"],
            categories=p.get("categories", []),
            primary_category=p.get("primary_category", ""),
            published_at=p.get("published_at", ""),
        )
        for p in raw
    ]


class TfIdf:
    """Plain TF-IDF cosine. The non-learned baseline Task A has to beat.

    Deliberately hand-written rather than pulled from sklearn: it is four lines
    of arithmetic and the point of a baseline is that its behaviour is obvious.
    """

    def __init__(self, docs: list[str]) -> None:
        self.tokenized = [tokenize(d) for d in docs]
        self.df = Counter()
        for toks in self.tokenized:
            for t in set(toks):
                self.df[t] += 1
        self.n = len(docs)
        self.vectors = [self._vec(toks) for toks in self.tokenized]

    def idf(self, term: str) -> float:
        return math.log((self.n + 1) / (self.df.get(term, 0) + 1)) + 1.0

    def _vec(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        if not tf:
            return {}
        vec = {t: (c / len(tokens)) * self.idf(t) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def query(self, text: str) -> list[float]:
        qv = self._vec(tokenize(text))
        out = []
        for dv in self.vectors:
            shared = set(qv) & set(dv)
            out.append(sum(qv[t] * dv[t] for t in shared))
        return out


def build_pools() -> dict[str, list[str]]:
    papers = load_papers()
    tfidf = TfIdf([p.text for p in papers])
    rng = random.Random(SEED)
    pools: dict[str, list[str]] = {}

    for topic, description in TOPICS.items():
        scores = tfidf.query(description)
        ranked = sorted(range(len(papers)), key=lambda i: -scores[i])
        top = ranked[:TOP_K]
        rest = ranked[TOP_K:]
        sampled = rng.sample(rest, min(RANDOM_K, len(rest)))
        # Sorted by arxiv_id so the annotation order carries no signal about
        # which arm a paper came from — otherwise an annotator learns that the
        # first 40 are the likely positives and anchors on it.
        pool = sorted({papers[i].arxiv_id for i in top + sampled})
        pools[topic] = pool
        print(f"  {topic:<8} pool={len(pool)}  (top-{len(top)} tfidf + {len(sampled)} random)")

    POOL_PATH.write_text(json.dumps({"seed": SEED, "top_k": TOP_K,
                                     "random_k": RANDOM_K, "pools": pools}, indent=2))
    print(f"\nwritten -> {POOL_PATH}")
    return pools


if __name__ == "__main__":
    print(f"building pooled candidates for {len(TOPICS)} topics over {len(load_papers())} papers")
    build_pools()
