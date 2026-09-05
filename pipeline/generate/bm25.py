"""BM25 over a chunk set.

The existing sparse path (`Retriever._sparse_cosine`) is raw term-frequency
cosine: no IDF, no length normalization, no stopword removal. That makes it
score on the wrong things — "the model" matches "the method" on `the`, and a
long chunk beats a short one purely by having more terms to hit.

BM25 fixes both: IDF down-weights terms that appear in most chunks, and the
b·(len/avglen) term normalizes for chunk length. It is not a fallback for dense
retrieval — it is a complementary signal that wins on exact identifiers, numbers
and rare technical terms, which is precisely the query class this corpus is full
of ("what was the throughput on A100", "what is the selective scan").

Deliberately dependency-free: rank_bm25 is 100 lines and adding it would hide
the part of this that is worth showing.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# Standard Robertson/Sparck-Jones defaults. k1 controls term-frequency
# saturation, b controls length normalization.
K1 = 1.5
B = 0.75

# Small, explicit stoplist. Deliberately short: aggressive stoplists hurt on
# technical queries where "state" and "space" are content words.
STOPWORDS = frozenset("""
a an and are as at be by for from has have in into is it its of on or that the
their there these this to was were which will with we our they he she
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords removed, single chars dropped.

    Keeps `.`, `_` and `-` inside tokens so identifiers and numbers survive
    intact — "gpt-4", "3.2x" and "h_t" are exactly the terms BM25 should be able
    to match exactly.
    """
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if len(t) > 1 and t not in STOPWORDS
    ]


@dataclass
class BM25Index:
    doc_ids: list[str] = field(default_factory=list)
    doc_tokens: list[list[str]] = field(default_factory=list)
    doc_freqs: list[Counter] = field(default_factory=list)
    doc_len: list[int] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    avgdl: float = 0.0
    k1: float = K1
    b: float = B

    @classmethod
    def build(cls, docs: list[tuple[str, str]], *, k1: float = K1, b: float = B) -> "BM25Index":
        """docs: list of (doc_id, text)."""
        index = cls(k1=k1, b=b)
        for doc_id, text in docs:
            tokens = tokenize(text)
            index.doc_ids.append(doc_id)
            index.doc_tokens.append(tokens)
            freqs = Counter(tokens)
            index.doc_freqs.append(freqs)
            index.doc_len.append(len(tokens))
            for term in freqs:
                index.df[term] += 1
        index.avgdl = (sum(index.doc_len) / len(index.doc_len)) if index.doc_len else 0.0
        return index

    def idf(self, term: str) -> float:
        """Robertson-Sparck-Jones IDF with the +0.5 smoothing, floored at 0.

        The unfloored form goes negative for terms in >half the corpus, which
        lets a common term actively subtract from a score. Floored here because
        a chunk set for one paper is small enough that several content words
        legitimately appear in most chunks.
        """
        n = len(self.doc_ids)
        df = self.df.get(term, 0)
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def score(self, query: str) -> dict[str, float]:
        q_terms = tokenize(query)
        scores: dict[str, float] = {}
        for i, doc_id in enumerate(self.doc_ids):
            freqs = self.doc_freqs[i]
            dl = self.doc_len[i] or 1
            total = 0.0
            for term in q_terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                total += self.idf(term) * (tf * (self.k1 + 1)) / denom
            scores[doc_id] = total
        return scores

    def top_k(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scored = sorted(self.score(query).items(), key=lambda kv: kv[1], reverse=True)
        return scored[:k]


def tie_aware_ranks(scores: dict[str, float]) -> dict[str, float]:
    """1-based ranks, with tied scores sharing their average rank.

    Rank-based fusion consumes ranks, not scores, so equal scores would
    otherwise be split by whatever order the sort happened to produce — making
    the final ranking depend on input order rather than on the data. Averaging
    tied ranks ("competition ranking") removes that: two chunks with identical
    text now provably score identically, which is a property a test can assert.
    """
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        average = (i + 1 + j + 1) / 2.0     # 1-based, inclusive
        for cid, _ in ordered[i : j + 1]:
            ranks[cid] = average
        i = j + 1
    return ranks


def fuse_rank_maps(rank_maps: list[dict[str, float]], *, k: int = 60) -> dict[str, float]:
    """RRF over pre-computed (possibly tie-aware) rank maps."""
    fused: dict[str, float] = {}
    for ranks in rank_maps:
        for doc_id, rank in ranks.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def reciprocal_rank_fusion(
    rankings: list[list[str]], *, k: int = 60
) -> list[tuple[str, float]]:
    """Combine ranked lists by RRF: score(d) = sum 1/(k + rank(d)).

    Chosen over a weighted sum of normalized scores because BM25 scores and
    cosine similarities live on different, corpus-dependent scales — normalizing
    them requires a tuning step that RRF avoids entirely. k=60 is the value from
    the original Cormack et al. paper; it damps the influence of the very top
    ranks so one confident-but-wrong system cannot dominate.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
