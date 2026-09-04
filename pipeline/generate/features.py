"""Ranking features for the learned reranker.

Shared by training and serving on purpose. Training/serving skew — computing a
feature one way offline and another way online — is the classic way a model that
looked good offline underperforms in production, and keeping one implementation
is the cheapest guard against it.

The feature set is chosen to *contain* the current hand-tuned heuristic rather
than replace it wholesale: `Retriever.section_bonus` is a one-hot over sections
with hand-set weights, so section one-hots are here and the model is free to
learn those weights itself. The evaluation showed the hand-set values actively
hurt (dense -> dense+prior = -0.027 nDCG@10, p<0.001), so "what does the model
learn instead" is the question this is built to answer.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .bm25 import BM25Index, tokenize

# Fixed order: the model's coefficient vector is only interpretable against a
# stable feature order, and a reordering would silently corrupt a saved model.
SECTIONS = [
    "abstract", "introduction", "background", "methods",
    "experiments", "results", "discussion", "limitations", "conclusion", "body",
]

FEATURE_NAMES = [
    "bm25",
    "bm25_z",
    "bm25_rank_recip",
    "dense",
    "dense_z",
    "dense_rank_recip",
    "score_agreement",
    "term_overlap",
    "term_coverage",
    "idf_overlap",
    "max_sent_overlap",
    "chunk_tokens",
    "rel_position",
    "is_section_start",
    "numeric_density",
    "number_match",
    "query_tokens",
    *[f"section_{s}" for s in SECTIONS],
]

_NUMBER_RE = re.compile(r"\d[\d.,]*")


@dataclass
class Candidate:
    chunk_id: str
    section: str
    content: str
    chunk_index: int
    features: list[float]


def _numbers(text: str) -> set[str]:
    return {n.rstrip(".,") for n in _NUMBER_RE.findall(text)}


def build_features(
    chunks: list[dict],
    query: str,
    *,
    dense_scores: dict[str, float] | None = None,
) -> list[Candidate]:
    """Featurize one query against one paper's chunks.

    Scores are rank-normalized within the query rather than used raw: BM25 is
    unbounded and corpus-dependent, so a raw value is not comparable across
    queries and a model trained on raw values would learn the corpus, not the
    ranking. Reciprocal rank is scale-free and is what RRF uses for the same
    reason.
    """
    idx = BM25Index.build([(c["id"], c["content"]) for c in chunks])
    bm25 = idx.score(query)
    dense = dense_scores or {}

    bm25_order = sorted(bm25, key=lambda cid: bm25[cid], reverse=True)
    bm25_rank = {cid: i + 1 for i, cid in enumerate(bm25_order)}
    dense_order = sorted(dense, key=lambda cid: dense.get(cid, 0.0), reverse=True)
    dense_rank = {cid: i + 1 for i, cid in enumerate(dense_order)}

    q_tokens = set(tokenize(query))
    q_numbers = _numbers(query)
    max_bm25 = max(bm25.values()) or 1.0
    n_chunks = len(chunks) or 1

    # Within-query standardization as well as max-normalization. They encode
    # different things: max-norm says "how close to the best candidate", z-score
    # says "how far above the pack". A query where everything scores similarly
    # is a different situation from one with a clear winner, and only the
    # z-score can express that.
    bm_vals = list(bm25.values())
    bm_mu = sum(bm_vals) / len(bm_vals) if bm_vals else 0.0
    bm_sd = (sum((v - bm_mu) ** 2 for v in bm_vals) / len(bm_vals)) ** 0.5 if bm_vals else 0.0
    dn_vals = list(dense.values()) or [0.0]
    dn_mu = sum(dn_vals) / len(dn_vals)
    dn_sd = (sum((v - dn_mu) ** 2 for v in dn_vals) / len(dn_vals)) ** 0.5

    out: list[Candidate] = []
    for chunk in chunks:
        cid = chunk["id"]
        c_tokens = set(tokenize(chunk["content"]))
        shared = q_tokens & c_tokens

        overlap = len(shared) / len(q_tokens) if q_tokens else 0.0
        # IDF-weighted overlap separates "shares a rare technical term" from
        # "shares three common words", which plain overlap cannot.
        idf_overlap = sum(idx.idf(t) for t in shared)

        c_numbers = _numbers(chunk["content"])
        tokens_in_chunk = len(c_tokens) or 1

        # Best single sentence, not the whole chunk. A chunk that answers the
        # query in one sentence and then drifts is diluted by whole-chunk
        # overlap; this recovers it.
        max_sent = 0.0
        for sent in re.split(r"(?<=[.!?])\s+", chunk["content"]):
            s_tokens = set(tokenize(sent))
            if s_tokens and q_tokens:
                max_sent = max(max_sent, len(q_tokens & s_tokens) / len(q_tokens))

        bm_raw = bm25.get(cid, 0.0)
        dn_raw = dense.get(cid, 0.0)
        bm_r = bm25_rank.get(cid, n_chunks)
        dn_r = dense_rank.get(cid, n_chunks) if dense else n_chunks

        feats = {
            "bm25": bm_raw / max_bm25,
            "bm25_z": (bm_raw - bm_mu) / bm_sd if bm_sd else 0.0,
            "bm25_rank_recip": 1.0 / bm_r,
            "dense": dn_raw,
            "dense_z": (dn_raw - dn_mu) / dn_sd if dn_sd else 0.0,
            "dense_rank_recip": 1.0 / dn_r if dense else 0.0,
            # Both retrievers agreeing is the signal RRF exploits; given here
            # explicitly so the model can weight it rather than rediscover it.
            "score_agreement": 1.0 / (1.0 + abs(bm_r - dn_r)),
            "term_overlap": overlap,
            "term_coverage": len(shared) / len(c_tokens) if c_tokens else 0.0,
            "idf_overlap": math.log1p(idf_overlap),
            "max_sent_overlap": max_sent,
            "is_section_start": 1.0 if chunk.get("chunk_index", 0) == 0 else 0.0,
            "query_tokens": math.log1p(len(q_tokens)),
            # Log-scaled: chunk lengths are roughly log-normal and a raw count
            # lets one long chunk dominate a linear model's gradient.
            "chunk_tokens": math.log1p(len(chunk["content"].split())),
            "rel_position": chunk.get("chunk_index", 0) / 20.0,
            "numeric_density": len(c_numbers) / tokens_in_chunk,
            "number_match": 1.0 if (q_numbers & c_numbers) else 0.0,
        }
        for s in SECTIONS:
            feats[f"section_{s}"] = 1.0 if chunk.get("section") == s else 0.0

        out.append(Candidate(
            chunk_id=cid,
            section=chunk.get("section", ""),
            content=chunk["content"],
            chunk_index=chunk.get("chunk_index", 0),
            features=[feats[name] for name in FEATURE_NAMES],
        ))
    return out
