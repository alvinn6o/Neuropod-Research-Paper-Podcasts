"""Query + label generation for the retrieval benchmark.

The hard part of a retrieval benchmark is not the metrics, it is getting labels.
Two sources are implemented, with different trade-offs, and they are kept
separate rather than pooled into one undifferentiated set:

**ICT (Inverse Cloze Task) — deterministic, free, large.**
Sample a sentence out of a chunk, remove it from that chunk, and use it as the
query; the chunk it came from is the gold answer. This is the standard
self-supervised retrieval pretraining signal (Lee et al. 2019, ORQA; Chang et
al. 2020). It needs no model and no API key, it scales to thousands of queries,
and it is perfectly reproducible — which is what makes it usable as a CI gate.

Its bias must be stated, because it is large: ICT queries are *extracted
sentences*, not questions. They share vocabulary with the gold chunk far more
than a real user question does, which systematically favours lexical matching.
Expect BM25 to look better on ICT than it would on natural questions. So ICT is
used for **regression detection** — "did this change make retrieval worse?" —
and explicitly not as evidence that one retrieval strategy is better in
production.

**LLM-generated natural questions — realistic, costs money, smaller.**
Generate questions conditioned on a chunk, then discard the conditioning so the
question reads naturally (doc2query). Requires an API key. Implemented here and
runs when `OPENAI_API_KEY` is set; cached by content hash so re-runs are free.

The honest framing is that ICT is a proxy and the LLM set is the closer measure,
and the two should be reported separately rather than averaged.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS_DIR = ROOT / "eval" / "corpus"
CHUNKS = CORPUS_DIR / "chunks.jsonl"
ICT_QUERIES = CORPUS_DIR / "queries_ict.jsonl"
LLM_QUERIES = CORPUS_DIR / "queries_llm.jsonl"
LLM_CACHE = CORPUS_DIR / "_llm_query_cache.json"

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# A usable ICT query needs enough content words to be answerable. Too short and
# the "gold" chunk is not recoverable even in principle, which shows up as an
# unfixable recall ceiling rather than as a retrieval problem.
MIN_QUERY_WORDS = 8
MAX_QUERY_WORDS = 40


@dataclass
class EvalQuery:
    query_id: str
    paper_id: str
    query: str
    gold_chunk_id: str
    source: str          # "ict" | "llm"
    section: str


def load_chunks() -> list[dict]:
    if not CHUNKS.exists():
        raise SystemExit("run `python -m eval.corpus_build build` first")
    return [json.loads(line) for line in CHUNKS.read_text().splitlines() if line.strip()]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def build_ict(per_paper: int = 12, seed: int = 17) -> list[EvalQuery]:
    """Generate ICT queries, at most `per_paper` per paper.

    Sampling is seeded and papers are processed in sorted order so the query set
    is byte-identical across runs — a benchmark whose queries drift is not a
    benchmark.
    """
    chunks = load_chunks()
    by_paper: dict[str, list[dict]] = {}
    for c in chunks:
        by_paper.setdefault(c["paper_id"], []).append(c)

    rng = random.Random(seed)
    out: list[EvalQuery] = []

    for paper_id in sorted(by_paper):
        candidates: list[tuple[dict, str]] = []
        for chunk in sorted(by_paper[paper_id], key=lambda c: (c["section"], c["chunk_index"])):
            for sentence in _sentences(chunk["content"]):
                n = len(sentence.split())
                if MIN_QUERY_WORDS <= n <= MAX_QUERY_WORDS:
                    candidates.append((chunk, sentence))
        if not candidates:
            continue
        rng.shuffle(candidates)

        # One query per chunk at most, so a single verbose chunk cannot
        # dominate the paper's queries and skew its per-paper mean.
        used_chunks: set[str] = set()
        for chunk, sentence in candidates:
            if len(used_chunks) >= per_paper:
                break
            if chunk["id"] in used_chunks:
                continue
            used_chunks.add(chunk["id"])
            qid = hashlib.sha1(f"{paper_id}:{chunk['id']}:{sentence}".encode()).hexdigest()[:16]
            out.append(EvalQuery(
                query_id=qid, paper_id=paper_id, query=sentence,
                gold_chunk_id=chunk["id"], source="ict", section=chunk["section"],
            ))
    return out


def _strip_sentence(text: str, sentence: str) -> str:
    return re.sub(r"\s+", " ", text.replace(sentence, " ")).strip()


def _pick_sentence(text: str, seed_key: str) -> str | None:
    """Deterministically choose one sentence to drop from a non-gold chunk.

    The eligible length window is the SAME one used to select query sentences
    (MIN_QUERY_WORDS..MAX_QUERY_WORDS). That matters: with only a lower bound,
    non-gold chunks could lose very long sentences while gold chunks lost a
    query sentence capped at 40 words, leaving gold systematically longer. That
    residual was measurable — ranking by longest-first scored nDCG@10 = 0.168
    against random's 0.130 — and chunk length is the reranker's most-split
    feature, so it was being partly consumed as signal.
    """
    sents = [
        s for s in _sentences(text)
        if MIN_QUERY_WORDS <= len(s.split()) <= MAX_QUERY_WORDS
    ]
    if not sents:
        sents = [s for s in _sentences(text) if len(s.split()) >= MIN_QUERY_WORDS]
    if not sents:
        sents = _sentences(text)
    if not sents:
        return None
    i = int(hashlib.sha1(seed_key.encode()).hexdigest()[:8], 16) % len(sents)
    return sents[i]


def redact_gold(chunks: list[dict], queries: list[EvalQuery]) -> dict[str, dict]:
    """The gold chunk with its query sentence removed, keyed by query_id.

    Without this the task is trivial: the query appears verbatim inside its own
    gold chunk, so any lexical matcher scores ~100%.
    """
    by_id = {c["id"]: c for c in chunks}
    out: dict[str, dict] = {}
    for q in queries:
        gold = by_id.get(q.gold_chunk_id)
        if gold is None:
            continue
        out[q.query_id] = {**gold, "content": _strip_sentence(gold["content"], q.query)}
    return out


def redact_pool(paper_chunks: list[dict], query: EvalQuery) -> list[dict]:
    """Build the candidate pool with one sentence removed from EVERY chunk.

    Redacting only the gold chunk leaks catastrophically. The chunker caps
    chunks at 110 words, so 88.9% of untouched chunks sit exactly at the cap
    while a redacted gold chunk never does. Measured: ranking by chunk length
    alone — ignoring the query entirely — scored nDCG@10 = 0.369 on this corpus,
    beating BM25's 0.224. Any model with access to a length feature learns
    "shorter than the cap" and looks excellent while having learned nothing
    about relevance.

    Removing one sentence from every candidate equalizes the length
    distribution, so length carries no information about which chunk is gold.
    The gold chunk still loses specifically the query sentence, preserving ICT
    semantics. `tests/test_eval_harness.py` asserts the leak stays closed.
    """
    out: list[dict] = []
    for chunk in paper_chunks:
        if chunk["id"] == query.gold_chunk_id:
            out.append({**chunk, "content": _strip_sentence(chunk["content"], query.query)})
            continue
        # Seeded by chunk id alone, not (query, chunk): each chunk then has one
        # canonical redacted form, so its embedding and features can be cached
        # across all queries. The leak is closed by every chunk losing *a*
        # sentence, not by which sentence, so this costs nothing.
        victim = _pick_sentence(chunk["content"], chunk["id"])
        content = _strip_sentence(chunk["content"], victim) if victim else chunk["content"]
        out.append({**chunk, "content": content})
    return out


def write(queries: list[EvalQuery], path: Path) -> None:
    with path.open("w") as fh:
        for q in queries:
            fh.write(json.dumps(asdict(q)) + "\n")
    print(f"{len(queries)} queries -> {path}")


def read(path: Path) -> list[EvalQuery]:
    if not path.exists():
        return []
    return [EvalQuery(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# LLM-generated natural questions (optional; requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------

GEN_PROMPT = (
    "You are building a retrieval benchmark for research papers. Given one "
    "excerpt from a paper, write {n} distinct questions that this excerpt "
    "answers. Write them the way a researcher skimming the paper would ask — "
    "natural, specific, and standalone. Do NOT quote the excerpt, do NOT reuse "
    "its distinctive phrasing, and do NOT refer to 'the excerpt' or 'the text'. "
    "Return ONLY a JSON array of strings."
)


def build_llm(per_paper: int = 6, questions_per_chunk: int = 2, seed: int = 17) -> list[EvalQuery]:
    """doc2query-style natural questions. No-ops without an API key."""
    from pipeline._http import ProviderError, post_json

    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        print("OPENAI_API_KEY not set — skipping LLM query generation.")
        print("ICT queries are the deterministic fallback and are already sufficient for CI.")
        return []

    cache: dict[str, list[str]] = json.loads(LLM_CACHE.read_text()) if LLM_CACHE.exists() else {}
    chunks = load_chunks()
    by_paper: dict[str, list[dict]] = {}
    for c in chunks:
        by_paper.setdefault(c["paper_id"], []).append(c)

    rng = random.Random(seed)
    out: list[EvalQuery] = []
    for paper_id in sorted(by_paper):
        picks = sorted(by_paper[paper_id], key=lambda c: (c["section"], c["chunk_index"]))
        rng.shuffle(picks)
        for chunk in picks[:per_paper]:
            # Cache on content, not chunk id, so re-chunking does not invalidate
            # judgments we already paid for.
            ckey = hashlib.sha256(
                f"{questions_per_chunk}:{chunk['content']}".encode()
            ).hexdigest()
            if ckey in cache:
                questions = cache[ckey]
            else:
                try:
                    result = post_json(
                        provider="openai",
                        url="https://api.openai.com/v1/chat/completions",
                        headers={"Content-Type": "application/json",
                                 "Authorization": f"Bearer {key}"},
                        body={
                            "model": "gpt-4o-mini",
                            "temperature": 0,
                            "response_format": {"type": "json_object"},
                            "messages": [
                                {"role": "system",
                                 "content": GEN_PROMPT.format(n=questions_per_chunk)
                                 + ' Respond as {"questions": [...]}.'},
                                {"role": "user", "content": chunk["content"]},
                            ],
                        },
                        timeout=45,
                    )
                    payload = json.loads(result["choices"][0]["message"]["content"])
                    questions = [q for q in payload.get("questions", []) if isinstance(q, str)]
                except (ProviderError, KeyError, IndexError, json.JSONDecodeError) as exc:
                    print(f"  gen failed for {chunk['id']}: {exc}")
                    continue
                cache[ckey] = questions
                LLM_CACHE.write_text(json.dumps(cache))
                time.sleep(0.2)

            for q in questions:
                qid = hashlib.sha1(f"{chunk['id']}:{q}".encode()).hexdigest()[:16]
                out.append(EvalQuery(
                    query_id=qid, paper_id=paper_id, query=q,
                    gold_chunk_id=chunk["id"], source="llm", section=chunk["section"],
                ))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ict", "llm"], default="ict")
    ap.add_argument("--per-paper", type=int, default=12)
    args = ap.parse_args()
    if args.mode == "ict":
        write(build_ict(per_paper=args.per_paper), ICT_QUERIES)
    else:
        write(build_llm(per_paper=args.per_paper), LLM_QUERIES)
