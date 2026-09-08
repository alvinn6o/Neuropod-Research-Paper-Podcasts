"""Embed the frozen corpus with a real embedding model, and cache it.

Every retrieval number in this project so far used `HashEmbedder`: SHA256
bag-of-words, 1536 dimensions, zero semantics — "car" and "automobile" land in
unrelated buckets. It was the honest default with no API key, and its
contribution was measured (zeroing the three dense features costs the reranker
0.076 nDCG@10, and they account for 23.8% of LambdaMART's splits), which is
what makes replacing it worth doing: a *hashing scheme* was carrying that much,
so a semantic model has real headroom.

The cache is deliberately NOT committed. It is ~40MB, it is regenerable for
about two cents, and committing it would tempt CI into depending on it. CI
keeps running on hash embeddings, which are deterministic and free, so the gate
stays offline. The OpenAI numbers are a documented comparison, not the gate.

    OPENAI_API_KEY=... python -m eval.embed_corpus
    python -m eval.embed_corpus --check      # report cache state, embed nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# eval/ scripts are run standalone, so nothing has loaded .env for them the way
# api/config.py does for the app. Without this the key is invisible here even
# when it is configured for the service.
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from eval import queries as q_mod
from pipeline.generate.embedder import EMBEDDING_DIM, OpenAIEmbedder
from pipeline.ingest.tokenizer import count_tokens
from pipeline.models import PaperChunk

CACHE = ROOT / "eval" / "corpus" / "_openai_embeddings.json"
MODEL = "text-embedding-3-small"
BATCH = 96


def load_cache() -> dict[str, list[float]]:
    if not CACHE.exists():
        return {}
    raw = json.loads(CACHE.read_text())
    if raw.get("model") != MODEL or raw.get("dim") != EMBEDDING_DIM:
        print(f"cache is for {raw.get('model')}/{raw.get('dim')}d, wanted {MODEL}/{EMBEDDING_DIM}d "
              "— ignoring it rather than mixing spaces")
        return {}
    return raw["vectors"]


def save_cache(vectors: dict[str, list[float]]) -> None:
    CACHE.write_text(json.dumps(
        {"model": MODEL, "dim": EMBEDDING_DIM, "vectors": vectors}))


def text_key(text: str) -> str:
    """Cache key is a hash of the TEXT, not a chunk id.

    Two reasons. Re-chunking or re-pinning the corpus does not invalidate
    vectors already paid for. And more importantly, evaluation does not embed
    the raw chunks: ICT redacts a sentence from every candidate, so the text
    actually scored differs from the text on disk. Keying by id would silently
    return the vector for the *un-redacted* chunk — a mismatch between the
    query's space and the document's, which is precisely the class of bug this
    project spent Phase 0 removing.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def texts_used_in_evaluation() -> dict[str, str]:
    """Every string the harness will actually embed, keyed by text hash."""
    chunks = q_mod.load_chunks()
    queries = q_mod.read(q_mod.ICT_QUERIES)
    by_paper: dict[str, list[dict]] = {}
    for c in chunks:
        by_paper.setdefault(c["paper_id"], []).append(c)

    wanted: dict[str, str] = {}

    def want(text: str) -> None:
        # A chunk that was a single sentence redacts to nothing, and the
        # embeddings API rejects an empty input for the whole batch. The
        # harness already drops those queries; skipping them here keeps one
        # degenerate chunk from failing 96 good ones.
        if text and text.strip():
            wanted[text_key(text)] = text

    for c in chunks:
        want(c["content"])

    # The canonical redacted form of every chunk (redaction is seeded per
    # chunk, so there is exactly one per chunk, not one per query).
    for paper_id, paper_chunks in by_paper.items():
        dummy = q_mod.EvalQuery("", paper_id, "", "", "ict", "")
        for c in q_mod.redact_pool(paper_chunks, dummy):
            want(c["content"])

    # Each query, and its gold chunk with that query's sentence removed.
    by_id = {c["id"]: c for c in chunks}
    for q in queries:
        want(q.query)
        gold = by_id.get(q.gold_chunk_id)
        if gold:
            want(q_mod._strip_sentence(gold["content"], q.query))
    return wanted


class CachedEmbedder:
    """Serves pre-computed vectors, and refuses to guess.

    A miss raises rather than falling back to a hash vector. Falling back is
    exactly the failure Phase 0 removed from the production embedder: it puts
    two incomparable spaces in one index and every cosine after that is
    meaningless while still looking like a number.
    """

    model_id = MODEL

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.dimensions = EMBEDDING_DIM

    def embed_text(self, text: str) -> list[float]:
        # A chunk that was a single sentence redacts to nothing. It has no
        # content to embed and should match nothing, so a zero vector is the
        # honest answer — cosine with it is 0 against everything. This is not
        # a fallback into another space; it is the absence of content.
        if not text or not text.strip():
            return [0.0] * self.dimensions
        vec = self.vectors.get(text_key(text))
        if vec is None:
            raise KeyError(
                f"no cached embedding for {text[:60]!r}... — run "
                "`python -m eval.embed_corpus` after changing the corpus"
            )
        return vec

    def embed_chunks(self, chunks):
        for chunk in chunks:
            chunk.embedding = self.embed_text(chunk.content)
            chunk.embedding_model = self.model_id
            chunk.embedding_dim = self.dimensions
        return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    wanted = texts_used_in_evaluation()

    cache = load_cache()
    missing = {k: v for k, v in wanted.items() if k not in cache}

    tokens = sum(count_tokens(v) for v in missing.values())
    print(f"corpus+queries: {len(wanted)} items")
    print(f"cached        : {len(wanted) - len(missing)}")
    print(f"to embed      : {len(missing)}  (~{tokens:,} tokens, ~${tokens/1e6*0.02:.4f})")

    if args.check or not missing:
        if not missing:
            print("cache is complete.")
        return

    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set")

    embedder = OpenAIEmbedder(api_key=key, model=MODEL, batch=BATCH)
    keys = list(missing)
    start = time.time()
    for i in range(0, len(keys), BATCH):
        window = keys[i : i + BATCH]
        batch = [
            PaperChunk(id=k, paper_id="", section="", chunk_index=0,
                       content=missing[k], token_count=0)
            for k in window
        ]
        for chunk in embedder.embed_chunks(batch):
            cache[chunk.id] = chunk.embedding
        if (i // BATCH) % 10 == 0:
            done = min(i + BATCH, len(keys))
            rate = done / max(time.time() - start, 1e-9)
            print(f"  {done}/{len(keys)}  ({rate:.0f}/s)", flush=True)
            save_cache(cache)     # checkpoint: a crash must not lose paid work

    save_cache(cache)
    print(f"\n{len(cache)} vectors -> {CACHE.name} ({CACHE.stat().st_size/1e6:.1f} MB, gitignored)")


if __name__ == "__main__":
    main()
