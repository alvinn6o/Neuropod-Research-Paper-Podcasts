"""Pre-compute chunks + embeddings for the recall@k benchmark.

Why: the test fixture needs deterministic chunks + their embeddings so
`tests/test_recall.py` can run in CI without burning API tokens. Run this
once locally; commit the JSON outputs.

Usage:
    # Free / deterministic (default — uses hash embedder):
    python -m eval.precompute_fixtures

    # Real OpenAI embeddings (better recall numbers; costs ~$0.001):
    OPENAI_API_KEY=sk-... NEUROPOD_EMBEDDER=openai python -m eval.precompute_fixtures
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.generate.embedder import get_embedder
from pipeline.ingest.chunker import SectionAwareChunker
from pipeline.ingest.pdf_extractor import PDFExtractor
from pipeline.models import PaperCandidate

PDF_PATH = ROOT / "tests" / "fixtures" / "mamba.pdf"
CHUNKS_PATH = ROOT / "tests" / "fixtures" / "mamba_chunks.json"
META_PATH = ROOT / "tests" / "fixtures" / "mamba_meta.json"


def main() -> None:
    print(f"reading {PDF_PATH}")
    pdf_bytes = PDF_PATH.read_bytes()

    candidate = PaperCandidate(
        arxiv_id="2312.00752",
        title="Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        abstract="",
        authors=["Albert Gu", "Tri Dao"],
        categories=["cs.LG"],
        published_at="2023-12-01T00:00:00Z",
        pdf_url="https://arxiv.org/pdf/2312.00752",
        sections={},
    )

    extractor = PDFExtractor()
    sections = extractor._extract_from_pdf(pdf_bytes)
    print(f"extracted {len(sections)} sections: {list(sections)}")

    chunker = SectionAwareChunker(max_words=110, overlap_words=24)
    chunks = chunker.chunk_sections(candidate.arxiv_id, sections)
    print(f"chunked into {len(chunks)} pieces")

    embedder = get_embedder()
    backend = type(embedder).__name__
    print(f"embedding with {backend}...")
    embedder.embed_chunks(chunks)

    chunk_dicts = [asdict(chunk) for chunk in chunks]
    CHUNKS_PATH.write_text(json.dumps(chunk_dicts, indent=2))
    print(f"wrote {CHUNKS_PATH} ({CHUNKS_PATH.stat().st_size:,} bytes)")

    META_PATH.write_text(json.dumps({
        "paper": {
            "arxiv_id": candidate.arxiv_id,
            "title": candidate.title,
            "authors": candidate.authors,
            "categories": candidate.categories,
        },
        "chunk_count": len(chunks),
        "embedder_backend": backend,
        "embedding_dim": len(chunks[0].embedding) if chunks else 0,
        "sections": {name: len(text) for name, text in sections.items()},
    }, indent=2))
    print(f"wrote {META_PATH}")


if __name__ == "__main__":
    main()
