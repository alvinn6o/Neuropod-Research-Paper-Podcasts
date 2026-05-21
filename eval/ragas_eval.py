"""LLM-as-judge evaluation for the script generator.

Run manually:  python -m eval.ragas_eval

This DOES make real API calls (~$0.10 per paper). Skipped in CI by design.

Metrics (mirrors Ragas' definitions, implemented without the dependency):
  * faithfulness        — fraction of claims in script verifiably grounded in retrieved chunks
  * answer_relevancy    — does the script address the audience topics it was generated for
  * context_precision   — fraction of retrieved chunks that contributed to the final script

For each paper in the test set we generate a script via the live pipeline,
then use the same LLM as judge (different system prompt) to score it.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline._http import ProviderError, post_json
from pipeline.demo_catalog import get_demo_catalog
from pipeline.discover.arxiv_client import ArxivClient
from pipeline.generate.embedder import get_embedder
from pipeline.generate.retriever import Retriever
from pipeline.generate.scriptwriter import ScriptWriter
from pipeline.ingest.chunker import SectionAwareChunker
from pipeline.ingest.pdf_extractor import PDFExtractor


@dataclass
class EvalResult:
    paper_title: str
    script_words: int
    faithfulness: float
    answer_relevancy: float
    context_precision: float


JUDGE_SYSTEM = (
    "You are a strict research-podcast quality evaluator. You return ONLY a JSON "
    "object with three floats between 0 and 1, no prose:\n"
    "{\n"
    '  "faithfulness": <fraction of script claims grounded in the provided chunks>,\n'
    '  "answer_relevancy": <how well the script addresses the audience topics>,\n'
    '  "context_precision": <fraction of provided chunks that were actually used>\n'
    "}\n"
    "Be conservative — round down when uncertain."
)


def _judge(script: str, chunks: list[dict], topics: list[str], openai_key: str) -> dict[str, float]:
    chunks_block = "\n\n".join(f"[{c['section']}] {c['content']}" for c in chunks[:10])
    user = (
        f"AUDIENCE TOPICS: {', '.join(topics)}\n\n"
        f"RETRIEVED CHUNKS:\n{chunks_block}\n\n"
        f"GENERATED SCRIPT:\n{script}\n\n"
        "Score the script."
    )
    result = post_json(
        provider="openai",
        url="https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}",
        },
        body={
            "model": "gpt-4o-mini",
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
        },
        timeout=45,
    )
    payload = json.loads(result["choices"][0]["message"]["content"])
    return {
        "faithfulness": float(payload.get("faithfulness", 0.0)),
        "answer_relevancy": float(payload.get("answer_relevancy", 0.0)),
        "context_precision": float(payload.get("context_precision", 0.0)),
    }


def run() -> list[EvalResult]:
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        print("OPENAI_API_KEY required to run the LLM-as-judge eval.", file=sys.stderr)
        sys.exit(2)

    topics = ["language models", "retrieval augmented generation", "agents"]
    keys = {"openai": openai_key, "anthropic": os.getenv("ANTHROPIC_API_KEY", "")}

    # Use the curated demo catalog as the held-out set.
    discovery = ArxivClient()
    candidates = discovery.search(topics=topics, max_results=3)[:3]

    extractor = PDFExtractor()
    chunker = SectionAwareChunker()
    embedder = get_embedder(keys=keys)
    retriever = Retriever(embedder=embedder)
    writer = ScriptWriter(keys=keys)

    results: list[EvalResult] = []
    for candidate in candidates:
        print(f"\n--- {candidate.title}")
        sections = extractor.extract_sections(candidate)
        chunk_models = chunker.chunk_sections("eval-paper", sections)
        chunk_models = embedder.embed_chunks(chunk_models)
        chunk_dicts = [c.to_dict() for c in chunk_models]
        retrieved = retriever.retrieve(chunk_dicts, f"{candidate.title} {candidate.abstract}", limit=10)

        t0 = time.time()
        script, label = writer.write(candidate, retrieved, topics)
        print(f"   generated ({label}) in {int(time.time() - t0)}s — {len(script.split())} words")

        scores = _judge(script, retrieved, topics, openai_key)
        results.append(EvalResult(
            paper_title=candidate.title,
            script_words=len(script.split()),
            **scores,
        ))
        print(f"   faithfulness={scores['faithfulness']:.2f}  relevancy={scores['answer_relevancy']:.2f}  precision={scores['context_precision']:.2f}")

    print("\n=== Summary ===")
    for metric in ("faithfulness", "answer_relevancy", "context_precision"):
        values = [getattr(r, metric) for r in results]
        print(f"  {metric:<22} mean={statistics.mean(values):.2f}  min={min(values):.2f}")
    return results


if __name__ == "__main__":
    run()
