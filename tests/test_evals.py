"""Deterministic evaluations for the RAG + script pipeline.

These run in CI without API credentials — they exercise the retrieval scoring,
chunking invariants, QA heuristic, and embedder behavior using the demo
fallback adapters. For the LLM-as-judge eval (Ragas), see eval/ragas_eval.py."""
from __future__ import annotations

from pipeline.generate.embedder import HashEmbedder, get_embedder
from pipeline.generate.qa_check import QAChecker
from pipeline.generate.retriever import Retriever
from pipeline.ingest.chunker import SectionAwareChunker


# ---------------------------------------------------------------------------
# Chunker invariants
# ---------------------------------------------------------------------------

def test_chunker_respects_max_words():
    chunker = SectionAwareChunker(max_words=50, overlap_words=10)
    text = " ".join(["word"] * 240)
    chunks = chunker.chunk_sections("paper-1", {"introduction": text})

    assert chunks, "chunker should produce at least one chunk"
    for chunk in chunks:
        assert chunk.token_count <= 50, f"chunk over cap: {chunk.token_count}"


def test_chunker_preserves_section_label():
    chunker = SectionAwareChunker(max_words=30, overlap_words=5)
    sections = {"abstract": " ".join(["a"] * 100), "results": " ".join(["r"] * 80)}
    chunks = chunker.chunk_sections("paper-2", sections)

    section_counts = {"abstract": 0, "results": 0}
    for chunk in chunks:
        assert chunk.section in section_counts
        section_counts[chunk.section] += 1
    assert section_counts["abstract"] > 1
    assert section_counts["results"] > 1


def test_chunker_short_section_emits_single_chunk():
    chunker = SectionAwareChunker(max_words=200, overlap_words=20)
    chunks = chunker.chunk_sections("paper-3", {"methods": "short methods text"})
    assert len(chunks) == 1
    assert chunks[0].section == "methods"


# ---------------------------------------------------------------------------
# Embedder shape + auto-selection
# ---------------------------------------------------------------------------

def test_hash_embedder_returns_unit_norm_vector():
    embedder = HashEmbedder(dimensions=48)
    vector = embedder.embed_text("retrieval augmented generation")
    assert len(vector) == 48
    norm = sum(value * value for value in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_hash_embedder_handles_empty_input():
    vector = HashEmbedder().embed_text("   ")
    assert all(v == 0.0 for v in vector)


def test_embedder_selection_falls_back_without_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedder = get_embedder()
    assert isinstance(embedder, HashEmbedder)


# ---------------------------------------------------------------------------
# Retriever: ranking semantics
# ---------------------------------------------------------------------------

def _chunk(section: str, content: str, idx: int = 0) -> dict:
    return {
        "id": f"c-{section}-{idx}",
        "paper_id": "p1",
        "section": section,
        "chunk_index": idx,
        "content": content,
        "token_count": len(content.split()),
        "embedding": [],
    }


def test_retriever_returns_top_k():
    chunks = [
        _chunk("abstract", "transformers attention scaling laws", 0),
        _chunk("methods", "we propose a sparse routing layer", 1),
        _chunk("results", "we observe 24 percent improvement", 2),
        _chunk("conclusion", "future work explores efficiency", 3),
    ]
    out = Retriever().retrieve(chunks, "what are the results?", limit=2)
    assert len(out) == 2


def test_retriever_does_not_apply_a_section_prior():
    """The hand-set section prior is deliberately NOT in the scoring path.

    This test previously asserted the opposite — that a `results` chunk should
    outrank an identical `introduction` chunk purely because of its section.
    Measured on the 168-paper benchmark, that prior is harmful: dense ->
    dense+prior costs 0.034 nDCG@10, CI [-0.042, -0.025], p<0.001, and
    rrf -> rrf+prior costs 0.011, p<0.001. The weights were never fit to
    anything.

    With identical text, the two chunks must now tie on score. `section_bonus`
    survives as a class attribute only so eval/ can reproduce the old baseline
    and the learned reranker's section weights can be read against it.
    """
    chunks = [
        _chunk("introduction", "language models are widely studied", 0),
        _chunk("results", "language models are widely studied", 1),
    ]
    scored = Retriever().retrieve_scored(chunks, "language models", limit=2)
    assert scored[0]["final_score"] == scored[1]["final_score"], (
        "identical text in different sections must score identically"
    )
    assert all(row["section_bonus"] == 0.0 for row in scored)


def test_retriever_uses_dense_embeddings_when_present():
    embedder = HashEmbedder(dimensions=32)
    chunks = []
    for idx, content in enumerate([
        "retrieval augmented generation pipeline",
        "completely unrelated text about marine biology",
    ]):
        chunk = _chunk("abstract", content, idx)
        chunk["embedding"] = embedder.embed_text(content)
        chunks.append(chunk)

    top = Retriever(embedder=embedder).retrieve(chunks, "RAG retrieval", limit=1)
    assert "retrieval" in top[0]["content"]


def test_retriever_answer_question_includes_citation_grounded_text():
    chunks = [
        _chunk("results", "Hallucinations dropped by 24 percent on document QA.", 0),
        _chunk("methods", "We add a small verifier model after generation.", 1),
    ]
    answer = Retriever().answer_question(
        {"title": "Self-Verification Loops"},
        chunks,
        "what did the authors measure?",
    )
    assert "Self-Verification Loops" in answer
    assert "Hallucinations" in answer or "verifier" in answer


# ---------------------------------------------------------------------------
# QA checker — script-vs-source grounding
# ---------------------------------------------------------------------------

def test_qa_verifier_passes_grounded_script():
    chunks = [
        _chunk("abstract", "We introduce a self-verification loop for small language models."),
        _chunk("results", "Hallucinations dropped by 24 percent on document QA."),
    ]
    script = (
        "Researchers introduce a self-verification loop for small language models. "
        "Hallucinations dropped by 24 percent on document QA."
    )
    status, _ = QAChecker().verify(script, chunks)
    assert status == "verified"


def test_qa_verifier_flags_ungrounded_script():
    chunks = [
        _chunk("abstract", "We study sparse routing layers for long context."),
    ]
    script = (
        "The authors propose a novel diffusion model for protein folding "
        "trained on AlphaFold's residue embeddings."
    )
    status, _ = QAChecker().verify(script, chunks)
    assert status == "flagged"
