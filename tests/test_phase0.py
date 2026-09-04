"""Phase 0 invariants: embedding-space integrity, telemetry, and spend caps.

These are regression tests for bugs that were silent — each one previously
either produced wrong data with no error, or failed the whole job when it
should have degraded.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from api import budget, store_db
from pipeline import usage
from pipeline._http import ProviderError
from pipeline.generate.embedder import EMBEDDING_DIM, EmbeddingError, HashEmbedder, OpenAIEmbedder
from pipeline.generate.retriever import Retriever
from pipeline.generate.scriptwriter import _extract_text
from pipeline.ingest.chunker import SectionAwareChunker
from pipeline.ingest.tokenizer import count_tokens, truncate_to_tokens
from pipeline.models import PaperChunk
from pipeline.synthesize.tts import _OPENAI_TTS_CHAR_LIMIT, _split_for_tts

from .conftest import requires_postgres


# ---------------------------------------------------------------------------
# 0.1 — one embedding model per index
# ---------------------------------------------------------------------------

def test_embedder_records_its_model_on_every_chunk():
    chunks = [PaperChunk("i", "p", "abstract", 0, "state space models", 3)]
    out = HashEmbedder().embed_chunks(chunks)[0]
    assert out.embedding_model == "hash-bow-v1"
    assert out.embedding_dim == EMBEDDING_DIM


def test_openai_embedder_raises_rather_than_mixing_spaces(monkeypatch):
    """A failed batch must NOT be backfilled with hash vectors.

    The old behaviour put SHA256 bag-of-words vectors and OpenAI vectors in the
    same 1536-dim column for the same paper, with nothing recording which was
    which. Cosine between them is meaningless, so retrieval was silently wrong.
    """
    embedder = OpenAIEmbedder(api_key="sk-test")
    monkeypatch.setattr(
        "pipeline.generate.embedder.post_json",
        lambda **kw: (_ for _ in ()).throw(ProviderError("openai-embed", 500, "boom")),
    )
    chunks = [PaperChunk(str(uuid.uuid4()), "p", "abstract", i, f"chunk {i}", 2) for i in range(3)]
    with pytest.raises(EmbeddingError):
        embedder.embed_chunks(chunks)
    assert all(not c.embedding for c in chunks), "must not leave partial vectors behind"


def test_openai_embedder_rejects_short_response(monkeypatch):
    """Fewer vectors than inputs used to leave surplus chunks with embedding=[]."""
    embedder = OpenAIEmbedder(api_key="sk-test")
    monkeypatch.setattr(
        "pipeline.generate.embedder.post_json",
        lambda **kw: {"data": [{"embedding": [0.0] * EMBEDDING_DIM}], "usage": {"prompt_tokens": 1}},
    )
    chunks = [PaperChunk(str(uuid.uuid4()), "p", "abstract", i, f"chunk {i}", 2) for i in range(2)]
    with pytest.raises(EmbeddingError, match="expected 2 vectors"):
        embedder.embed_chunks(chunks)


def test_openai_embedder_rejects_wrong_dimension(monkeypatch):
    embedder = OpenAIEmbedder(api_key="sk-test")
    monkeypatch.setattr(
        "pipeline.generate.embedder.post_json",
        lambda **kw: {"data": [{"embedding": [0.0] * 512}], "usage": {}},
    )
    with pytest.raises(EmbeddingError, match="dim 512"):
        embedder.embed_chunks([PaperChunk("i", "p", "abstract", 0, "x", 1)])


def test_query_embedding_failure_returns_none_not_a_hash_vector(monkeypatch):
    """None routes the query to sparse retrieval, which is at least coherent."""
    embedder = OpenAIEmbedder(api_key="sk-test")
    monkeypatch.setattr(
        "pipeline.generate.embedder.post_json",
        lambda **kw: (_ for _ in ()).throw(ProviderError("openai-embed", 503, "down")),
    )
    assert embedder.embed_text("what are the results?") is None


def test_retriever_refuses_dense_scoring_across_mixed_models():
    e = HashEmbedder(dimensions=32)
    chunks = []
    for i, (content, model) in enumerate([("alpha beta", "hash-bow-v1"), ("gamma delta", "text-embedding-3-small")]):
        chunks.append({
            "id": str(i), "section": "abstract", "content": content,
            "embedding": e.embed_text(content), "embedding_model": model,
        })
    scored = Retriever(embedder=e).retrieve_scored(chunks, "alpha", limit=2)
    # Sparse path taken: dense_score is None for every row.
    assert all(row["dense_score"] is None for row in scored)
    assert all(row["sparse_score"] is not None for row in scored)


def test_store_refuses_to_persist_mixed_model_chunks():
    chunks = [
        {"section": "abstract", "content": "a", "token_count": 1, "embedding_model": "hash-bow-v1"},
        {"section": "results", "content": "b", "token_count": 1, "embedding_model": "text-embedding-3-small"},
    ]
    with pytest.raises(ValueError, match="more than one model"):
        store_db.replace_chunks_for_paper(uuid.uuid4(), chunks)


def test_empty_embedding_encodes_to_null_not_empty_list():
    """'[]' into a vector(1536) column is `must have at least 1 dimension`.

    This never surfaced because every test runs on the SQLite shim, where the
    column is plain TEXT.
    """
    assert store_db._encode_embedding([]) is None
    assert store_db._encode_embedding(None) is None
    assert store_db._encode_embedding([0.5] * EMBEDDING_DIM) is not None
    with pytest.raises(ValueError, match="expects"):
        store_db._encode_embedding([0.5] * 48)


# ---------------------------------------------------------------------------
# 0.2 — real token counts
# ---------------------------------------------------------------------------

def test_token_count_is_not_a_word_count():
    text = "The selective_state_space mechanism achieves 5.2x throughput on A100 GPUs."
    assert count_tokens(text) > len(text.split())


def test_chunks_carry_token_provenance():
    chunk = SectionAwareChunker().chunk_sections("p", {"abstract": "mamba " * 50})[0]
    assert chunk.token_source.startswith(("tiktoken:", "estimate:"))
    assert chunk.token_count > 0


def test_truncate_to_tokens_respects_budget():
    text = "word " * 500
    assert count_tokens(truncate_to_tokens(text, 50)) <= 50
    assert truncate_to_tokens("short text", 1000) == "short text"


# ---------------------------------------------------------------------------
# 0.3 — usage ledger and retrieval traces
# ---------------------------------------------------------------------------

def test_cost_estimation_matches_published_rates():
    assert usage.estimate_cost("claude-sonnet-4-6", prompt_tokens=3000, completion_tokens=1500) == pytest.approx(0.0315)
    assert usage.estimate_cost("gpt-4o-mini", prompt_tokens=3000, completion_tokens=1500) == pytest.approx(0.00135)
    # An unpriced model is logged with a NULL cost rather than counted as free.
    assert usage.estimate_cost("some-new-model", prompt_tokens=1000) is None


def test_cached_tokens_are_priced_at_the_discounted_rate():
    full = usage.estimate_cost("claude-sonnet-4-6", prompt_tokens=1000)
    cached = usage.estimate_cost("claude-sonnet-4-6", cached_read_tokens=1000)
    assert cached == pytest.approx(full * 0.10)


def test_usage_collection_is_scoped_to_its_context():
    with usage.collect() as outer:
        usage.record(usage.LLMCall(purpose="script", provider="anthropic", model="claude-sonnet-4-6"))
        with usage.collect() as inner:
            usage.record(usage.LLMCall(purpose="ask", provider="openai", model="gpt-4o-mini"))
        assert len(inner) == 1
    assert len(outer) == 1, "inner context must not leak into outer"


def test_retriever_exposes_score_components_for_traces():
    chunks = [
        {"id": "a", "section": "results", "content": "we observe 24 percent improvement"},
        {"id": "b", "section": "introduction", "content": "language models are studied"},
    ]
    scored = Retriever().retrieve_scored(chunks, "what improvement?", limit=2)
    assert [r["rank"] for r in scored] == [0, 1]
    for row in scored:
        assert row["section_bonus"] is not None
        assert row["final_score"] == pytest.approx(row["sparse_score"] + row["section_bonus"])


def test_pipeline_persists_retrieval_traces(client, auth_headers):
    client.post("/topics", json={"topics": ["language models"]}, headers=auth_headers)
    run = client.post("/pipeline/run?episodes=1", headers=auth_headers)
    assert run.status_code == 200, run.text
    episodes = client.get("/episodes", headers=auth_headers).json()["items"]
    assert episodes, "expected at least one episode"

    traces = store_db.get_retrieval_traces(uuid.UUID(episodes[0]["id"]))
    assert traces, "retrieval trace was computed and dropped"
    assert traces[0]["rank"] == 0
    assert traces[0]["retriever_version"]
    assert any(t["used_in_prompt"] for t in traces)


# ---------------------------------------------------------------------------
# 0.5 — provider response shape guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    {}, {"content": []}, {"content": [{}]}, {"content": [{"text": "   "}]},
    {"error": {"type": "overloaded"}}, {"content": "not a list"},
])
def test_malformed_provider_response_raises_provider_error(payload):
    """Must be ProviderError, not KeyError.

    A KeyError is not caught by `except ProviderError` in the fallback chain, so
    a shape change failed the entire job instead of trying the next provider.
    """
    with pytest.raises(ProviderError):
        _extract_text(payload, "anthropic")


def test_well_formed_response_is_stripped():
    assert _extract_text({"content": [{"text": "  hello  "}]}, "anthropic") == "hello"
    assert _extract_text({"choices": [{"message": {"content": " hi "}}]}, "openai") == "hi"


# ---------------------------------------------------------------------------
# Spend caps
# ---------------------------------------------------------------------------

def test_llm_disabled_context_flips_the_guard():
    assert usage.llm_allowed()
    with usage.llm_disabled("over budget"):
        assert not usage.llm_allowed()
    assert usage.llm_allowed()


def test_scriptwriter_degrades_to_demo_when_over_budget(monkeypatch):
    """Over budget must produce a usable page, not a 500."""
    from pipeline.generate.scriptwriter import ScriptWriter
    from pipeline.models import PaperCandidate

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-be-used")
    writer = ScriptWriter()

    def explode(**kwargs):
        raise AssertionError("no provider call may be made while over budget")

    monkeypatch.setattr("pipeline.generate.scriptwriter.post_json", explode)
    candidate = PaperCandidate(
        arxiv_id="1", title="T", abstract="An abstract sentence.", authors=["A"],
        categories=["cs.LG"], published_at="2026-01-01T00:00:00Z", pdf_url="", sections={},
    )
    with usage.llm_disabled("monthly budget reached"):
        script, provider = writer.write(candidate, [], ["topic"])
    assert provider == "demo-budget", "degradation must be visible in the data"
    assert script


def test_spend_ledger_accumulates_and_trips_the_budget(monkeypatch):
    calls = [usage.LLMCall(
        purpose="script", provider="anthropic", model="claude-sonnet-4-6",
        prompt_tokens=3000, completion_tokens=1500,
    ) for _ in range(3)]
    before = store_db.spend_usd_since(datetime.now(timezone.utc) - timedelta(hours=1))
    store_db.insert_llm_calls(calls)
    after = store_db.spend_usd_since(datetime.now(timezone.utc) - timedelta(hours=1))
    assert after == pytest.approx(before + 3 * 0.0315, abs=1e-6)

    # With the ceiling set below current spend, the guard must close.
    from api.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("NEUROPOD_MONTHLY_BUDGET_USD", "0.01")
    try:
        allowed, reason = budget.llm_spend_allowed()
        assert not allowed
        assert "monthly budget" in reason
    finally:
        get_settings.cache_clear()


def test_global_run_cap_rejects_even_a_fresh_identity(client, monkeypatch):
    """A new account must not reset the global cap — that is the whole point.

    Per-user quotas are free to reset via /auth/stub/login, so the global
    counter is the control that actually bounds spend.
    """
    from api.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("NEUROPOD_GLOBAL_DAILY_RUN_LIMIT", "1")
    try:
        first = client.post("/auth/stub/login", json={"email": f"a-{uuid.uuid4().hex[:6]}@x.com"}).json()["token"]
        h1 = {"Authorization": f"Bearer {first}"}
        client.post("/topics", json={"topics": ["ml"]}, headers=h1)
        client.post("/pipeline/run?episodes=1", headers=h1)

        # Brand new identity, brand new per-user quota — still blocked.
        second = client.post("/auth/stub/login", json={"email": f"b-{uuid.uuid4().hex[:6]}@x.com"}).json()["token"]
        h2 = {"Authorization": f"Bearer {second}"}
        client.post("/topics", json={"topics": ["ml"]}, headers=h2)
        blocked = client.post("/pipeline/run?episodes=1", headers=h2)
        assert blocked.status_code == 429
        assert "global daily run limit" in blocked.json()["detail"]
    finally:
        get_settings.cache_clear()


def test_status_exposes_budget_without_auth(client):
    body = client.get("/status").json()
    assert "budget" in body
    assert set(body["budget"]) >= {"month_usd", "monthly_budget_usd", "llm_enabled"}


# ---------------------------------------------------------------------------
# TTS truncation
# ---------------------------------------------------------------------------

def test_tts_splits_instead_of_dropping_a_third_of_the_script():
    """`script[:4000]` silently discarded ~35% of every episode."""
    script = ("This is a narration sentence about state space models. " * 140).strip()
    assert len(script) > 4000
    parts = _split_for_tts(script, limit=_OPENAI_TTS_CHAR_LIMIT)
    assert len(parts) > 1
    assert all(len(p) <= _OPENAI_TTS_CHAR_LIMIT for p in parts)
    # Only the whitespace joining sentences is lost.
    assert sum(len(p) for p in parts) + (len(parts) - 1) == len(script)


def test_tts_hard_splits_an_overlong_sentence_rather_than_dropping_it():
    text = "x" * 9000
    parts = _split_for_tts(text, limit=3800)
    assert sum(len(p) for p in parts) == 9000


# ---------------------------------------------------------------------------
# Postgres-only. These are the assertions the SQLite shim cannot make, because
# it maps `vector(1536)` to TEXT and accepts anything.
# ---------------------------------------------------------------------------

@requires_postgres
def test_chunk_without_embedding_persists_on_real_pgvector():
    """The bug that breaks first on deploy.

    `json.dumps(embedding or [])` wrote the string "[]" into a vector(1536)
    column; pgvector rejects it with "vector must have at least 1 dimension",
    so one un-embedded chunk failed the whole paper's insert.
    """
    paper_id = store_db.upsert_paper({
        "arxiv_id": f"test.{uuid.uuid4().hex[:8]}", "title": "T", "authors": ["A"],
        "abstract": "a", "categories": ["cs.LG"],
        "published_at": "2026-01-01T00:00:00Z", "pdf_url": "", "citation_count": 0,
    })
    store_db.replace_chunks_for_paper(paper_id, [
        {"section": "abstract", "content": "embedded", "token_count": 1,
         "embedding": [0.01] * EMBEDDING_DIM, "embedding_model": "hash-bow-v1",
         "embedding_dim": EMBEDDING_DIM, "chunk_index": 0},
        # No embedding at all — this is the row that used to blow up.
        {"section": "results", "content": "not embedded", "token_count": 1,
         "embedding": [], "chunk_index": 1},
    ])
    rows = store_db.get_chunks_for_paper(paper_id)
    assert len(rows) == 2
    assert len(rows[0]["embedding"]) == EMBEDDING_DIM
    assert rows[1]["embedding"] == []


@requires_postgres
def test_wrong_dimension_is_rejected_before_it_reaches_pgvector():
    paper_id = store_db.upsert_paper({
        "arxiv_id": f"test.{uuid.uuid4().hex[:8]}", "title": "T", "authors": ["A"],
        "abstract": "a", "categories": ["cs.LG"],
        "published_at": "2026-01-01T00:00:00Z", "pdf_url": "", "citation_count": 0,
    })
    with pytest.raises(ValueError, match="expects"):
        store_db.replace_chunks_for_paper(paper_id, [
            {"section": "abstract", "content": "x", "token_count": 1,
             "embedding": [0.1] * 512, "chunk_index": 0},
        ])
