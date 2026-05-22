"""Verify the Anthropic-529 retry + fallback chain works as designed.

These tests mock the HTTP layer so they run in CI without real API keys
or network. They simulate Anthropic returning 529 ("Overloaded") and
confirm:

  1. 529 is in the retryable status set, so post_json retries with backoff.
  2. After exhausting retries, post_json raises ProviderError(status=529).
  3. ScriptWriter catches that and falls through to the next provider in
     the chain (OpenAI → demo).
"""
from __future__ import annotations

import urllib.error
from io import BytesIO

import pytest

from pipeline import _http
from pipeline._http import ProviderError, _RETRYABLE_STATUS, post_json
from pipeline.generate.scriptwriter import ScriptWriter
from pipeline.models import PaperCandidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Counter:
    """Tracks how many times a mocked function was called."""
    def __init__(self) -> None:
        self.calls = 0


def _make_http_error(status: int, body: bytes = b'{"error":"overloaded"}') -> urllib.error.HTTPError:
    """Build a urllib HTTPError with a readable body — what real arXiv/Anthropic returns."""
    err = urllib.error.HTTPError(
        url="https://api.anthropic.com/v1/messages",
        code=status,
        msg="overloaded",
        hdrs=None,  # type: ignore[arg-type]
        fp=BytesIO(body),
    )
    return err


# ---------------------------------------------------------------------------
# Bug-fix invariant: 529 is in the retry whitelist
# ---------------------------------------------------------------------------

def test_anthropic_529_is_retryable():
    """Anthropic explicitly documents 529 as retryable; we must include it."""
    assert 529 in _RETRYABLE_STATUS


# ---------------------------------------------------------------------------
# post_json retry behavior
# ---------------------------------------------------------------------------

def test_post_json_retries_on_529_then_gives_up(monkeypatch):
    """If 529 keeps coming back, post_json should retry (retries+1 attempts) then raise."""
    counter = _Counter()

    def always_fail(req, timeout):
        counter.calls += 1
        raise _make_http_error(529)

    monkeypatch.setattr(_http.urllib.request, "urlopen", always_fail)
    # Skip the time.sleep delay so the test runs in milliseconds.
    monkeypatch.setattr(_http.time, "sleep", lambda _: None)

    with pytest.raises(ProviderError) as excinfo:
        post_json(
            provider="anthropic",
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "test"},
            body={"messages": []},
            retries=2,
        )

    assert excinfo.value.status == 529
    # retries=2 means up to 3 total attempts (initial + 2 retries)
    assert counter.calls == 3, f"expected 3 attempts, got {counter.calls}"


def test_post_json_succeeds_after_transient_529(monkeypatch):
    """A 529 followed by a 200 should return the parsed body — retry actually worked."""
    counter = _Counter()
    success_body = b'{"content":[{"text":"ok"}]}'

    class _StubResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return None

    def fail_then_succeed(req, timeout):
        counter.calls += 1
        if counter.calls == 1:
            raise _make_http_error(529)
        return _StubResponse(success_body)

    monkeypatch.setattr(_http.urllib.request, "urlopen", fail_then_succeed)
    monkeypatch.setattr(_http.time, "sleep", lambda _: None)

    result = post_json(
        provider="anthropic",
        url="https://api.anthropic.com/v1/messages",
        headers={"x-api-key": "test"},
        body={"messages": []},
        retries=2,
    )

    assert result == {"content": [{"text": "ok"}]}
    assert counter.calls == 2, "should have succeeded on the second attempt"


# ---------------------------------------------------------------------------
# ScriptWriter fallback chain — the integration story
# ---------------------------------------------------------------------------

def _candidate() -> PaperCandidate:
    return PaperCandidate(
        arxiv_id="9999.test",
        title="Test Paper",
        abstract="We propose a thing that does a thing in a clever way. " * 4,
        authors=["A. Author"],
        categories=["cs.AI"],
        published_at="2026-05-22T00:00:00Z",
        pdf_url="",
        sections={"abstract": "Abstract content here for the fallback."},
    )


def test_scriptwriter_falls_through_to_demo_when_anthropic_529s(monkeypatch):
    """The Anthropic 529 path must not poison the run — fall through to demo template."""
    # conftest forces provider=demo for the suite; flip it to auto for this test.
    monkeypatch.setenv("NEUROPOD_LLM_PROVIDER", "auto")
    # Pretend the operator has only Anthropic configured (no OpenAI key)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def always_529(req, timeout):
        raise _make_http_error(529)

    monkeypatch.setattr(_http.urllib.request, "urlopen", always_529)
    monkeypatch.setattr(_http.time, "sleep", lambda _: None)

    writer = ScriptWriter()
    script, provider = writer.write(
        candidate=_candidate(),
        retrieved_chunks=[
            {"section": "abstract", "content": "Abstract content here for the fallback."},
        ],
        audience_topics=["agents"],
    )

    assert provider == "demo", f"expected fallback to demo, got {provider}"
    assert len(script) > 100, "demo script should still produce real text"
    assert "today's paper" in script.lower() or "the paper" in script.lower()


def test_scriptwriter_uses_openai_when_anthropic_529s_and_openai_works(monkeypatch):
    """Real-world degradation: Anthropic 529s, OpenAI is up, episode still generates via OpenAI."""
    monkeypatch.setenv("NEUROPOD_LLM_PROVIDER", "auto")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    state = {"call_count": 0}

    class _OpenAIResponse:
        body = b'{"choices":[{"message":{"content":"OpenAI-generated script body."}}]}'
        def read(self): return self.body
        def __enter__(self): return self
        def __exit__(self, *_): return None

    def router(req, timeout):
        state["call_count"] += 1
        # Anthropic 529s on every retry; OpenAI succeeds on first try.
        if "anthropic.com" in req.full_url:
            raise _make_http_error(529)
        if "openai.com" in req.full_url:
            return _OpenAIResponse()
        raise AssertionError(f"unexpected URL: {req.full_url}")

    monkeypatch.setattr(_http.urllib.request, "urlopen", router)
    monkeypatch.setattr(_http.time, "sleep", lambda _: None)

    writer = ScriptWriter()
    script, provider = writer.write(
        candidate=_candidate(),
        retrieved_chunks=[{"section": "abstract", "content": "Abstract content."}],
        audience_topics=["agents"],
    )

    assert provider == "openai", f"expected openai fallback, got {provider}"
    assert "OpenAI-generated" in script
    # Anthropic should have been retried (3 attempts) before OpenAI fired (1 call)
    assert state["call_count"] >= 4, f"expected >=4 total calls, got {state['call_count']}"
