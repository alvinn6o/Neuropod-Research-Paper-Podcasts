"""Per-call LLM usage and cost accounting.

Every provider response already carries a `usage` block; before this module all
six call sites read `["content"][0]["text"]` and threw the rest away. Without it
there is no cost curve, no tokens-per-episode number, no way to evaluate whether
a cheaper model is worth it, and no way to enforce a spend cap.

The pipeline package must stay importable without the API package (the worker
imports `api`, not the reverse), so call sites push records into a context-local
buffer here and `api.pipeline_runner` / the route handlers drain it and persist.
`contextvars` rather than a module global so concurrent requests in the same
process don't pool their costs together.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

# USD per 1M tokens. Kept explicit rather than fetched: a wrong number here
# under-bills the kill-switch, so it should change only by deliberate edit.
PRICING: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}

# Anthropic prompt-caching multipliers against the base input rate.
_CACHE_WRITE_MULT = 1.25
_CACHE_READ_MULT = 0.10

# Models we have no price for still get logged, with cost_usd=None so the
# ledger can tell "free" apart from "unpriced".
UNPRICED = None


@dataclass
class LLMCall:
    purpose: str                 # script | ask | judge | embed
    provider: str                # anthropic | openai | bedrock | demo
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_read_tokens: int = 0
    cached_write_tokens: int = 0
    latency_ms: int = 0
    ok: bool = True
    status: int | None = None
    error: str | None = None
    cost_usd: float | None = field(default=None)

    def __post_init__(self) -> None:
        if self.cost_usd is None:
            self.cost_usd = estimate_cost(
                self.model,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                cached_read_tokens=self.cached_read_tokens,
                cached_write_tokens=self.cached_write_tokens,
            )


def estimate_cost(
    model: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_read_tokens: int = 0,
    cached_write_tokens: int = 0,
) -> float | None:
    price = PRICING.get(model)
    if price is None:
        return UNPRICED
    in_rate, out_rate = price
    # Cached tokens are reported separately by Anthropic and are NOT included in
    # prompt_tokens, so they are added rather than subtracted.
    total = (
        prompt_tokens * in_rate
        + completion_tokens * out_rate
        + cached_read_tokens * in_rate * _CACHE_READ_MULT
        + cached_write_tokens * in_rate * _CACHE_WRITE_MULT
    )
    return round(total / 1_000_000, 8)


_buffer: contextvars.ContextVar[list[LLMCall] | None] = contextvars.ContextVar(
    "neuropod_llm_calls", default=None
)


@contextmanager
def collect() -> Iterator[list[LLMCall]]:
    """Collect every LLMCall recorded inside this block.

    Nesting is intentional: an inner block gets its own buffer, so a sub-task
    can account for itself without double-counting into the outer one.
    """
    calls: list[LLMCall] = []
    token = _buffer.set(calls)
    try:
        yield calls
    finally:
        _buffer.reset(token)


def record(call: LLMCall) -> None:
    calls = _buffer.get()
    if calls is not None:
        calls.append(call)


# ---------------------------------------------------------------------------
# Response-shape parsing. These are the guards for Phase 0.5: a provider that
# changes its response shape used to raise KeyError, which is not ProviderError,
# so it escaped the fallback chain and failed the whole job instead of falling
# through to the next provider.
# ---------------------------------------------------------------------------

def anthropic_usage(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("usage") or {}
    return {
        "prompt_tokens": int(raw.get("input_tokens") or 0),
        "completion_tokens": int(raw.get("output_tokens") or 0),
        "cached_read_tokens": int(raw.get("cache_read_input_tokens") or 0),
        "cached_write_tokens": int(raw.get("cache_creation_input_tokens") or 0),
    }


def openai_usage(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("usage") or {}
    cached = ((raw.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    prompt = int(raw.get("prompt_tokens") or 0)
    return {
        # OpenAI includes cached tokens inside prompt_tokens; split them out so
        # the two providers are priced by the same rule.
        "prompt_tokens": max(prompt - int(cached), 0),
        "completion_tokens": int(raw.get("completion_tokens") or 0),
        "cached_read_tokens": int(cached),
        "cached_write_tokens": 0,
    }


# ---------------------------------------------------------------------------
# Budget kill-switch.
#
# The spend ledger lives in the database, which the pipeline package cannot
# import. So the *decision* is made by the caller (api.pipeline_runner, the ask
# route) and communicated down as a context-local flag. Call sites check
# `llm_allowed()` and fall through to their existing zero-cost path — the demo
# scriptwriter and the deterministic metadata answerer — rather than erroring.
# Degrading is the right behaviour for a public demo: the page still works.
# ---------------------------------------------------------------------------

_llm_allowed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "neuropod_llm_allowed", default=True
)


@contextmanager
def llm_disabled(reason: str = "budget") -> Iterator[None]:
    token = _llm_allowed.set(False)
    try:
        yield
    finally:
        _llm_allowed.reset(token)


def llm_allowed() -> bool:
    return _llm_allowed.get()
