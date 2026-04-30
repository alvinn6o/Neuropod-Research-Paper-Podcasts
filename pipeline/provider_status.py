"""Process-wide tracker for provider call outcomes — surfaced via /status."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

_lock = threading.Lock()
_state: dict[str, dict[str, Any]] = {}


def record_success(provider: str, *, latency_ms: int | None = None) -> None:
    with _lock:
        _state[provider] = {
            "ok": True,
            "at": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            "error": None,
        }


def record_failure(provider: str, *, error: str, status: int | None = None) -> None:
    with _lock:
        _state[provider] = {
            "ok": False,
            "at": datetime.now(timezone.utc).isoformat(),
            "latency_ms": None,
            "error": error[:300],
            "status": status,
        }


def snapshot() -> dict[str, dict[str, Any]]:
    with _lock:
        return {k: dict(v) for k, v in _state.items()}
