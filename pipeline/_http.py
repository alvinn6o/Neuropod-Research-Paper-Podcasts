"""Shared HTTP helper for provider calls: retry, backoff, structured errors."""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("neuropod.http")


class ProviderError(RuntimeError):
    def __init__(self, provider: str, status: int | None, detail: str) -> None:
        super().__init__(f"[{provider}] {status or 'error'}: {detail}")
        self.provider = provider
        self.status = status
        self.detail = detail


def post_json(
    *,
    provider: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | bytes,
    timeout: int = 60,
    retries: int = 2,
    backoff: float = 1.5,
) -> dict[str, Any]:
    payload = body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode()
    last: ProviderError | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                return json.loads(raw.decode())
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode()[:400]
            except Exception:
                detail = str(exc)
            last = ProviderError(provider, exc.code, detail)
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                sleep_for = backoff ** attempt
                logger.warning("%s retrying after %.1fs (status=%s)", provider, sleep_for, exc.code)
                time.sleep(sleep_for)
                continue
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = ProviderError(provider, None, str(exc))
            if attempt < retries:
                sleep_for = backoff ** attempt
                time.sleep(sleep_for)
                continue
            break
        except Exception as exc:
            last = ProviderError(provider, None, str(exc))
            break

    assert last is not None
    raise last


def post_for_bytes(
    *,
    provider: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: int = 120,
    retries: int = 2,
    backoff: float = 1.5,
) -> bytes:
    payload = json.dumps(body).encode()
    last: ProviderError | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode()[:400]
            except Exception:
                detail = str(exc)
            last = ProviderError(provider, exc.code, detail)
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(backoff ** attempt)
                continue
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = ProviderError(provider, None, str(exc))
            if attempt < retries:
                time.sleep(backoff ** attempt)
                continue
            break
        except Exception as exc:
            last = ProviderError(provider, None, str(exc))
            break
    assert last is not None
    raise last
