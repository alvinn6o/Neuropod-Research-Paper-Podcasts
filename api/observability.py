"""JSON-structured logs + request-id middleware for CloudWatch-friendly traces."""
from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("neuropod_request_id", default="-")


def current_request_id() -> str:
    return _request_id_var.get()


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", _request_id_var.get()),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for extra_key in ("user_id", "duration_ms", "status_code", "path", "method"):
            value = getattr(record, extra_key, None)
            if value is not None:
                payload[extra_key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Replace the root handler with a JSON one. Idempotent."""
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonLogFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    # quiet uvicorn defaults — they have their own format we override
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Generates / reads X-Request-Id, sets contextvar, logs each request once."""

    LOGGER = logging.getLogger("neuropod.access")

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = _request_id_var.set(rid)
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-Id"] = rid
            return response
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            self.LOGGER.info(
                "request",
                extra={
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                },
            )
            _request_id_var.reset(token)
