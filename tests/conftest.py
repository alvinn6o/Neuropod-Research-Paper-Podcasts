"""Shared pytest fixtures."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

# Force stub-mode auth so tests can exercise the pipeline with demo providers
# and no external API calls.
os.environ.setdefault("NEUROPOD_AUTH_MODE", "stub")
# Default to the SQLite shim, but honour an externally-set URL so the same
# suite can run against real Postgres + pgvector in CI. This matters: the shim
# rewrites `vector(1536)` to TEXT and strips the HNSW index, so bugs at the
# pgvector boundary (an empty embedding serialized as "[]", a dimension
# mismatch) are invisible when only the shim is exercised.
os.environ.setdefault("NEUROPOD_DATABASE_URL", "")
os.environ["NEUROPOD_LIVE_DISCOVERY"] = "false"
os.environ["NEUROPOD_LLM_PROVIDER"] = "demo"
os.environ["NEUROPOD_TTS_PROVIDER"] = "demo"
os.environ["NEUROPOD_EMBEDDER"] = "demo"
os.environ["NEUROPOD_GENERATE_AUDIO_ON_PIPELINE"] = "false"

# Always run tests against a clean per-session SQLite file.
_SESSION_DB = Path("data") / f"test_{uuid.uuid4().hex[:8]}.sqlite3"
os.environ["NEUROPOD_SQLITE_PATH"] = str(_SESSION_DB)


ON_POSTGRES = bool(os.environ.get("NEUROPOD_DATABASE_URL", "").strip())

# Skip marker for assertions that only hold on one backend.
requires_postgres = pytest.mark.skipif(not ON_POSTGRES, reason="requires Postgres + pgvector")


@pytest.fixture(scope="session", autouse=True)
def _cleanup_db():
    if ON_POSTGRES:
        # Start from a clean schema so a re-run does not inherit rows.
        from api.db import cursor, init_schema
        with cursor() as cur:
            cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        init_schema()
        yield
        return

    yield
    for ext in ("", "-shm", "-wal"):
        path = _SESSION_DB.with_name(_SESSION_DB.name + ext)
        if path.exists():
            path.unlink()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


@pytest.fixture
def auth_token(client):
    """Logged-in stub session, returns a bearer token."""
    response = client.post("/auth/stub/login", json={"email": f"pytest-{uuid.uuid4().hex[:6]}@example.com"})
    assert response.status_code == 200, response.text
    return response.json()["token"]


@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}
