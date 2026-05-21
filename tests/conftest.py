"""Shared pytest fixtures."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

# Force stub-mode auth and disable user-key requirement so tests can exercise
# the pipeline with the demo fallback (no API calls).
os.environ.setdefault("NEUROPOD_AUTH_MODE", "stub")
os.environ.setdefault("NEUROPOD_REQUIRE_USER_KEYS", "false")
os.environ.setdefault("NEUROPOD_DATABASE_URL", "")  # force SQLite shim

# Always run tests against a clean per-session SQLite file.
_SESSION_DB = Path("data") / f"test_{uuid.uuid4().hex[:8]}.sqlite3"
os.environ["NEUROPOD_SQLITE_PATH"] = str(_SESSION_DB)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_db():
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
