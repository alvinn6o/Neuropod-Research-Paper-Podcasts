"""Postgres connection pool + low-level helpers.

Reads NEUROPOD_DATABASE_URL. If unset, falls back to a SQLite-on-disk
emulation via the local-only `LocalShim` so the app still boots in pure-demo
mode without needing Postgres installed.

Production: always set NEUROPOD_DATABASE_URL=postgres://...
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger("neuropod.db")

_DATABASE_URL = os.getenv("NEUROPOD_DATABASE_URL", "").strip()
_pool = None
_pool_lock = threading.Lock()


def _have_postgres() -> bool:
    return bool(_DATABASE_URL)


def _get_pool():
    global _pool
    if not _have_postgres():
        return None
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        from psycopg_pool import ConnectionPool
        _pool = ConnectionPool(_DATABASE_URL, min_size=1, max_size=8, open=True)
        return _pool


@contextmanager
def cursor() -> Iterator[Any]:
    """Yield a cursor from the Postgres pool, or a SQLite shim cursor."""
    if _have_postgres():
        pool = _get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                yield cur
            conn.commit()
        return

    # ---- SQLite shim path ----
    shim = _LocalShim.instance()
    with shim.cursor() as cur:
        yield cur
    shim.commit()


def init_schema() -> None:
    """Apply schema.sql on the configured database."""
    schema_path = Path(__file__).resolve().parents[1] / "db" / "schema.sql"
    if not schema_path.exists():
        return

    if _have_postgres():
        statements = schema_path.read_text()
        with cursor() as cur:
            cur.execute(statements)
        return

    _LocalShim.instance().init_schema()


# ----------------------------------------------------------------------------
# User helpers (called by auth)
# ----------------------------------------------------------------------------

def upsert_user_by_email(
    *, email: str, feed_slug: str, fixed_user_id: Optional[uuid.UUID] = None
) -> tuple[uuid.UUID, str]:
    """Insert or fetch the user row. Returns (user_id, assigned_feed_slug)."""
    email = email.strip().lower()
    with cursor() as cur:
        cur.execute("SELECT id, feed_slug FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        if row:
            return _as_uuid(row[0]), row[1]

        user_id = fixed_user_id or uuid.uuid4()
        # ensure unique slug
        candidate = feed_slug
        suffix = 0
        while True:
            cur.execute("SELECT 1 FROM users WHERE feed_slug = %s", (candidate,))
            if not cur.fetchone():
                break
            suffix += 1
            candidate = f"{feed_slug}-{suffix}"

        cur.execute(
            "INSERT INTO users (id, email, feed_slug) VALUES (%s, %s, %s)",
            (str(user_id), email, candidate),
        )
        return user_id, candidate


def get_user_by_slug(slug: str) -> Optional[dict[str, Any]]:
    with cursor() as cur:
        cur.execute(
            "SELECT id, email, feed_slug, display_name FROM users WHERE feed_slug = %s",
            (slug,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"id": _as_uuid(row[0]), "email": row[1], "feed_slug": row[2], "display_name": row[3]}


# ----------------------------------------------------------------------------
# SQLite local shim - keeps the dev story painless.
#   * Same SQL surface area we use in the Postgres path
#   * Transparent JSON <-> list for array columns
#   * Vector columns stored as JSON text (no similarity search - falls back to sparse)
# ----------------------------------------------------------------------------

_PG_TO_SQLITE = [
    ("TIMESTAMPTZ", "TEXT"),
    ("TIMESTAMP", "TEXT"),
    ("BYTEA", "BLOB"),
    ("DOUBLE PRECISION", "REAL"),
    ("DATE", "TEXT"),
    ("TEXT[]", "TEXT"),
    ("INT[]", "TEXT"),
    ("vector(1536)", "TEXT"),
    ("UUID", "TEXT"),
    ("UUID PRIMARY KEY", "TEXT PRIMARY KEY"),
    ("DEFAULT NOW()", "DEFAULT CURRENT_TIMESTAMP"),
    ("DEFAULT gen_random_uuid()", ""),
]


def _translate_pg_to_sqlite(statements: str) -> str:
    text = statements
    for needle, replacement in _PG_TO_SQLITE:
        text = text.replace(needle, replacement)
    # strip extension lines
    keep = []
    skip_next = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("create extension"):
            continue
        if "USING hnsw" in line:
            # convert HNSW vector index to a no-op btree on paper_id (already covered)
            continue
        keep.append(line)
    return "\n".join(keep)


class _LocalShim:
    _singleton: Optional["_LocalShim"] = None
    _singleton_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "_LocalShim":
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    def __init__(self) -> None:
        path_env = os.getenv("NEUROPOD_SQLITE_PATH", "data/neuropod.sqlite3")
        self._path = Path(path_env)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._lock = threading.Lock()
        self._schema_applied = False

    def init_schema(self) -> None:
        if self._schema_applied:
            return
        schema_path = Path(__file__).resolve().parents[1] / "db" / "schema.sql"
        sql = _translate_pg_to_sqlite(schema_path.read_text())
        with self._lock:
            self._conn.executescript(sql)
        self._schema_applied = True

    @contextmanager
    def cursor(self) -> Iterator[Any]:
        if not self._schema_applied:
            self.init_schema()
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield _CursorAdapter(cur)
            finally:
                cur.close()

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()


class _CursorAdapter:
    """Translates psycopg-style %s placeholders to sqlite ? and serialises arrays as JSON."""

    def __init__(self, cur: sqlite3.Cursor) -> None:
        self._cur = cur

    def execute(self, sql: str, params: Any = None):
        sql_translated = sql.replace("%s", "?")
        if params is None:
            self._cur.execute(sql_translated)
        else:
            self._cur.execute(sql_translated, _normalize_params(params))
        return self

    def executemany(self, sql: str, seq_of_params):
        sql_translated = sql.replace("%s", "?")
        self._cur.executemany(sql_translated, [_normalize_params(p) for p in seq_of_params])
        return self

    def fetchone(self):
        row = self._cur.fetchone()
        return _decode_row(row) if row else row

    def fetchall(self):
        return [_decode_row(row) for row in self._cur.fetchall()]

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount


def _normalize_params(params: Any) -> tuple:
    if isinstance(params, dict):
        return params  # type: ignore[return-value]
    out = []
    for p in params:
        if isinstance(p, (list, tuple)) and not isinstance(p, str):
            out.append(json.dumps(list(p)))
        elif isinstance(p, dict):
            out.append(json.dumps(p))
        elif isinstance(p, datetime):
            out.append(p.astimezone(timezone.utc).isoformat())
        elif isinstance(p, date):
            out.append(p.isoformat())
        elif isinstance(p, uuid.UUID):
            out.append(str(p))
        elif isinstance(p, (bytes, bytearray, memoryview)):
            out.append(bytes(p))
        else:
            out.append(p)
    return tuple(out)


def _decode_row(row: tuple) -> tuple:
    out = []
    for value in row:
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            try:
                out.append(json.loads(value))
                continue
            except Exception:
                pass
        out.append(value)
    return tuple(out)


def _as_uuid(value: Any) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))
