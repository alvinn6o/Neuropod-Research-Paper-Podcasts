"""DB-backed multi-user store (replaces the legacy DemoStore JSON file).

All read/write paths are scoped to a `user_id`. Papers and chunks are shared
across users (de-duplicated on arxiv_id). Episodes, topics, feedback,
pipeline_jobs, rate_limits are per-user.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timezone
from typing import Any, Optional

from .db import cursor

logger = logging.getLogger("neuropod.store")


# ---------- Topics ----------

def get_topics(user_id: uuid.UUID) -> list[str]:
    with cursor() as cur:
        cur.execute(
            "SELECT topic FROM user_topics WHERE user_id = %s ORDER BY position, topic",
            (str(user_id),),
        )
        return [row[0] for row in cur.fetchall()]


def set_topics(user_id: uuid.UUID, topics: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for topic in topics:
        t = (topic or "").strip()
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        cleaned.append(t)

    with cursor() as cur:
        cur.execute("DELETE FROM user_topics WHERE user_id = %s", (str(user_id),))
        for index, topic in enumerate(cleaned):
            cur.execute(
                "INSERT INTO user_topics (user_id, topic, position) VALUES (%s, %s, %s)",
                (str(user_id), topic, index),
            )
    return cleaned


def get_categories(user_id: uuid.UUID) -> list[str]:
    with cursor() as cur:
        cur.execute(
            "SELECT category FROM user_categories WHERE user_id = %s ORDER BY position, category",
            (str(user_id),),
        )
        return [row[0] for row in cur.fetchall()]


def set_categories(user_id: uuid.UUID, categories: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for cat in categories:
        c = (cat or "").strip()
        if not c or c.lower() in seen:
            continue
        # Light validation: arXiv categories look like 'cs.AI', 'q-bio.QM', 'hep-th'
        if len(c) > 32 or any(ch in c for ch in [" ", ":", "(", ")", '"']):
            continue
        seen.add(c.lower())
        cleaned.append(c)
    with cursor() as cur:
        cur.execute("DELETE FROM user_categories WHERE user_id = %s", (str(user_id),))
        for index, category in enumerate(cleaned):
            cur.execute(
                "INSERT INTO user_categories (user_id, category, position) VALUES (%s, %s, %s)",
                (str(user_id), category, index),
            )
    return cleaned


# ---------- Papers + chunks ----------

def upsert_paper(record: dict[str, Any]) -> uuid.UUID:
    """Insert or fetch a paper by arxiv_id. Returns the paper's UUID."""
    with cursor() as cur:
        cur.execute("SELECT id FROM papers WHERE arxiv_id = %s", (record["arxiv_id"],))
        row = cur.fetchone()
        if row:
            return _as_uuid(row[0])

        paper_id = uuid.uuid4()
        cur.execute(
            """
            INSERT INTO papers
              (id, arxiv_id, title, authors, abstract, categories,
               published_at, pdf_url, citation_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(paper_id),
                record["arxiv_id"],
                record["title"],
                record.get("authors", []),
                record["abstract"],
                record.get("categories", []),
                record["published_at"],
                record.get("pdf_url"),
                int(record.get("citation_count") or 0),
            ),
        )
        return paper_id


def replace_chunks_for_paper(paper_id: uuid.UUID, chunks: list[dict[str, Any]]) -> None:
    with cursor() as cur:
        cur.execute("DELETE FROM paper_chunks WHERE paper_id = %s", (str(paper_id),))
        for chunk in chunks:
            cur.execute(
                """
                INSERT INTO paper_chunks
                  (id, paper_id, section, chunk_index, content, token_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    chunk.get("id") or str(uuid.uuid4()),
                    str(paper_id),
                    chunk["section"],
                    int(chunk.get("chunk_index", 0)),
                    chunk["content"],
                    int(chunk.get("token_count", 0)),
                    json.dumps(chunk.get("embedding") or []),
                ),
            )


def get_chunks_for_paper(paper_id: uuid.UUID) -> list[dict[str, Any]]:
    with cursor() as cur:
        cur.execute(
            """
            SELECT id, paper_id, section, chunk_index, content, token_count, embedding
            FROM paper_chunks WHERE paper_id = %s
            ORDER BY chunk_index
            """,
            (str(paper_id),),
        )
        rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({
            "id": str(row[0]),
            "paper_id": str(row[1]),
            "section": row[2],
            "chunk_index": row[3],
            "content": row[4],
            "token_count": row[5],
            "embedding": _decode_embedding(row[6]),
        })
    return out


def get_paper(paper_id: uuid.UUID) -> Optional[dict[str, Any]]:
    with cursor() as cur:
        cur.execute(
            """
            SELECT id, arxiv_id, title, authors, abstract, categories,
                   published_at, pdf_url, citation_count
            FROM papers WHERE id = %s
            """,
            (str(paper_id),),
        )
        row = cur.fetchone()
        if not row:
            return None
    return _paper_row_to_dict(row)


# ---------- Episodes ----------

def find_episode_for_paper(user_id: uuid.UUID, paper_id: uuid.UUID) -> Optional[uuid.UUID]:
    """Return existing episode_id for this (user, paper) pair, or None.

    Used to dedupe: if a user already has an episode for a paper, the next
    pipeline run shouldn't create a second row for it.
    """
    with cursor() as cur:
        cur.execute(
            "SELECT id FROM episodes WHERE user_id = %s AND paper_id = %s LIMIT 1",
            (str(user_id), str(paper_id)),
        )
        row = cur.fetchone()
    return _as_uuid(row[0]) if row else None


def delete_duplicate_episodes() -> int:
    """One-shot cleanup: for each (user_id, paper_id) keep the oldest episode,
    delete the rest. Returns the number of deleted rows."""
    with cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, paper_id, created_at
            FROM episodes ORDER BY user_id, paper_id, created_at
            """
        )
        rows = cur.fetchall()
        keep: set[tuple[str, str]] = set()
        to_delete: list[str] = []
        for row in rows:
            key = (str(row[1]), str(row[2]))
            if key in keep:
                to_delete.append(str(row[0]))
            else:
                keep.add(key)
        for episode_id in to_delete:
            cur.execute("DELETE FROM episodes WHERE id = %s", (episode_id,))
    return len(to_delete)


def insert_episode(user_id: uuid.UUID, paper_id: uuid.UUID, episode: dict[str, Any]) -> uuid.UUID:
    episode_id = uuid.uuid4()
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO episodes
              (id, user_id, paper_id, title, description, topic, score,
               script, qa_status, qa_notes, duration_secs,
               llm_provider, tts_provider, audio_key, audio_mime)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(episode_id),
                str(user_id),
                str(paper_id),
                episode["title"],
                episode.get("description", ""),
                episode.get("topic", "general AI"),
                float(episode.get("score") or 0.0),
                episode.get("script", ""),
                episode.get("qa_status", "verified"),
                episode.get("qa_notes"),
                int(episode.get("duration_secs") or 0),
                episode.get("llm_provider", "demo"),
                episode.get("tts_provider", "none"),
                episode.get("audio_key"),
                episode.get("audio_mime", "audio/wav"),
            ),
        )
    return episode_id


def list_episodes(
    user_id: uuid.UUID,
    *,
    topic: Optional[str] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    sql = """
        SELECT e.id, e.user_id, e.paper_id, e.title, e.description, e.topic,
               e.score, e.script, e.qa_status, e.qa_notes, e.duration_secs,
               e.llm_provider, e.tts_provider, e.audio_key, e.audio_mime,
               e.created_at,
               p.id, p.arxiv_id, p.title, p.authors, p.abstract, p.categories,
               p.published_at, p.pdf_url, p.citation_count
        FROM episodes e
        JOIN papers p ON p.id = e.paper_id
        WHERE e.user_id = %s
    """
    params: list[Any] = [str(user_id)]
    if topic:
        sql += " AND LOWER(e.topic) LIKE %s"
        params.append(f"%{topic.lower()}%")
    sql += " ORDER BY e.created_at DESC LIMIT %s"
    params.append(int(limit))

    with cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    return [_episode_row_to_dict(row) for row in rows]


def get_episode(user_id: uuid.UUID, episode_id: uuid.UUID) -> Optional[dict[str, Any]]:
    with cursor() as cur:
        cur.execute(
            """
            SELECT e.id, e.user_id, e.paper_id, e.title, e.description, e.topic,
                   e.score, e.script, e.qa_status, e.qa_notes, e.duration_secs,
                   e.llm_provider, e.tts_provider, e.audio_key, e.audio_mime,
                   e.created_at,
                   p.id, p.arxiv_id, p.title, p.authors, p.abstract, p.categories,
                   p.published_at, p.pdf_url, p.citation_count
            FROM episodes e
            JOIN papers p ON p.id = e.paper_id
            WHERE e.user_id = %s AND e.id = %s
            """,
            (str(user_id), str(episode_id)),
        )
        row = cur.fetchone()
    return _episode_row_to_dict(row) if row else None


def get_episode_by_slug(slug: str, episode_id: uuid.UUID) -> Optional[dict[str, Any]]:
    with cursor() as cur:
        cur.execute(
            """
            SELECT id FROM users WHERE feed_slug = %s
            """,
            (slug,),
        )
        u = cur.fetchone()
        if not u:
            return None
    return get_episode(_as_uuid(u[0]), episode_id)


def update_episode_audio(
    episode_id: uuid.UUID, *, audio_key: str, audio_mime: str, tts_provider: str
) -> None:
    with cursor() as cur:
        cur.execute(
            """
            UPDATE episodes
            SET audio_key = %s, audio_mime = %s, tts_provider = %s
            WHERE id = %s
            """,
            (audio_key, audio_mime, tts_provider, str(episode_id)),
        )


# ---------- Feedback ----------

def add_feedback(
    user_id: uuid.UUID, episode_id: uuid.UUID, event_type: str, position_secs: int
) -> int:
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO feedback_events (user_id, episode_id, event_type, position_secs)
            VALUES (%s, %s, %s, %s)
            """,
            (str(user_id), str(episode_id), event_type, int(position_secs)),
        )
        cur.execute(
            "SELECT COUNT(*) FROM feedback_events WHERE user_id = %s",
            (str(user_id),),
        )
        return int(cur.fetchone()[0])


def feedback_for_user(user_id: uuid.UUID) -> list[dict[str, Any]]:
    with cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, episode_id, event_type, position_secs, created_at
            FROM feedback_events WHERE user_id = %s ORDER BY created_at DESC LIMIT 500
            """,
            (str(user_id),),
        )
        rows = cur.fetchall()
    return [
        {
            "id": str(r[0]),
            "user_id": str(r[1]),
            "episode_id": str(r[2]),
            "event_type": r[3],
            "position_secs": r[4],
            "created_at": _iso(r[5]),
        }
        for r in rows
    ]


# ---------- Pipeline jobs ----------

def create_job(
    user_id: uuid.UUID,
    *,
    window_days: int,
    topics: list[str],
    episode_count: int,
) -> uuid.UUID:
    job_id = uuid.uuid4()
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline_jobs
              (id, user_id, status, window_days, topics, episode_count)
            VALUES (%s, %s, 'queued', %s, %s, %s)
            """,
            (str(job_id), str(user_id), int(window_days), topics, int(episode_count)),
        )
    return job_id


def latest_job(user_id: uuid.UUID) -> Optional[dict[str, Any]]:
    with cursor() as cur:
        cur.execute(
            """
            SELECT id, status, window_days, topics, episode_count,
                   created_at, started_at, finished_at, error, result_count, skipped_count
            FROM pipeline_jobs
            WHERE user_id = %s ORDER BY created_at DESC LIMIT 1
            """,
            (str(user_id),),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": str(row[0]),
        "status": row[1],
        "window_days": row[2],
        "topics": row[3],
        "episode_count": row[4],
        "created_at": _iso(row[5]),
        "started_at": _iso(row[6]),
        "finished_at": _iso(row[7]),
        "error": row[8],
        "result_count": row[9],
        "skipped_count": row[10] if len(row) > 10 else 0,
    }


def get_job(job_id: uuid.UUID) -> Optional[dict[str, Any]]:
    with cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, status, window_days, topics, episode_count,
                   created_at, started_at, finished_at, error, result_count, skipped_count
            FROM pipeline_jobs WHERE id = %s
            """,
            (str(job_id),),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": str(row[0]),
        "user_id": str(row[1]),
        "status": row[2],
        "window_days": row[3],
        "topics": row[4],
        "episode_count": row[5],
        "created_at": _iso(row[6]),
        "started_at": _iso(row[7]),
        "finished_at": _iso(row[8]),
        "error": row[9],
        "result_count": row[10],
        "skipped_count": row[11] if len(row) > 11 else 0,
    }


def update_job(
    job_id: uuid.UUID,
    *,
    status: Optional[str] = None,
    error: Optional[str] = None,
    started_at: bool = False,
    finished_at: bool = False,
    result_count: Optional[int] = None,
    skipped_count: Optional[int] = None,
) -> None:
    fragments: list[str] = []
    params: list[Any] = []
    if status is not None:
        fragments.append("status = %s")
        params.append(status)
    if error is not None:
        fragments.append("error = %s")
        params.append(error)
    if started_at:
        fragments.append("started_at = CURRENT_TIMESTAMP")
    if finished_at:
        fragments.append("finished_at = CURRENT_TIMESTAMP")
    if result_count is not None:
        fragments.append("result_count = %s")
        params.append(int(result_count))
    if skipped_count is not None:
        fragments.append("skipped_count = %s")
        params.append(int(skipped_count))
    if not fragments:
        return
    params.append(str(job_id))
    with cursor() as cur:
        cur.execute(
            f"UPDATE pipeline_jobs SET {', '.join(fragments)} WHERE id = %s",
            tuple(params),
        )


# ---------- Rate limit ----------

def increment_rate_limit(
    user_id: uuid.UUID, bucket: str, *, daily_max: int
) -> tuple[int, bool]:
    """Atomically increment and check the per-user daily counter.

    Uses a single UPSERT that returns the post-increment count, so two
    concurrent requests can't both see "count=9, ok to add" and both succeed
    past a 10/day cap. Works on both Postgres and the SQLite shim because
    both support INSERT ... ON CONFLICT DO UPDATE ... RETURNING.

    Returns (count_after_increment, was_under_or_at_limit).
    """
    today = date.today().isoformat()
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO rate_limits (user_id, bucket, day, count)
            VALUES (%s, %s, %s, 1)
            ON CONFLICT(user_id, bucket, day)
            DO UPDATE SET count = rate_limits.count + 1
            RETURNING count
            """,
            (str(user_id), bucket, today),
        )
        row = cur.fetchone()
        new_count = int(row[0]) if row else 1

    if new_count > daily_max:
        # Already incremented; consumer logs and rejects. We don't decrement
        # back — keeping the over-cap value lets you audit "this user
        # repeatedly hammered the API" instead of hiding the attempt.
        return new_count, False
    return new_count, True


# ---------- Helpers ----------

def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _decode_embedding(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, str):
        try:
            return list(json.loads(raw))
        except Exception:
            return []
    return []


def _paper_row_to_dict(row: tuple) -> dict[str, Any]:
    return {
        "id": str(row[0]),
        "arxiv_id": row[1],
        "title": row[2],
        "authors": row[3] or [],
        "abstract": row[4] or "",
        "categories": row[5] or [],
        "published_at": _iso(row[6]) or "",
        "pdf_url": row[7] or "",
        "citation_count": int(row[8] or 0),
    }


def _episode_row_to_dict(row: tuple) -> dict[str, Any]:
    paper = {
        "id": str(row[16]),
        "arxiv_id": row[17],
        "title": row[18],
        "authors": row[19] or [],
        "abstract": row[20] or "",
        "categories": row[21] or [],
        "published_at": _iso(row[22]) or "",
        "pdf_url": row[23] or "",
        "citation_count": int(row[24] or 0),
        "score": 0.0,
    }
    return {
        "id": str(row[0]),
        "user_id": str(row[1]),
        "paper_id": str(row[2]),
        "title": row[3],
        "description": row[4] or "",
        "topic": row[5],
        "score": float(row[6] or 0.0),
        "script": row[7] or "",
        "qa_status": row[8],
        "qa_notes": row[9],
        "duration_secs": int(row[10] or 0),
        "llm_provider": row[11] or "demo",
        "tts_provider": row[12] or "none",
        "audio_key": row[13],
        "audio_mime": row[14] or "audio/wav",
        "created_at": _iso(row[15]),
        "paper": paper,
    }


def _as_uuid(value: Any) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))
