"""Seed a working demo: one user, real topics, episodes already generated.

Why seed rather than let the recording trigger a pipeline run: a live run takes
30-90 seconds, needs network, and can fail on an arXiv timeout. None of that is
what the demo is about. This produces the same rows the pipeline would, using
the offline demo catalog and the deterministic hash embedder, so the UI has
content the moment it loads.

Runs with no API keys and no network. If keys ARE configured it will use them,
which is worth knowing before recording: scripts then cost real money and take
longer, but read far better.

    python scripts/demo.py            # seed
    python scripts/demo.py --reset    # wipe and reseed
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEMO_EMAIL = "demo@neuropod.local"
DEMO_TOPICS = ["language models", "retrieval augmented generation"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete existing demo data first")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    os.environ.setdefault("NEUROPOD_AUTH_MODE", "stub")
    os.environ.setdefault("NEUROPOD_LIVE_DISCOVERY", "false")

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    from api import store_db
    from api.db import cursor, init_schema
    from api.pipeline_runner import run_for_user

    init_schema()

    # Say which mode this is up front. The offline path writes ~170-word
    # template scripts; with a provider key it writes real 800-word narration.
    # Discovering that difference mid-recording is not the moment to find out.
    has_key = any(os.getenv(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"))
    if has_key:
        print("MODE: provider keys found — scripts will be model-generated "
              "(~800 words, costs a few cents)")
    else:
        print("MODE: no provider keys — scripts come from the offline template "
              "(~170 words, free)")
        print("      Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env for real scripts.")

    with cursor() as cur:
        cur.execute("SELECT id FROM users WHERE email = %s", (DEMO_EMAIL,))
        row = cur.fetchone()

    if row and args.reset:
        with cursor() as cur:
            cur.execute("DELETE FROM users WHERE email = %s", (DEMO_EMAIL,))
        row = None
        print("reset: removed the previous demo user and its episodes")

    if row:
        user_id = uuid.UUID(str(row[0]))
        print(f"demo user exists ({DEMO_EMAIL})")
    else:
        user_id = uuid.uuid4()
        with cursor() as cur:
            cur.execute(
                "INSERT INTO users (id, email, feed_slug) VALUES (%s, %s, %s)",
                (str(user_id), DEMO_EMAIL, "demo"),
            )
        print(f"created demo user {DEMO_EMAIL}")

    store_db.set_topics(user_id, DEMO_TOPICS)

    existing = store_db.list_episodes(user_id, limit=50)
    if len(existing) >= args.episodes:
        print(f"{len(existing)} episodes already present — nothing to generate")
    else:
        print(f"generating {args.episodes} episodes (offline catalog, no API calls needed)...")
        result = run_for_user(
            user_id, topics=DEMO_TOPICS,
            episode_count=args.episodes, window_days=7,
        )
        print(f"  {result['result_count']} episodes, "
              f"{result['llm_calls']} llm calls, ${result['cost_usd']}")

    episodes = store_db.list_episodes(user_id, limit=50)
    print(f"\n{len(episodes)} episodes ready:")
    for e in episodes[:5]:
        words = len((e.get("script") or "").split())
        print(f"  [{e.get('qa_status', '?'):<8}] {words:>4}w  {e['title'][:56]}")

    print(f"\n  Log in at http://localhost:3000/login with: {DEMO_EMAIL}")
    print("  (stub auth — any email works, no password)")


if __name__ == "__main__":
    main()
