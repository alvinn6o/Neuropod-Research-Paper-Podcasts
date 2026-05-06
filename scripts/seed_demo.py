"""Bootstrap a local demo user + run a single pipeline pass."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import store_db
from api.auth import stub_login
from api.config import get_settings
from api.db import init_schema
from api.pipeline_runner import run_for_user


DEMO_EMAIL = "demo@neuropod.local"


def main() -> None:
    settings = get_settings()
    init_schema()

    _, user = stub_login(DEMO_EMAIL)
    print(f"Demo user ready: {user.email} ({user.feed_slug}) id={user.id}")

    if not store_db.get_topics(user.id):
        store_db.set_topics(user.id, settings.default_topics)
        print(f"Seeded topics: {settings.default_topics}")

    if not store_db.list_episodes(user.id, limit=1):
        result = run_for_user(
            user.id,
            topics=settings.default_topics,
            episode_count=settings.default_episode_count,
            window_days=settings.discovery_window_days,
            require_user_keys=False,  # demo seed: allow env-key fallback
        )
        print(f"Generated {result['result_count']} demo episodes.")
    else:
        print("Episodes already present; skipping pipeline run.")


if __name__ == "__main__":
    main()
