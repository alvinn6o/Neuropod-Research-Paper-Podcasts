"""CLI to run the pipeline against the demo user."""
from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Neuropod pipeline.")
    parser.add_argument("--email", default="demo@neuropod.local")
    parser.add_argument("--topics", default="")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--require-user-keys", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    init_schema()

    _, user = stub_login(args.email)
    topics = [t.strip() for t in args.topics.split(",") if t.strip()] or store_db.get_topics(user.id) or settings.default_topics

    result = run_for_user(
        user.id,
        topics=topics,
        episode_count=args.num_episodes,
        window_days=args.window or settings.discovery_window_days,
        require_user_keys=args.require_user_keys,
    )
    print(f"Generated {result['result_count']} episodes for {user.email}.")
    for ep_id in result["episode_ids"]:
        print(f"  - {ep_id}")


if __name__ == "__main__":
    main()
