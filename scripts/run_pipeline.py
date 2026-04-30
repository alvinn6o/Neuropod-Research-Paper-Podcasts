from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.config import get_settings
from api.storage import DemoStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Neuropod pipeline.")
    parser.add_argument("--topics", default="", help="Comma-separated topics to prioritize.")
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of episodes to generate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    store = DemoStore(settings.store_path)
    topics = [item.strip() for item in args.topics.split(",") if item.strip()] or settings.default_topics
    payload = store.run_pipeline(topics=topics, episode_count=args.num_episodes)

    print("Pipeline complete.")
    print(f"Generated at: {payload['generated_at']}")
    for episode in payload["episodes"]:
        print(f"- {episode['title']} ({episode['duration_secs']}s, {episode['qa_status']})")


if __name__ == "__main__":
    main()
