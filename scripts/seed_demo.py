from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.config import get_settings
from api.storage import DemoStore


def main() -> None:
    settings = get_settings()
    store = DemoStore(settings.store_path)
    payload = store.reset(settings.default_topics, settings.default_episode_count)
    print(f"Seeded {len(payload['episodes'])} demo episodes into {settings.store_path}")


if __name__ == "__main__":
    main()
