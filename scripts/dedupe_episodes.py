"""One-shot cleanup: remove duplicate episodes per (user_id, paper_id).

Keeps the oldest episode for each pair, deletes the rest. Useful after the
dedup logic was added — clears any old duplicates created by earlier
pipeline runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import store_db
from api.db import init_schema


def main() -> None:
    init_schema()
    deleted = store_db.delete_duplicate_episodes()
    print(f"Deleted {deleted} duplicate episode{'s' if deleted != 1 else ''}.")


if __name__ == "__main__":
    main()
