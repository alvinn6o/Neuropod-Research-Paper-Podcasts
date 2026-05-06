from __future__ import annotations

from functools import lru_cache

from .config import Settings, get_settings
from .db import init_schema


@lru_cache(maxsize=1)
def ensure_db_ready() -> bool:
    init_schema()
    return True


def get_app_settings() -> Settings:
    return get_settings()
