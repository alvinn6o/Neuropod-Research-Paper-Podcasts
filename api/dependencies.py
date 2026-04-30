from __future__ import annotations

from functools import lru_cache

from .config import Settings, get_settings
from .storage import DemoStore


@lru_cache(maxsize=1)
def get_store() -> DemoStore:
    settings = get_settings()
    store = DemoStore(settings.store_path)
    store.ensure_seeded(
        settings.default_topics,
        settings.default_episode_count,
        settings.discovery_window_days,
    )
    return store


def get_app_settings() -> Settings:
    return get_settings()
