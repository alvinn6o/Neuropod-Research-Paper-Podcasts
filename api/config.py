from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_str(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@dataclass(frozen=True)
class Settings:
    base_url: str
    frontend_url: str
    default_topics: list[str]
    default_episode_count: int
    enable_scheduler: bool
    scheduler_timezone: str
    discovery_window_days: int
    live_discovery: bool
    auth_mode: str
    audio_backend: str
    generate_audio_on_pipeline: bool
    s3_bucket: str
    database_url: str
    daily_pipeline_limit: int
    daily_ask_limit: int
    cors_origins: list[str]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    cors_raw = os.getenv("NEUROPOD_CORS_ORIGINS", "")
    if cors_raw.strip():
        cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
    else:
        cors_origins = ["http://localhost:3000"]

    return Settings(
        base_url=_env_str("NEUROPOD_BASE_URL", default="http://localhost:8000").rstrip("/"),
        frontend_url=_env_str("NEUROPOD_FRONTEND_URL", default="http://localhost:3000").rstrip("/"),
        default_topics=_env_list(
            "NEUROPOD_DEFAULT_TOPICS",
            ["language models", "retrieval augmented generation", "robotics"],
        ),
        default_episode_count=int(_env_str("NEUROPOD_DEFAULT_EPISODES", default="3")),
        enable_scheduler=_env_flag("ENABLE_SCHEDULER", False),
        scheduler_timezone=_env_str("NEUROPOD_SCHEDULER_TIMEZONE", default="UTC"),
        discovery_window_days=int(os.getenv("NEUROPOD_DISCOVERY_WINDOW_DAYS", "7")),
        live_discovery=_env_flag("NEUROPOD_LIVE_DISCOVERY", False),
        auth_mode=(os.getenv("NEUROPOD_AUTH_MODE") or "stub").lower(),
        audio_backend=(os.getenv("NEUROPOD_AUDIO_BACKEND") or "disk").lower(),
        generate_audio_on_pipeline=_env_flag("NEUROPOD_GENERATE_AUDIO_ON_PIPELINE", False),
        s3_bucket=os.getenv("NEUROPOD_S3_BUCKET", ""),
        database_url=os.getenv("NEUROPOD_DATABASE_URL", ""),
        daily_pipeline_limit=int(os.getenv("NEUROPOD_DAILY_PIPELINE_LIMIT", "20")),
        daily_ask_limit=int(os.getenv("NEUROPOD_DAILY_ASK_LIMIT", "200")),
        cors_origins=cors_origins,
    )
