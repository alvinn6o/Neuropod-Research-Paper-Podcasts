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
    demo_mode: bool
    store_path: Path
    base_url: str
    frontend_url: str
    default_topics: list[str]
    default_episode_count: int
    enable_scheduler: bool
    scheduler_timezone: str
    openai_api_key: str
    anthropic_api_key: str
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    tts_provider: str
    llm_provider: str
    embedder_choice: str
    discovery_window_days: int
    live_discovery: bool

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_elevenlabs(self) -> bool:
        return bool(self.elevenlabs_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = ROOT
    raw_store = _env_str("NEUROPOD_STORE_PATH", "PAPERPOD_STORE_PATH", default=str(root / "data" / "demo_store.json"))
    store_path = Path(raw_store)
    if not store_path.is_absolute():
        store_path = (root / store_path).resolve()

    return Settings(
        demo_mode=_env_flag("NEUROPOD_DEMO_MODE", _env_flag("PAPERPOD_DEMO_MODE", True)),
        store_path=store_path,
        base_url=_env_str("NEUROPOD_BASE_URL", "PAPERPOD_BASE_URL", default="http://localhost:8000").rstrip("/"),
        frontend_url=_env_str("NEUROPOD_FRONTEND_URL", "PAPERPOD_FRONTEND_URL", default="http://localhost:3000").rstrip("/"),
        default_topics=_env_list(
            "NEUROPOD_DEFAULT_TOPICS",
            _env_list("PAPERPOD_DEFAULT_TOPICS", ["language models", "retrieval augmented generation", "robotics"]),
        ),
        default_episode_count=int(_env_str("NEUROPOD_DEFAULT_EPISODES", "PAPERPOD_DEFAULT_EPISODES", default="3")),
        enable_scheduler=_env_flag("ENABLE_SCHEDULER", False),
        scheduler_timezone=_env_str("NEUROPOD_SCHEDULER_TIMEZONE", "PAPERPOD_SCHEDULER_TIMEZONE", default="UTC"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        tts_provider=os.getenv("NEUROPOD_TTS_PROVIDER", "auto"),
        llm_provider=os.getenv("NEUROPOD_LLM_PROVIDER", "auto"),
        embedder_choice=os.getenv("NEUROPOD_EMBEDDER", "auto"),
        discovery_window_days=int(os.getenv("NEUROPOD_DISCOVERY_WINDOW_DAYS", "7")),
        live_discovery=_env_flag("NEUROPOD_LIVE_DISCOVERY", False),
    )
