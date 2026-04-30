from __future__ import annotations

from fastapi import APIRouter, Depends

from pipeline.provider_status import snapshot
from pipeline.synthesize.tts import TTSProvider

from ..config import Settings, get_settings
from ..dependencies import get_store
from ..storage import DemoStore

router = APIRouter(prefix="/status", tags=["status"])


@router.get("")
def get_status(
    settings: Settings = Depends(get_settings),
    store: DemoStore = Depends(get_store),
) -> dict:
    meta = store.get_meta()
    tts = TTSProvider().provider_name

    if settings.has_anthropic:
        llm = "anthropic"
    elif settings.has_openai:
        llm = "openai"
    else:
        llm = "demo"

    if settings.has_openai and settings.embedder_choice != "demo":
        embedder = "openai"
    else:
        embedder = "demo"

    return {
        "demo_mode": settings.demo_mode,
        "providers": {
            "llm": llm,
            "tts": tts,
            "embedder": embedder,
            "openai": settings.has_openai,
            "anthropic": settings.has_anthropic,
            "elevenlabs": settings.has_elevenlabs,
        },
        "topics": store.get_topics(),
        "last_pipeline_run": meta.get("last_pipeline_run"),
        "scheduler_enabled": settings.enable_scheduler,
        "live_discovery": settings.live_discovery,
        "discovery_window_days": settings.discovery_window_days,
        "provider_calls": snapshot(),
    }
