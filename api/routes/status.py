from __future__ import annotations

from fastapi import APIRouter

from pipeline.provider_status import snapshot

from .. import keys_repo, store_db
from ..auth import AuthUser, OptionalUser
from ..config import get_settings

router = APIRouter(prefix="/status", tags=["status"])


@router.get("")
def get_status(user: AuthUser | None = OptionalUser) -> dict:
    settings = get_settings()
    out: dict = {
        "auth_mode": settings.auth_mode,
        "live_discovery": settings.live_discovery,
        "discovery_window_days": settings.discovery_window_days,
        "audio_backend": settings.audio_backend,
        "scheduler_enabled": settings.enable_scheduler,
        "require_user_keys": settings.require_user_keys,
        "provider_calls": snapshot(),
    }

    if not user:
        out["authenticated"] = False
        return out

    masked = keys_repo.list_masked(user.id)
    out["authenticated"] = True
    out["user"] = {"feed_slug": user.feed_slug, "email": user.email}
    out["topics"] = store_db.get_topics(user.id)
    out["keys"] = masked
    if "bedrock" in masked:
        llm = "bedrock"
    elif "anthropic" in masked:
        llm = "anthropic"
    elif "openai" in masked:
        llm = "openai"
    else:
        llm = "demo"
    out["providers"] = {
        "llm": llm,
        "tts": "elevenlabs" if "elevenlabs" in masked else ("openai" if "openai" in masked else "demo"),
        "embedder": "openai" if "openai" in masked else "demo",
    }
    out["last_job"] = store_db.latest_job(user.id)
    return out
