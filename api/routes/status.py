from __future__ import annotations

import os

from fastapi import APIRouter

from pipeline.provider_status import snapshot

from .. import budget, store_db
from ..auth import AuthUser, OptionalUser
from ..config import get_settings

router = APIRouter(prefix="/status", tags=["status"])


def _detect_providers() -> dict[str, str]:
    """Which provider each stage will use, based on operator env config."""
    has_openai = bool(os.getenv("OPENAI_API_KEY", "").strip())
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
    has_elevenlabs = bool(os.getenv("ELEVENLABS_API_KEY", "").strip())
    has_bedrock = os.getenv("NEUROPOD_BEDROCK_OPERATOR", "").lower() in {"1", "true", "yes"}

    if has_anthropic:
        llm = "anthropic"
    elif has_bedrock:
        llm = "bedrock"
    elif has_openai:
        llm = "openai"
    else:
        llm = "demo"

    if has_elevenlabs:
        tts = "elevenlabs"
    elif has_openai:
        tts = "openai"
    else:
        tts = "demo"

    embedder = "openai" if has_openai else "demo"
    return {"llm": llm, "tts": tts, "embedder": embedder}


@router.get("")
def get_status(user: AuthUser | None = OptionalUser) -> dict:
    settings = get_settings()
    out: dict = {
        "auth_mode": settings.auth_mode,
        "live_discovery": settings.live_discovery,
        "discovery_window_days": settings.discovery_window_days,
        "audio_backend": settings.audio_backend,
        "generate_audio_on_pipeline": settings.generate_audio_on_pipeline,
        "scheduler_enabled": settings.enable_scheduler,
        "providers": _detect_providers(),
        "provider_calls": snapshot(),
        # Spend against the caps. Exposed unauthenticated on purpose: it is the
        # honest answer to "is this demo going to cost the author money", and it
        # shows a visitor why they may be seeing template scripts.
        "budget": budget.budget_state(),
    }

    if not user:
        out["authenticated"] = False
        return out

    out["authenticated"] = True
    # Token/cost rollup by model for the current month.
    out["spend"] = store_db.spend_summary()
    out["user"] = {"feed_slug": user.feed_slug, "email": user.email}
    out["topics"] = store_db.get_topics(user.id)
    out["categories"] = store_db.get_categories(user.id)
    out["last_job"] = store_db.latest_job(user.id)
    return out
