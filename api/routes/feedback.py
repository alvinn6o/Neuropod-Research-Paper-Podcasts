from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from .. import store_db
from ..auth import AuthUser, CurrentUser
from ..models import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
def log_feedback(payload: FeedbackRequest, user: AuthUser = CurrentUser) -> FeedbackResponse:
    try:
        episode_id = uuid.UUID(payload.episode_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid episode id")

    episode = store_db.get_episode(user.id, episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    total = store_db.add_feedback(
        user.id, episode_id, payload.event_type, payload.position_secs
    )
    return FeedbackResponse(total_events=total)
