from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_store
from ..models import FeedbackRequest, FeedbackResponse
from ..storage import DemoStore

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
def log_feedback(
    payload: FeedbackRequest,
    store: DemoStore = Depends(get_store),
) -> FeedbackResponse:
    episode = store.get_episode(payload.episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")

    total_events = store.add_feedback(
        {
            "id": str(uuid4()),
            "episode_id": payload.episode_id,
            "event_type": payload.event_type,
            "position_secs": payload.position_secs,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return FeedbackResponse(total_events=total_events)
