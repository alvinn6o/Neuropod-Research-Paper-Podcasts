from __future__ import annotations

from fastapi import APIRouter

from .. import store_db
from ..auth import AuthUser, CurrentUser
from ..models import TopicResponse, TopicUpdateRequest

router = APIRouter(prefix="/topics", tags=["topics"])


@router.get("", response_model=TopicResponse)
def get_topics(user: AuthUser = CurrentUser) -> TopicResponse:
    return TopicResponse(topics=store_db.get_topics(user.id))


@router.post("", response_model=TopicResponse)
def update_topics(payload: TopicUpdateRequest, user: AuthUser = CurrentUser) -> TopicResponse:
    return TopicResponse(topics=store_db.set_topics(user.id, payload.topics))
