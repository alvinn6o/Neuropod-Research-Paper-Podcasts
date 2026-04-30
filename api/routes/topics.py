from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_store
from ..models import TopicResponse, TopicUpdateRequest
from ..storage import DemoStore

router = APIRouter(prefix="/topics", tags=["topics"])


@router.get("", response_model=TopicResponse)
def get_topics(store: DemoStore = Depends(get_store)) -> TopicResponse:
    return TopicResponse(topics=store.get_topics())


@router.post("", response_model=TopicResponse)
def update_topics(
    payload: TopicUpdateRequest,
    store: DemoStore = Depends(get_store),
) -> TopicResponse:
    return TopicResponse(topics=store.set_topics(payload.topics))
