from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import store_db
from ..auth import AuthUser, CurrentUser
from ..models import TopicResponse, TopicUpdateRequest

router = APIRouter(tags=["preferences"])


class CategoryListResponse(BaseModel):
    categories: list[str]


class CategoryUpdateRequest(BaseModel):
    categories: list[str] = Field(default_factory=list)


@router.get("/topics", response_model=TopicResponse)
def get_topics(user: AuthUser = CurrentUser) -> TopicResponse:
    return TopicResponse(topics=store_db.get_topics(user.id))


@router.post("/topics", response_model=TopicResponse)
def update_topics(payload: TopicUpdateRequest, user: AuthUser = CurrentUser) -> TopicResponse:
    return TopicResponse(topics=store_db.set_topics(user.id, payload.topics))


@router.get("/categories", response_model=CategoryListResponse)
def get_categories(user: AuthUser = CurrentUser) -> CategoryListResponse:
    return CategoryListResponse(categories=store_db.get_categories(user.id))


@router.post("/categories", response_model=CategoryListResponse)
def update_categories(
    payload: CategoryUpdateRequest, user: AuthUser = CurrentUser
) -> CategoryListResponse:
    return CategoryListResponse(
        categories=store_db.set_categories(user.id, payload.categories)
    )
