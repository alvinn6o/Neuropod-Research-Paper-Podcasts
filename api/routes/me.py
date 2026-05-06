from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import keys_repo
from ..auth import AuthUser, CurrentUser
from ..models import KeyUpdateRequest, MeResponse

router = APIRouter(prefix="/me", tags=["me"])


@router.get("", response_model=MeResponse)
def get_me(user: AuthUser = CurrentUser) -> MeResponse:
    return MeResponse(
        id=str(user.id),
        email=user.email,
        feed_slug=user.feed_slug,
        keys=keys_repo.list_masked(user.id),
    )


@router.put("/keys", response_model=MeResponse)
def update_key(payload: KeyUpdateRequest, user: AuthUser = CurrentUser) -> MeResponse:
    try:
        keys_repo.set_key(user.id, payload.provider, payload.api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return MeResponse(
        id=str(user.id),
        email=user.email,
        feed_slug=user.feed_slug,
        keys=keys_repo.list_masked(user.id),
    )


@router.delete("/keys/{provider}", response_model=MeResponse)
def delete_key(provider: str, user: AuthUser = CurrentUser) -> MeResponse:
    if provider not in keys_repo.VALID_PROVIDERS:
        raise HTTPException(status_code=400, detail="unknown provider")
    keys_repo.delete_key(user.id, provider)
    return MeResponse(
        id=str(user.id),
        email=user.email,
        feed_slug=user.feed_slug,
        keys=keys_repo.list_masked(user.id),
    )
