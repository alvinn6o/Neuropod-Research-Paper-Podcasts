from __future__ import annotations

from fastapi import APIRouter

from ..auth import AuthUser, CurrentUser
from ..models import MeResponse

router = APIRouter(prefix="/me", tags=["me"])


@router.get("", response_model=MeResponse)
def get_me(user: AuthUser = CurrentUser) -> MeResponse:
    return MeResponse(
        id=str(user.id),
        email=user.email,
        feed_slug=user.feed_slug,
    )
