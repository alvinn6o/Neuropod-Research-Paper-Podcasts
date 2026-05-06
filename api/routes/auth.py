from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from .. import keys_repo
from ..auth import AuthUser, CurrentUser, auth_mode, stub_login, stub_logout
from ..models import AuthSession, MeResponse, StubLoginRequest

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/mode")
def get_mode() -> dict:
    return {"mode": auth_mode()}


@router.post("/stub/login", response_model=AuthSession)
def stub_login_endpoint(payload: StubLoginRequest) -> AuthSession:
    if auth_mode() != "stub":
        raise HTTPException(status_code=400, detail="stub login disabled in this environment")
    try:
        token, user = stub_login(payload.email)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    masked = keys_repo.list_masked(user.id)
    return AuthSession(
        token=token,
        user=MeResponse(
            id=str(user.id),
            email=user.email,
            feed_slug=user.feed_slug,
            keys=masked,
        ),
    )


@router.post("/logout")
def logout(authorization: str | None = None, user: AuthUser = CurrentUser) -> dict:
    if authorization and authorization.lower().startswith("bearer "):
        stub_logout(authorization.split(" ", 1)[1].strip())
    return {"ok": True}
