"""Authentication: stub bearer tokens for local dev, Cognito JWT for prod.

Stub mode (NEUROPOD_AUTH_MODE=stub, the default): the frontend POSTs an email
to /auth/stub/login and gets back a UUID token. The backend stores
{token: user_id} in-memory. Good enough for local dev and self-hosted single-user.

JWT mode (NEUROPOD_AUTH_MODE=cognito): the frontend sends a Cognito-issued
JWT in `Authorization: Bearer <jwt>`. We verify the signature against the
configured Cognito JWKS and treat `sub` as the user_id.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from .db import upsert_user_by_email

logger = logging.getLogger("neuropod.auth")

_AUTH_MODE = (os.getenv("NEUROPOD_AUTH_MODE") or "stub").lower()
_COGNITO_REGION = os.getenv("NEUROPOD_COGNITO_REGION", "")
_COGNITO_POOL_ID = os.getenv("NEUROPOD_COGNITO_POOL_ID", "")
_COGNITO_AUDIENCE = os.getenv("NEUROPOD_COGNITO_CLIENT_ID", "")

_stub_lock = threading.Lock()
_stub_tokens: dict[str, "AuthUser"] = {}


@dataclass(frozen=True)
class AuthUser:
    id: uuid.UUID
    email: str
    feed_slug: str


def _slug_from_email(email: str) -> str:
    raw = email.split("@", 1)[0].lower()
    raw = re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")
    return (raw or "user")[:32]


def stub_login(email: str) -> tuple[str, AuthUser]:
    """Create or fetch a stub-mode session for the given email."""
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("invalid email")
    feed_slug = _slug_from_email(email)
    user_id, slug_assigned = upsert_user_by_email(email=email, feed_slug=feed_slug)
    user = AuthUser(id=user_id, email=email, feed_slug=slug_assigned)
    token = str(uuid.uuid4())
    with _stub_lock:
        _stub_tokens[token] = user
    return token, user


def stub_logout(token: str) -> None:
    with _stub_lock:
        _stub_tokens.pop(token, None)


def _resolve_stub(token: str) -> Optional[AuthUser]:
    with _stub_lock:
        return _stub_tokens.get(token)


def _resolve_cognito(token: str) -> Optional[AuthUser]:
    try:
        import jwt
        from jwt import PyJWKClient
    except ImportError:
        logger.error("PyJWT not installed; cannot run cognito auth")
        return None
    if not (_COGNITO_REGION and _COGNITO_POOL_ID):
        logger.error("cognito mode requires NEUROPOD_COGNITO_REGION and POOL_ID")
        return None
    issuer = f"https://cognito-idp.{_COGNITO_REGION}.amazonaws.com/{_COGNITO_POOL_ID}"
    jwks_url = f"{issuer}/.well-known/jwks.json"
    try:
        signing_key = PyJWKClient(jwks_url).get_signing_key_from_jwt(token).key
        decoded = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            audience=_COGNITO_AUDIENCE or None,
            issuer=issuer,
            options={"verify_aud": bool(_COGNITO_AUDIENCE)},
        )
    except Exception as exc:
        logger.warning("cognito jwt rejected: %s", exc)
        return None
    sub = decoded.get("sub")
    email = decoded.get("email") or decoded.get("cognito:username") or sub
    if not sub or not email:
        return None
    feed_slug = _slug_from_email(email)
    user_id, slug_assigned = upsert_user_by_email(
        email=email, feed_slug=feed_slug, fixed_user_id=uuid.UUID(sub)
    )
    return AuthUser(id=user_id, email=email, feed_slug=slug_assigned)


def require_auth(
    authorization: Optional[str] = Header(default=None),
) -> AuthUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    user = (_resolve_stub(token) if _AUTH_MODE == "stub" else _resolve_cognito(token))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    return user


def maybe_auth(
    authorization: Optional[str] = Header(default=None),
) -> Optional[AuthUser]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    return _resolve_stub(token) if _AUTH_MODE == "stub" else _resolve_cognito(token)


CurrentUser = Depends(require_auth)
OptionalUser = Depends(maybe_auth)


def auth_mode() -> str:
    return _AUTH_MODE
