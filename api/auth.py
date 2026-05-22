"""Authentication: stub bearer tokens (DB-persisted) for local/self-hosted,
Cognito JWT for prod.

Stub mode (NEUROPOD_AUTH_MODE=stub, default): /auth/stub/login mints a UUID
token and writes it to the auth_sessions table. Tokens survive process
restarts and (in prod) Lambda cold starts.

JWT mode (NEUROPOD_AUTH_MODE=cognito): no DB row — the JWT itself is the
session. We verify the signature against the Cognito JWKS each request and
extract `sub` as user_id.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, Query, status

from .db import cursor, get_user_by_slug, upsert_user_by_email

logger = logging.getLogger("neuropod.auth")

_AUTH_MODE = (os.getenv("NEUROPOD_AUTH_MODE") or "stub").lower()
_COGNITO_REGION = os.getenv("NEUROPOD_COGNITO_REGION", "")
_COGNITO_POOL_ID = os.getenv("NEUROPOD_COGNITO_POOL_ID", "")
_COGNITO_AUDIENCE = os.getenv("NEUROPOD_COGNITO_CLIENT_ID", "")


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
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("invalid email")
    feed_slug = _slug_from_email(email)
    user_id, slug_assigned = upsert_user_by_email(email=email, feed_slug=feed_slug)
    user = AuthUser(id=user_id, email=email, feed_slug=slug_assigned)
    token = uuid.uuid4().hex
    with cursor() as cur:
        cur.execute(
            "INSERT INTO auth_sessions (token, user_id) VALUES (%s, %s)",
            (token, str(user_id)),
        )
    return token, user


def stub_logout(token: str) -> None:
    with cursor() as cur:
        cur.execute("DELETE FROM auth_sessions WHERE token = %s", (token,))


def _resolve_stub(token: str) -> Optional[AuthUser]:
    with cursor() as cur:
        cur.execute(
            """
            SELECT u.id, u.email, u.feed_slug
            FROM auth_sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = %s
            """,
            (token,),
        )
        row = cur.fetchone()
        if not row:
            return None
        cur.execute(
            "UPDATE auth_sessions SET last_seen = CURRENT_TIMESTAMP WHERE token = %s",
            (token,),
        )
    return AuthUser(id=_as_uuid(row[0]), email=row[1], feed_slug=row[2])


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


def require_auth_with_query_token(
    authorization: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None, alias="token"),
) -> AuthUser:
    """Like require_auth, but also accepts the bearer token as ?token=... query
    param. Needed for endpoints invoked by the browser's native <audio>/<img>
    elements, which can't attach an Authorization header. Use sparingly — query
    tokens leak into server access logs; only use for short-lived audio URLs."""
    raw: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        raw = authorization.split(" ", 1)[1].strip()
    elif token:
        raw = token.strip()
    if not raw:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")

    user = _resolve_stub(raw) if _AUTH_MODE == "stub" else _resolve_cognito(raw)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    return user


CurrentUser = Depends(require_auth)
CurrentUserWithQueryToken = Depends(require_auth_with_query_token)
OptionalUser = Depends(maybe_auth)


def auth_mode() -> str:
    return _AUTH_MODE


def _as_uuid(value) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))
