"""Per-user provider keys: encrypt on write, decrypt on read."""
from __future__ import annotations

import uuid
from typing import Optional

from .crypto import decrypt, encrypt, hint
from .db import cursor


VALID_PROVIDERS = {"openai", "anthropic", "elevenlabs"}


def set_key(user_id: uuid.UUID, provider: str, plaintext: str) -> str:
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"unknown provider {provider}")
    plaintext = (plaintext or "").strip()
    if not plaintext:
        raise ValueError("key cannot be empty")

    cipher = encrypt(plaintext)
    masked = hint(plaintext)
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_keys (user_id, provider, ciphertext, hint, updated_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, provider) DO UPDATE
              SET ciphertext = EXCLUDED.ciphertext,
                  hint = EXCLUDED.hint,
                  updated_at = CURRENT_TIMESTAMP
            """,
            (str(user_id), provider, cipher, masked),
        )
    return masked


def delete_key(user_id: uuid.UUID, provider: str) -> None:
    with cursor() as cur:
        cur.execute(
            "DELETE FROM user_keys WHERE user_id = %s AND provider = %s",
            (str(user_id), provider),
        )


def list_masked(user_id: uuid.UUID) -> dict[str, str]:
    """{'openai': 'abc1', 'anthropic': '...'}  — never returns plaintext."""
    with cursor() as cur:
        cur.execute(
            "SELECT provider, hint FROM user_keys WHERE user_id = %s",
            (str(user_id),),
        )
        rows = cur.fetchall()
    return {row[0]: row[1] for row in rows}


def load_keys(user_id: uuid.UUID) -> dict[str, str]:
    """Returns {provider: plaintext}. Use only inside the request that needs it."""
    with cursor() as cur:
        cur.execute(
            "SELECT provider, ciphertext FROM user_keys WHERE user_id = %s",
            (str(user_id),),
        )
        rows = cur.fetchall()
    out: dict[str, str] = {}
    for provider, cipher in rows:
        try:
            out[provider] = decrypt(cipher)
        except Exception:
            continue
    return out


def get_key(user_id: uuid.UUID, provider: str) -> Optional[str]:
    return load_keys(user_id).get(provider)
