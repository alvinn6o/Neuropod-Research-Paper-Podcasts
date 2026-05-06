"""Symmetric encryption for per-user provider keys.

Uses Fernet (AES-128-CBC + HMAC-SHA256). The key lives in env / Parameter Store
as NEUROPOD_MASTER_KEY (base64 32 bytes, generated with Fernet.generate_key()).
"""
from __future__ import annotations

import base64
import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken


class CryptoError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    raw = os.getenv("NEUROPOD_MASTER_KEY", "").strip()
    if not raw:
        # Local-dev convenience: derive a deterministic key from a known string.
        # Production MUST set NEUROPOD_MASTER_KEY explicitly.
        raw = base64.urlsafe_b64encode(b"neuropod-local-dev-master-key!!!").decode()
    try:
        return Fernet(raw.encode())
    except Exception as exc:  # pragma: no cover
        raise CryptoError(f"invalid NEUROPOD_MASTER_KEY: {exc}")


def encrypt(plaintext: str) -> bytes:
    if not plaintext:
        raise CryptoError("cannot encrypt empty string")
    return _fernet().encrypt(plaintext.encode())


def decrypt(ciphertext: bytes) -> str:
    try:
        return _fernet().decrypt(bytes(ciphertext)).decode()
    except InvalidToken as exc:
        raise CryptoError("decryption failed") from exc


def hint(plaintext: str) -> str:
    """Last 4 chars of a key for masked display ('sk-...abc1')."""
    if len(plaintext) <= 4:
        return "*" * len(plaintext)
    return plaintext[-4:]


def generate_master_key() -> str:
    """For ops use: print a fresh key to set as NEUROPOD_MASTER_KEY."""
    return Fernet.generate_key().decode()
