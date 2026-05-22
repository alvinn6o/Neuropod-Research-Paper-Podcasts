"""Audio storage: S3 in production, local disk for dev.

Switched by NEUROPOD_AUDIO_BACKEND=s3 (default 'disk').
S3 mode reads NEUROPOD_S3_BUCKET, optional NEUROPOD_S3_PREFIX, region inferred
from the standard AWS env / IAM. This app returns S3 presigned URLs directly;
CloudFront can be placed in front of the bucket in the AWS reference design.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("neuropod.audio_store")

_BACKEND = (os.getenv("NEUROPOD_AUDIO_BACKEND") or "disk").lower()
_BUCKET = os.getenv("NEUROPOD_S3_BUCKET", "")
_PREFIX = os.getenv("NEUROPOD_S3_PREFIX", "audio/")
_DISK_ROOT = Path(os.getenv("NEUROPOD_DISK_AUDIO_ROOT", "data/audio_cache"))
_PRESIGN_TTL = int(os.getenv("NEUROPOD_PRESIGN_TTL", "86400"))

_lock = threading.Lock()
_DISK_ROOT.mkdir(parents=True, exist_ok=True)


def store(*, key: str, data: bytes, content_type: str) -> str:
    """Persist `data` under `key`. Returns the storage key (caller stores in DB)."""
    if _BACKEND == "s3" and _BUCKET:
        try:
            return _store_s3(key=key, data=data, content_type=content_type)
        except Exception as exc:
            logger.warning("s3 store failed (%s); falling back to disk", exc)
    return _store_disk(key=key, data=data, content_type=content_type)


def fetch(key: str) -> Optional[tuple[bytes, str]]:
    """Read raw bytes + content-type. Used when streaming via the API."""
    if _BACKEND == "s3" and _BUCKET:
        try:
            return _fetch_s3(key)
        except Exception as exc:
            logger.warning("s3 fetch failed (%s); trying disk", exc)
    return _fetch_disk(key)


def url_for(key: str) -> Optional[str]:
    """Return an externally fetchable URL: presigned S3 or local API endpoint."""
    if _BACKEND == "s3" and _BUCKET:
        try:
            return _presign_s3(key)
        except Exception as exc:
            logger.warning("s3 presign failed (%s)", exc)
            return None
    return None  # caller falls back to local /episodes/{id}/audio route


# ---- disk implementation ----

def _store_disk(*, key: str, data: bytes, content_type: str) -> str:
    bin_path = _DISK_ROOT / f"{key}.bin"
    meta_path = _DISK_ROOT / f"{key}.meta"
    with _lock:
        try:
            bin_path.write_bytes(data)
            meta_path.write_text(content_type)
            _evict_disk()
        except OSError as exc:
            logger.warning("disk write failed for %s: %s", key, exc)
    return key


def _fetch_disk(key: str) -> Optional[tuple[bytes, str]]:
    bin_path = _DISK_ROOT / f"{key}.bin"
    meta_path = _DISK_ROOT / f"{key}.meta"
    if not (bin_path.exists() and meta_path.exists()):
        return None
    try:
        bin_path.touch()
        meta_path.touch()
        return bin_path.read_bytes(), meta_path.read_text().strip() or "audio/wav"
    except OSError as exc:
        logger.warning("disk read failed for %s: %s", key, exc)
        return None


def _evict_disk(max_bytes: int = 500 * 1024 * 1024) -> None:
    files = [p for p in _DISK_ROOT.glob("*.bin") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    if total <= max_bytes:
        return
    files.sort(key=lambda p: p.stat().st_mtime)
    while files and total > max_bytes:
        victim = files.pop(0)
        try:
            size = victim.stat().st_size
            victim.unlink(missing_ok=True)
            victim.with_suffix(".meta").unlink(missing_ok=True)
            total -= size
        except OSError:
            break


# ---- S3 implementation ----

def _s3_client():
    import boto3
    return boto3.client("s3")


def _store_s3(*, key: str, data: bytes, content_type: str) -> str:
    object_key = f"{_PREFIX}{key}"
    _s3_client().put_object(
        Bucket=_BUCKET,
        Key=object_key,
        Body=data,
        ContentType=content_type,
        CacheControl="public, max-age=86400",
    )
    return key


def _fetch_s3(key: str) -> Optional[tuple[bytes, str]]:
    try:
        obj = _s3_client().get_object(Bucket=_BUCKET, Key=f"{_PREFIX}{key}")
    except Exception:
        return None
    return obj["Body"].read(), obj.get("ContentType", "audio/mpeg")


def _presign_s3(key: str) -> str:
    return _s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": _BUCKET, "Key": f"{_PREFIX}{key}"},
        ExpiresIn=_PRESIGN_TTL,
    )
