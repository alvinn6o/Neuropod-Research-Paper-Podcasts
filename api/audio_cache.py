from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger("neuropod.cache")


class AudioCache:
    """Disk cache for synthesized audio with LRU eviction by total bytes."""

    def __init__(self, root: Path, max_bytes: int = 500 * 1024 * 1024) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> tuple[bytes, str] | None:
        bin_path = self.root / f"{key}.bin"
        meta_path = self.root / f"{key}.meta"
        if not (bin_path.exists() and meta_path.exists()):
            return None
        try:
            bin_path.touch()
            meta_path.touch()
            return bin_path.read_bytes(), meta_path.read_text().strip() or "audio/wav"
        except OSError as exc:
            logger.warning("audio cache read failed for %s: %s", key, exc)
            return None

    def put(self, key: str, data: bytes, content_type: str) -> None:
        bin_path = self.root / f"{key}.bin"
        meta_path = self.root / f"{key}.meta"
        with self._lock:
            try:
                bin_path.write_bytes(data)
                meta_path.write_text(content_type)
            except OSError as exc:
                logger.warning("audio cache write failed for %s: %s", key, exc)
                return
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        files = [p for p in self.root.glob("*.bin") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        if total <= self.max_bytes:
            return
        files.sort(key=lambda p: p.stat().st_mtime)
        while files and total > self.max_bytes:
            victim = files.pop(0)
            size = victim.stat().st_size
            try:
                victim.unlink(missing_ok=True)
                meta = victim.with_suffix(".meta")
                meta.unlink(missing_ok=True)
                total -= size
                logger.info("audio cache evicted %s (%d bytes)", victim.name, size)
            except OSError as exc:
                logger.warning("audio cache eviction failed for %s: %s", victim, exc)
                break
