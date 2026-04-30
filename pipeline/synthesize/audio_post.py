"""Lightweight audio post-processing — fade-in/out + simple loudness pass.

Uses pydub if available; otherwise returns input unchanged. Never throws fatally."""
from __future__ import annotations

import io
import logging

logger = logging.getLogger("neuropod.audio_post")


def post_process(audio_bytes: bytes, content_type: str, target_dbfs: float = -16.0) -> bytes:
    """Apply gentle loudness normalization + small fade-in/out.

    Falls back to passthrough if pydub or ffmpeg isn't available."""
    try:
        from pydub import AudioSegment
    except Exception:
        return audio_bytes

    fmt = "mp3" if content_type == "audio/mpeg" else "wav"
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    except Exception as exc:
        logger.info("audio_post: decode failed (%s); passthrough", exc)
        return audio_bytes

    if audio.dBFS == float("-inf"):
        return audio_bytes

    gain = target_dbfs - audio.dBFS
    gain = max(min(gain, 8.0), -8.0)
    audio = audio.apply_gain(gain)
    audio = audio.fade_in(220).fade_out(420)

    out_buffer = io.BytesIO()
    try:
        audio.export(out_buffer, format=fmt, bitrate="128k" if fmt == "mp3" else None)
        return out_buffer.getvalue()
    except Exception as exc:
        logger.info("audio_post: export failed (%s); passthrough", exc)
        return audio_bytes
