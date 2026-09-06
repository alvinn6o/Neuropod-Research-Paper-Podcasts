from __future__ import annotations

import io
import logging
import re
import math
import os
import struct
import time
import wave

from .._http import ProviderError, post_for_bytes
from ..provider_status import record_failure, record_success

logger = logging.getLogger("neuropod.tts")

# OpenAI's /audio/speech input cap is 4096 characters; leave headroom so a
# sentence never lands exactly on the boundary.
_OPENAI_TTS_CHAR_LIMIT = 3800

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_for_tts(script: str, *, limit: int) -> list[str]:
    """Split on sentence boundaries into pieces under `limit` characters.

    A sentence longer than `limit` on its own is hard-split rather than dropped —
    rare in narration prose, but silently losing it would reintroduce exactly the
    bug this replaces.
    """
    script = script.strip()
    if len(script) <= limit:
        return [script] if script else []

    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(script):
        if not sentence:
            continue
        if len(sentence) > limit:
            if current:
                parts.append(current)
                current = ""
            for start in range(0, len(sentence), limit):
                parts.append(sentence[start : start + limit])
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= limit:
            current = candidate
        else:
            parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    return parts



class TTSProvider:
    """Routes TTS to ElevenLabs / OpenAI when keys are present, demo otherwise."""

    def __init__(self) -> None:
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.elevenlabs_voice = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.openai_voice = os.getenv("OPENAI_TTS_VOICE", "alloy")
        self.preferred = os.getenv("NEUROPOD_TTS_PROVIDER", "auto").lower()

    @property
    def provider_name(self) -> str:
        if self.preferred == "elevenlabs" and self.elevenlabs_key:
            return "elevenlabs"
        if self.preferred == "openai" and self.openai_key:
            return "openai"
        if self.preferred == "auto":
            if self.elevenlabs_key:
                return "elevenlabs"
            if self.openai_key:
                return "openai"
        return "demo"

    def synthesize(self, script: str, title: str = "") -> tuple[bytes, str, str]:
        provider = self.provider_name

        if provider == "elevenlabs":
            try:
                start = time.time()
                audio = self._call_elevenlabs(script)
                record_success("tts:elevenlabs", latency_ms=int((time.time() - start) * 1000))
                return audio, "audio/mpeg", "elevenlabs"
            except ProviderError as exc:
                record_failure("tts:elevenlabs", error=exc.detail, status=exc.status)
                logger.warning("elevenlabs failed: %s", exc)
                if self.openai_key:
                    provider = "openai"

        if provider == "openai":
            try:
                start = time.time()
                audio = self._call_openai(script)
                record_success("tts:openai", latency_ms=int((time.time() - start) * 1000))
                return audio, "audio/mpeg", "openai"
            except ProviderError as exc:
                record_failure("tts:openai", error=exc.detail, status=exc.status)
                logger.warning("openai-tts failed: %s", exc)

        return DemoTTSProvider().synthesize(script, title), "audio/wav", "demo"

    def _call_elevenlabs(self, script: str) -> bytes:
        return post_for_bytes(
            provider="elevenlabs",
            url=f"https://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice}",
            headers={
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_key,
                "Accept": "audio/mpeg",
            },
            body={
                "text": script,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {"stability": 0.55, "similarity_boost": 0.75},
            },
            timeout=180,
        )

    def _call_openai(self, script: str) -> bytes:
        """Synthesize the whole script, in sentence-aligned pieces.

        The OpenAI speech endpoint caps input at 4096 characters. This used to
        be `script[:4000]`, a hard truncation: scripts target 800-1200 words
        (~5,000-7,500 characters), so roughly a third of every episode was
        silently dropped and the audio ended mid-sentence. Nothing logged it.

        MP3 frames are self-contained, so concatenating the parts from one
        encoder produces a playable file. It is not a substitute for real
        stitching with crossfades — `synthesize/audio_post.py` exists for that
        and is currently unwired — but it does not lose a third of the script.
        """
        parts = _split_for_tts(script, limit=_OPENAI_TTS_CHAR_LIMIT)
        if len(parts) > 1:
            logger.info("openai tts: %d chars split into %d requests", len(script), len(parts))
        return b"".join(self._call_openai_chunk(part) for part in parts)

    def _call_openai_chunk(self, text: str) -> bytes:
        return post_for_bytes(
            provider="openai-tts",
            url="https://api.openai.com/v1/audio/speech",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_key}",
            },
            body={
                "model": "tts-1",
                "voice": self.openai_voice,
                "input": text,
                "response_format": "mp3",
            },
            timeout=180,
        )


class DemoTTSProvider:
    sample_rate = 22_050

    def synthesize(self, script: str, title: str = "") -> bytes:
        duration = min(max(len(script.split()) / 90.0, 2.5), 7.0)
        primary = 440 + (len(title) % 5) * 35
        secondary = primary * 1.25

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)

            total_frames = int(duration * self.sample_rate)
            for frame in range(total_frames):
                t = frame / self.sample_rate
                envelope = min(t / 0.35, 1.0) * min((duration - t) / 0.45, 1.0)
                amplitude = 9000 * max(envelope, 0.0)
                signal = (
                    math.sin(2 * math.pi * primary * t)
                    + 0.45 * math.sin(2 * math.pi * secondary * t)
                )
                sample = int(amplitude * signal / 1.45)
                wav_file.writeframes(struct.pack("<h", sample))

        return buffer.getvalue()
