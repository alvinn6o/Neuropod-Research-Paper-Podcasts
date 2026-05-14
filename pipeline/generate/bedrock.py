"""AWS Bedrock client for Anthropic Claude models.

Bedrock uses Anthropic's `Messages` API but exposes models under different IDs
and authenticates via SigV4 (AWS credentials) instead of an Anthropic API key.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from .._http import ProviderError
from ..provider_status import record_failure, record_success

logger = logging.getLogger("neuropod.bedrock")


# Inference profiles cover the most common Anthropic models on Bedrock.
# Map a logical name → Bedrock-specific model id (region-prefixed cross-region
# inference profiles preferred where they exist).
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-6-20251010-v1:0"


class BedrockClient:
    def __init__(
        self,
        *,
        access_key: str,
        secret_key: str,
        region: str,
        session_token: str | None = None,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.session_token = session_token or None
        self.model_id = model_id

    def _client(self):
        try:
            import boto3
        except ImportError as exc:
            raise ProviderError("bedrock", None, "boto3 not installed") from exc
        return boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            aws_session_token=self.session_token,
        )

    def messages(self, *, system: str, user_prompt: str, max_tokens: int = 2400) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        start = time.time()
        try:
            client = self._client()
            response = client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            payload = json.loads(response["body"].read())
        except ProviderError:
            raise
        except Exception as exc:
            err = str(exc)
            record_failure("script:bedrock", error=err[:300])
            raise ProviderError("bedrock", None, err) from exc

        record_success("script:bedrock", latency_ms=int((time.time() - start) * 1000))
        try:
            return payload["content"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("bedrock", None, f"unexpected response shape: {payload}") from exc


def parse_bedrock_credentials(raw: Any) -> dict[str, str] | None:
    """Accepts either a JSON string or dict with access_key/secret_key/region."""
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            return None
    else:
        data = raw
    if not isinstance(data, dict):
        return None
    if not (data.get("access_key") and data.get("secret_key") and data.get("region")):
        return None
    return {
        "access_key": data["access_key"],
        "secret_key": data["secret_key"],
        "region": data["region"],
        "session_token": data.get("session_token") or "",
        "model_id": data.get("model_id") or DEFAULT_MODEL_ID,
    }
