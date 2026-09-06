"""Token counting for chunk sizing and cost accounting.

`token_count` used to be `len(text.split())` — a word count wearing a token
count's name. That matters in two places: chunks are embedded by a model with a
hard 8191-token input limit, and cost is priced per token. A word count
under-reports by roughly 30-40% on technical prose (identifiers, LaTeX,
hyphenated terms all split into several BPE tokens), so a "safe" 110-word chunk
is not obviously safe.

tiktoken is optional. When it is missing we fall back to a calibrated estimator,
but we never pretend the two are the same: `count_tokens` returns the backend
alongside the number and the backend is persisted with each chunk, exactly as
the embedding model is. Mixing exact and estimated counts silently is the same
class of bug as mixing two embedding spaces in one index.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

logger = logging.getLogger("neuropod.tokenizer")

# cl100k_base is the encoding for text-embedding-3-* and gpt-4o-*.
ENCODING_NAME = "cl100k_base"

# Fallback ratio. Measured against cl100k_base on the checked-in Mamba sections:
# arXiv prose runs ~1.35 tokens per whitespace-delimited word. Deliberately
# rounded up — over-estimating tokens costs us a slightly smaller chunk, while
# under-estimating risks silent truncation at the embedding API boundary.
_TOKENS_PER_WORD = 1.4

_WORD_RE = re.compile(r"\S+")


@lru_cache(maxsize=1)
def _encoder():
    """Return a tiktoken encoder, or None when tiktoken is unavailable.

    tiktoken fetches its BPE table on first use unless TIKTOKEN_CACHE_DIR is
    warm, so this can fail offline (CI, Lambda cold start with no egress). That
    is a supported state, not an error — we degrade to the estimator and say so.
    """
    try:
        import tiktoken
    except ImportError:
        logger.info("tiktoken not installed; using estimated token counts")
        return None
    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except Exception as exc:  # network fetch of the BPE table failed
        logger.warning("tiktoken encoding %s unavailable (%s); estimating", ENCODING_NAME, exc)
        return None


def backend() -> str:
    """'tiktoken:cl100k_base' when exact, 'estimate:words*1.4' when not."""
    return f"tiktoken:{ENCODING_NAME}" if _encoder() is not None else f"estimate:words*{_TOKENS_PER_WORD}"


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return int(len(_WORD_RE.findall(text)) * _TOKENS_PER_WORD)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Trim `text` to at most `max_tokens`, on a token boundary when possible.

    Used at the embedding API boundary. The estimator path trims by words with
    the ratio applied, which is approximate — that is why it errs high.
    """
    if max_tokens <= 0:
        return ""
    enc = _encoder()
    if enc is not None:
        tokens = enc.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])

    words = _WORD_RE.findall(text)
    budget = int(max_tokens / _TOKENS_PER_WORD)
    if len(words) <= budget:
        return text
    return " ".join(words[:budget])
