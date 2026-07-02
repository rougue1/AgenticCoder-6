"""Token estimation shared by the client, context builder and compressor.

Uses ``tiktoken`` (cl100k_base) as a stable proxy for local-model tokenization.
Local models tokenize a bit differently, but cl100k is close enough for budget
arithmetic and far better than a naive char/4 heuristic. Falls back to char/4
if tiktoken is unavailable.
"""

from __future__ import annotations

from functools import lru_cache

try:  # tiktoken is in requirements but degrade gracefully if missing
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENC = None


@lru_cache(maxsize=4096)
def _count_cached(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


def estimate_tokens(text: str) -> int:
    """Estimate the token count of *text*."""
    if not text:
        return 0
    # Cache only modest strings; large file blobs would blow the cache.
    if len(text) <= 8192:
        return _count_cached(text)
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate tokens for a list of chat messages (+ small per-message overhead)."""
    total = 0
    for msg in messages:
        total += estimate_tokens(str(msg.get("content", ""))) + 4
    return total + 2
