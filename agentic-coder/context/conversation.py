"""Token-budget management for the Worker's per-subtask conversation.

The Worker runs a *growing* tool-call conversation whose ``messages`` list only
appends — assistant replies and full tool outputs — across dozens of rounds.
Without bounding, that list can outgrow the model's ``num_ctx``; Ollama then
silently front-truncates the prompt and drops the system message (the tool
protocol + the Manager's instructions), a silent, hard-to-debug failure.

Three pieces prevent it:

* :func:`pack_conversation` — a non-destructive send-time view: keep the system
  message pinned, then keep whole messages newest-first until the budget fills.
* :func:`compress_conversation` — the redesign's destructive fix-loop
  compression: when the conversation exceeds 70% of the Worker's resolved
  ``max_tokens``, the OLDEST tool-result entries (never the most recent
  failure, never the system context) are replaced in place with one-line
  summaries like ``[truncated: attempt 2 - exit code 1]``.
* :func:`cap_tool_output` — collapse one oversized tool result to head+tail
  with a marker telling the model how to recover the omitted span.

Worker messages may carry pipeline-internal bookkeeping keys (``_kind``,
``_attempt``, ``_exit``); the LLM client strips them before sending.
"""

from __future__ import annotations

from tokens import estimate_messages_tokens, estimate_tokens

# The tiktoken estimate (cl100k_base) is a proxy, not the local model's exact
# tokenizer, and it tends to UNDERcount code/JSON-heavy histories. Pack to a
# fraction of the declared budget so a slight undercount can't push the real
# prompt past num_ctx (where Ollama silently front-truncates).
PACK_HEADROOM = 0.95

# The fix-loop compression trigger: fraction of the Worker's resolved
# max_tokens above which old tool results are collapsed.
COMPRESS_TRIGGER_PCT = 0.70

# Conversation-facing cap for a single tool result, in tokens. The sandbox
# already truncates raw capture to ~60k chars; this is the smaller,
# model-actionable cap that keeps one noisy read/test-log from dominating.
TOOL_OUTPUT_CAP_TOKENS = 6000
TOOL_OUTPUT_HEAD_TAIL_TOKENS = 2000


def with_headroom(budget: int) -> int:
    """Apply :data:`PACK_HEADROOM` to *budget* (floored at a usable minimum)."""
    return max(1024, int(budget * PACK_HEADROOM))


def pack_conversation(messages: list[dict], budget: int) -> list[dict]:
    """Return a budget-fitting view of *messages* for sending to the model.

    The leading system message (if any) is always kept. The remaining messages
    are kept newest-first until ``budget`` (after :func:`with_headroom`) is
    full; the newest non-system message is always kept even if it alone exceeds
    the budget, so a single large turn degrades gracefully instead of vanishing.

    ``messages`` itself is never mutated — callers keep the full list as their
    durable record and send this trimmed copy.
    """
    if not messages:
        return messages

    effective = with_headroom(budget)

    has_system = messages[0].get("role") == "system"
    system = messages[:1] if has_system else []
    body = messages[1:] if has_system else list(messages)

    used = estimate_messages_tokens(system) if system else 0
    kept: list[dict] = []
    for msg in reversed(body):
        cost = estimate_tokens(str(msg.get("content", ""))) + 4
        if kept and used + cost > effective:
            break
        kept.append(msg)
        used += cost
    kept.reverse()
    return list(system) + kept


def compress_conversation(messages: list[dict], max_tokens: int) -> int:
    """Destructively collapse the oldest tool-result entries when the
    conversation exceeds :data:`COMPRESS_TRIGGER_PCT` of *max_tokens*.

    Only messages tagged ``_kind == "tool_result"`` are eligible; the system
    message and the most recent tool result (the latest failure evidence) are
    never touched. Returns the number of entries collapsed.
    """
    if not messages:
        return 0
    trigger = int(max_tokens * COMPRESS_TRIGGER_PCT)
    if estimate_messages_tokens(messages) <= trigger:
        return 0

    eligible = [
        i for i, m in enumerate(messages)
        if m.get("_kind") == "tool_result" and not m.get("_compressed")
    ]
    if len(eligible) <= 1:
        return 0
    eligible = eligible[:-1]  # never the most recent tool result

    collapsed = 0
    for i in eligible:
        msg = messages[i]
        attempt = msg.get("_attempt", "?")
        exit_code = msg.get("_exit", "?")
        msg["content"] = f"[truncated: attempt {attempt} - exit code {exit_code}]"
        msg["_compressed"] = True
        collapsed += 1
        if estimate_messages_tokens(messages) <= trigger:
            break
    return collapsed


def cap_tool_output(
    text: str,
    max_tokens: int = TOOL_OUTPUT_CAP_TOKENS,
    head_tail_tokens: int = TOOL_OUTPUT_HEAD_TAIL_TOKENS,
) -> str:
    """Collapse an oversized tool result to head + tail with a recovery marker.

    Output at or under *max_tokens* passes through unchanged. Larger output
    keeps the first and last ``head_tail_tokens`` worth of characters and
    inserts a marker telling the model the middle was OMITTED and to re-run
    narrower (grep/sed/head/tail) for the omitted span — so a buried detail is
    recoverable rather than silently lost.
    """
    if not text:
        return text
    total = estimate_tokens(text)
    if total <= max_tokens:
        return text
    limit = head_tail_tokens * 4  # chars (~4 chars/token)
    if len(text) <= 2 * limit:
        return text
    head = text[:limit]
    tail = text[-limit:]
    marker = (
        f"\n───── output truncated: ~{total} tokens total, "
        f"first ~{head_tail_tokens} + last ~{head_tail_tokens} shown, the middle is OMITTED. "
        f"This is a PARTIAL view — re-run narrower (grep/sed/head/tail) to read the "
        f"omitted span. ─────\n"
    )
    return head + marker + tail
