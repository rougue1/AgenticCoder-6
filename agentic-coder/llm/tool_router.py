"""Secondary tool-call recovery — the "does this carry tool intent?" router.

A strong coding/reasoning model often produces the RIGHT work in the WRONG shape:
it narrates a change in prose, dumps several files in fenced code blocks, or emits
one malformed ``<tool_call>`` instead of the one-call-per-message protocol. We do
not want to discard that work and burn a re-ask.

This module is the two-step recovery the user asked for:

1. :func:`looks_like_tool_content` — a cheap heuristic: *does this reply look like
   it was trying to act* (write files, run commands), even if nothing parsed?
2. :func:`salvage_calls` — when it does, hand the raw text to a fast, structured
   tool-calling model (the ``tool_caller`` phase) and ask it to re-express
   EVERYTHING as a clean sequence of ``<tool_call>`` blocks, then parse them all.

The result is a list of :class:`~llm.tool_parser.ToolCall` the caller executes in
order — so a single mis-formatted reply carrying several files still lands on disk
instead of being lost. Used by ``stages/implementer.py`` and ``stages/reviewer.py``.
"""

from __future__ import annotations

import re

from llm.tool_parser import ToolCall, extract_all_tool_calls
from tools.registry import TOOL_INSTRUCTIONS

# Reply is converted by a second model; cap how much we hand it so a runaway dump
# can't overrun the tool_caller's window. Head+tail keeps both the first files and
# the trailing DONE/last file when something is genuinely huge (rare — the heredoc
# discipline discourages it).
_MAX_CONVERT_CHARS = 48_000

# Substrings that betray an attempt to call a tool even when the JSON/tags are
# malformed. Cheap and deliberately generous: a false positive only costs one fast
# tool_caller call (which then finds nothing and we fall through to the re-ask).
_TOOL_HINTS = (
    "<tool_call>",
    "write_file",
    "edit_file",
    "read_file",
    '"tool"',
    '"tool_name"',
    '"cmd"',
    '"path"',
    '"content"',
    '"old_string"',
)

# A fenced code block usually means "here is a file body". Combined with a filename
# or a create/write verb nearby, it is a strong sign the model meant to write files
# rather than just chat.
_FILENAME_RE = re.compile(r"(?m)(?:^|[\s`(])[\w./-]+\.[A-Za-z][A-Za-z0-9]{0,7}\b")
_WRITE_VERB_RE = re.compile(
    r"\b(creat|writ|add|updat|edit|modif|implement|install)\w*\b.{0,40}?"
    r"\b(file|files|module|requirements|package|test|tests|dependenc)",
    re.I | re.S,
)

_SALVAGE_PROMPT = (
    "The assistant text below was supposed to act by emitting tool calls "
    "(read_file / write_file / edit_file / run), ONE per <tool_call> block, but it "
    "used the WRONG format — it may have narrated the work, dumped one or more files "
    "in code blocks, or written malformed JSON.\n\n"
    "Re-express EVERYTHING it intended to do as a sequence of VALID <tool_call> "
    "blocks, in order, exactly one tool call per block, and output NOTHING ELSE — no "
    "prose, no explanation, no markdown around the blocks.\n"
    "Preserve every file path and the FULL file contents EXACTLY as given: never "
    "summarize, truncate, abbreviate with '... unchanged ...', or invent. If it "
    "showed N files, emit N write_file calls. If it asked to run a command (e.g. "
    "installing dependencies), emit a run call for it.\n\n"
    f"{TOOL_INSTRUCTIONS}\n\n"
    "ASSISTANT OUTPUT TO CONVERT:\n"
)


def looks_like_tool_content(text: str) -> bool:
    """True when *text* looks like it was trying to call a tool / write files.

    The "does this have any tool-call format?" gate before spending a tool_caller
    conversion. Returns True for an explicit (even malformed) tool call, and for a
    fenced code block paired with a filename or a write/create verb — i.e. a file
    the model meant to write but emitted as prose.
    """
    if not text or not text.strip():
        return False
    low = text.lower()
    if any(h in low for h in _TOOL_HINTS):
        return True
    if "```" in text and (_FILENAME_RE.search(text) or _WRITE_VERB_RE.search(text)):
        return True
    return False


def salvage_calls(client, reply: str, *, max_calls: int = 16) -> list[ToolCall]:
    """Convert a mis-formatted *reply* into executable tool calls via ``tool_caller``.

    Returns the recovered calls in document order (possibly several — one per file
    the original reply dumped), or ``[]`` if the model is unavailable or produced
    nothing parseable. Never raises: recovery is best-effort and the caller falls
    back to a normal re-ask when it yields nothing.
    """
    if not reply or not reply.strip():
        return []
    snippet = (
        reply
        if len(reply) <= _MAX_CONVERT_CHARS
        else reply[: _MAX_CONVERT_CHARS // 2] + "\n…\n" + reply[-_MAX_CONVERT_CHARS // 2 :]
    )
    try:
        result = client.complete(
            "tool_caller",
            [{"role": "user", "content": _SALVAGE_PROMPT + snippet}],
            temperature=0.0,  # faithful structural conversion, not creativity
            dump=False,
        )
    except Exception:
        return []
    return extract_all_tool_calls(result.text or result.raw, limit=max_calls)
