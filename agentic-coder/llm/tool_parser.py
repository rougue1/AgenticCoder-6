"""Tolerant tool-call parser (spec §11).

Local models emit tool calls inconsistently. This parser accepts:

* ``<tool_call> {json} </tool_call>`` tag blocks,
* ```` ```json `` / ```` ``` `` fenced code blocks,
* raw JSON objects embedded in surrounding prose,
* trailing commas, and
* **raw (unescaped) newlines/tabs inside JSON string values** — the common case
  for ``write_file`` content.

It extracts the *first* valid tool call. Key aliases are normalized:
``tool``/``name`` for the tool, and ``args``/``arguments``/``parameters`` for
the argument object.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_TOOL_KEYS = ("tool", "name", "tool_name", "function")
_ARG_KEYS = ("args", "arguments", "parameters", "params", "input")
_KNOWN_TOOLS = {"read_file", "write_file", "edit_file", "run"}

_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|tool_call|tool)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


@dataclass
class ToolCall:
    """A normalized tool invocation request from the model."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    # True when this call was reconstructed by _salvage_tool_call rather than
    # parsed cleanly — the strong signal that the server truncated the arguments
    # mid-stream. The registry uses it to attach a heredoc-append recovery note so
    # the model stops re-emitting the same too-large write (spec adoption #4).
    salvaged: bool = False

    @property
    def is_known(self) -> bool:
        return self.name in _KNOWN_TOOLS


def extract_tool_call(text: str) -> ToolCall | None:
    """Return the first parseable tool call in *text*, or ``None``."""
    calls = extract_all_tool_calls(text, limit=1)
    return calls[0] if calls else None


def extract_json(text: str) -> Any | None:
    """Return the largest top-level JSON object found in *text*, tolerantly.

    Used by stages that expect a JSON document (e.g. the task planner). Strips
    code fences, tolerates trailing commas and unescaped newlines, and picks the
    longest balanced ``{...}`` that parses — which is almost always the intended
    payload rather than a small inline example.
    """
    if not text:
        return None
    # Prefer a fenced block if present (models often fence the JSON answer).
    candidates: list[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.extend(_json_objects(m.group(1)))
    candidates.extend(_json_objects(text))
    parsed = [obj for blob in candidates if (obj := _loads_tolerant(blob)) is not None]
    parsed = [p for p in parsed if isinstance(p, (dict, list))]
    if not parsed:
        return None
    # Heuristic: the richest structure (most keys/items) is the real document.
    return max(parsed, key=_richness)


def _richness(obj: Any) -> int:
    if isinstance(obj, dict):
        return len(json.dumps(obj, default=str))
    if isinstance(obj, list):
        return len(json.dumps(obj, default=str))
    return 0


def extract_all_tool_calls(text: str, limit: int | None = None) -> list[ToolCall]:
    """Return every parseable tool call, in document order, de-duplicated."""
    if not text:
        return []

    found: list[ToolCall] = []
    seen_spans: list[str] = []

    def _consider(blob: str) -> None:
        if limit is not None and len(found) >= limit:
            return
        obj = _loads_tolerant(blob)
        call = _to_tool_call(obj, blob)
        if call is not None and call.raw not in seen_spans:
            seen_spans.append(call.raw)
            found.append(call)

    # 1) explicit <tool_call> tags (highest signal)
    for m in _TAG_RE.finditer(text):
        for cand in _json_objects(m.group(1)):
            _consider(cand)

    # 2) fenced code blocks
    if not (limit and len(found) >= limit):
        for m in _FENCE_RE.finditer(text):
            for cand in _json_objects(m.group(1)):
                _consider(cand)

    # 3) bare JSON objects anywhere in the prose
    if not (limit and len(found) >= limit):
        for cand in _json_objects(text):
            _consider(cand)

    # 4) schema-aware salvage — handles the very common case where the model put
    #    code with unescaped quotes/newlines (e.g. Python """docstrings""") inside
    #    the write_file `content`, which makes the object un-parseable as JSON.
    if not found:
        salvaged = _salvage_tool_call(text)
        if salvaged is not None:
            found.append(salvaged)

    return found[:limit] if limit is not None else found


# ── internals ─────────────────────────────────────────────────────────────────
def _to_tool_call(obj: Any, raw: str) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None

    name = None
    for k in _TOOL_KEYS:
        if k in obj and isinstance(obj[k], str):
            name = obj[k].strip()
            break
    # OpenAI-ish shape: {"function": {"name": ..., "arguments": ...}}
    if name is None and isinstance(obj.get("function"), dict):
        fn = obj["function"]
        name = (fn.get("name") or "").strip() or None
        args = fn.get("arguments")
        if isinstance(args, str):
            args = _loads_tolerant(args) or {}
        if name:
            return ToolCall(name=name, args=args if isinstance(args, dict) else {}, raw=raw.strip())

    if not name:
        return None

    args: dict[str, Any] = {}
    for k in _ARG_KEYS:
        if k in obj and isinstance(obj[k], dict):
            args = obj[k]
            break
    else:
        # No explicit args object: treat the remaining keys as the args.
        args = {k: v for k, v in obj.items() if k not in _TOOL_KEYS}

    return ToolCall(name=name, args=args, raw=raw.strip())


def _salvage_tool_call(text: str) -> ToolCall | None:
    """Reconstruct a tool call from malformed JSON using the known schema.

    Local models frequently emit ``write_file`` with code in ``content`` that
    contains unescaped quotes/newlines (Python ``\"\"\"docstrings\"\"\"``,
    ``print(\"x\")`` …), which makes the object invalid JSON. We anchor on the
    known keys so embedded quotes in ``content`` don't matter, then unescape the
    raw value. Better an imperfect file the fix-loop can repair than nothing.
    """
    m = re.search(r'"(?:tool|name|tool_name)"\s*:\s*"(read_file|write_file|edit_file|run)"', text, re.I)
    if not m:
        return None
    tool = m.group(1).lower()

    if tool == "write_file":
        path = _grab_scalar(text, "path")
        if path is None:
            return None
        content = _grab_content(text)
        if content is None:
            return None
        summary = _grab_scalar(text, "summary") or _grab_scalar(text, "description") or ""
        return ToolCall("write_file", {"path": path, "content": content, "summary": summary}, raw=text[:200], salvaged=True)

    if tool == "edit_file":
        path = _grab_scalar(text, "path")
        if path is None:
            return None
        # old_string ends where new_string begins; new_string runs to the object
        # close. Like write_file content, these can hold unescaped quotes/newlines.
        old = _grab_between(text, "old_string", ("new_string",))
        new = _grab_between(text, "new_string", ("summary", "description", "path"))
        if old is None or new is None:
            return None
        return ToolCall("edit_file", {"path": path, "old_string": old, "new_string": new}, raw=text[:200], salvaged=True)

    if tool == "read_file":
        path = _grab_scalar(text, "path")
        return ToolCall("read_file", {"path": path}, raw=text[:200], salvaged=True) if path else None

    if tool == "run":
        cmd = _grab_scalar(text, "cmd") or _grab_scalar(text, "command")
        return ToolCall("run", {"cmd": cmd}, raw=text[:200], salvaged=True) if cmd else None
    return None


def _grab_scalar(text: str, key: str) -> str | None:
    """Extract a short string value for *key* (path/summary/cmd — no embedded quotes)."""
    m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    return _unescape(m.group(1)) if m else None


def _grab_content(text: str) -> str | None:
    """Extract the ``content`` value, tolerating unescaped quotes/newlines inside it."""
    return _grab_between(text, "content", ("summary", "description", "path"))


def _grab_between(text: str, key: str, next_keys: tuple[str, ...]) -> str | None:
    """Extract a long string value for *key*, tolerating unescaped quotes/newlines.

    The end is anchored on the next known key in *next_keys* or the closing of the
    object, so quotes within embedded code don't terminate it early. Used for the
    big free-form fields (write_file ``content``, edit_file ``old_string`` /
    ``new_string``) that make a call un-parseable as strict JSON.
    """
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"', text)
    if not m:
        return None
    rest = text[m.end():]
    alt = "|".join(re.escape(k) for k in next_keys)
    nxt = re.search(rf'"\s*,\s*"(?:{alt})"\s*:', rest) if alt else None
    if nxt:
        raw = rest[: nxt.start()]
    else:
        tail = re.search(r'"\s*\}\s*\}?\s*`*\s*$', rest.rstrip())
        raw = rest[: tail.start()] if tail else rest[: rest.rfind('"')] if '"' in rest else rest
    return _unescape(raw)


def _unescape(s: str) -> str:
    """Apply JSON string escapes (\\n, \\t, \\", \\\\, …) leaving raw newlines intact."""
    out: list[str] = []
    i = 0
    mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            out.append(mapping.get(s[i + 1], "\\" + s[i + 1]))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _json_objects(text: str) -> list[str]:
    """Yield balanced ``{...}`` substrings, respecting strings and escapes."""
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
                    start = -1
    return out


def _loads_tolerant(blob: str) -> Any:
    """Best-effort JSON decode through escalating repair passes."""
    if not isinstance(blob, str):
        return blob
    blob = blob.strip()
    if not blob:
        return None

    candidates = [
        blob,
        _strip_trailing_commas(blob),
        _escape_control_chars(blob),
        _strip_trailing_commas(_escape_control_chars(blob)),
    ]
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _strip_trailing_commas(s: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", s)


def _escape_control_chars(s: str) -> str:
    """Escape literal newlines/tabs/CR that appear *inside* JSON string values.

    Models frequently emit ``write_file`` content with real newlines inside the
    JSON string, which is invalid JSON. This walks the text tracking string
    state and replaces the offending control chars with their escape sequences.
    """
    out: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                out.append(ch)
                esc = False
            elif ch == "\\":
                out.append(ch)
                esc = True
            elif ch == '"':
                out.append(ch)
                in_str = False
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_str = True
    return "".join(out)
