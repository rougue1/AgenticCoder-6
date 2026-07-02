"""REVIEW PASS (spec §7.8) — final full-project review.

With the whole project assembled, the reviewer model runs the full test suite
once more, fixes quick cross-cutting/integration issues via tool calls, and
writes a review summary to ``review.md``. Anything it flags as unresolved is
appended to ``blocked.md`` for the user / a later run.
"""

from __future__ import annotations

import promptlib
from context.conversation import pack_conversation
from llm.tool_parser import extract_all_tool_calls
from llm.tool_router import looks_like_tool_content, salvage_calls
from services import Services, clean_doc
from stages.implementer import says_done
from taskstore import TaskStore
from tools.registry import TOOL_INSTRUCTIONS

_MAX_STEPS = 40


def run(services: Services) -> str:
    services.check_cancel()
    store = TaskStore.load(services.workspace)
    services.manifest.regenerate()

    system = (
        "You are the final reviewer in an autonomous build pipeline. You verify the "
        "whole project hangs together and fix small integration issues. Use tool "
        "calls. Stay within the locked stack. Never use git.\n\n# Tool Protocol\n"
        + TOOL_INSTRUCTIONS
    )
    instruction = promptlib.render(
        "reviewer",
        steering=services.loader.doc("steering.md"),
        stack=services.loader.doc("stack.md"),
        manifest=services.loader.doc("file_manifest.md"),
        task_summary=store.summary(),
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]

    summary_text = _drive(services, messages)
    review_doc = _build_review_doc(summary_text, store)
    services.workspace.write_agent_doc("review.md", review_doc)
    return review_doc


def _drive(services: Services, messages: list[dict]) -> str:
    """Run the reviewer's tool loop; return the final free-text summary."""
    last_reply = ""
    for _ in range(_MAX_STEPS):
        services.check_cancel()
        # Pack the growing review conversation to the phase budget (see
        # context.conversation) so a long review with big file reads can't
        # overrun num_ctx and silently drop the system prompt.
        packed = pack_conversation(messages, services.config.usable_budget_for("reviewer"))
        result = services.client.complete("reviewer", packed)
        reply = result.text or result.raw
        last_reply = reply
        messages.append({"role": "assistant", "content": reply})

        # Run every parsed call; recover mis-formatted tool intent via the
        # tool_caller model (same routing as the implementer — see llm/tool_router).
        calls = extract_all_tool_calls(reply, limit=12)
        done = says_done(reply)
        if not calls and not done and looks_like_tool_content(reply):
            calls = salvage_calls(services.client, reply, max_calls=12)

        if not calls:
            if done:
                break
            # Nudge once toward finishing if it produced neither a call nor DONE.
            messages.append(
                {"role": "user", "content": "If you are finished, reply with DONE then your review summary. Otherwise emit a tool call."}
            )
            continue

        for call in calls:
            if not call.is_known:
                messages.append({"role": "user", "content": f"Unknown tool {call.name!r}. Use read_file/write_file/edit_file/run."})
                continue
            tr = services.registry.dispatch(call, "reviewer")
            messages.append({"role": "user", "content": tr.display})
        # DONE is recognized only on a separate no-tool-call message (above).

    return _after_done(last_reply)


def _after_done(text: str) -> str:
    """Return the markdown the reviewer wrote after its final DONE line."""
    if not text:
        return ""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper() == "DONE":
            return clean_doc("\n".join(lines[i + 1 :]).strip())
    return clean_doc(text)


def _build_review_doc(summary_text: str, store: TaskStore) -> str:
    body = summary_text.strip() or "_(reviewer produced no summary)_"
    return f"# Final Review\n\n{body}\n\n---\n\n## Task Status at Review\n\n```\n{store.summary()}\n```\n"
