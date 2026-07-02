"""Shared helpers for the document-producing pipeline stages.

The upstream stages (requirements → … → task plan) all follow the same shape:
render a template, call the phase's model, clean the response, and write the
resulting ``.agent/`` document. :func:`generate_doc` captures that.
"""

from __future__ import annotations

import promptlib
from services import Services, clean_doc

SYSTEM_PROMPT = (
    "You are a senior software engineer operating inside an autonomous build "
    "pipeline. Follow the requested output format exactly. Be concrete and "
    "decisive. Never use git. Never propose technologies outside the locked stack."
)


def generate_doc(
    services: Services,
    phase: str,
    template: str,
    out_name: str | None,
    *,
    system: str = SYSTEM_PROMPT,
    **ctx,
) -> str:
    """Render *template*, call the model for *phase*, clean + (optionally) write."""
    services.check_cancel()
    instruction = promptlib.render(template, **ctx)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]
    result = services.client.complete(phase, messages)
    content = clean_doc(result.text)
    if not content.strip():
        # Some models emit everything inside <think>; fall back to raw.
        content = clean_doc(result.raw)
    if out_name:
        services.workspace.write_agent_doc(out_name, content)
    return content
