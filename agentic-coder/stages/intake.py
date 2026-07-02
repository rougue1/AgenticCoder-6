"""INTAKE stage (spec §7.1) — raw prompt -> project brief.

Interprets the user's request, proposes a project name + slug, and produces the
``project_brief.md`` content. Stack-agnostic. Because the project directory may
not be resolved yet (the slug can come from here), this stage does NOT write the
file — it returns the content + slug and the orchestrator writes it once the
workspace exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import promptlib
from services import Services, clean_doc
from stages.common import SYSTEM_PROMPT
from workspace import slugify

_NAME_RE = re.compile(r"^\s*PROJECT_NAME:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE)
_SLUG_RE = re.compile(r"^\s*SLUG:\s*([a-zA-Z0-9 _-]+)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass
class IntakeResult:
    brief: str
    project_name: str
    slug: str


def run(services: Services, prompt: str) -> IntakeResult:
    services.check_cancel()
    instruction = promptlib.render("intake", prompt=prompt)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    result = services.client.complete("intake", messages)
    text = clean_doc(result.text) or clean_doc(result.raw)

    name_m = _NAME_RE.search(text)
    slug_m = _SLUG_RE.search(text)
    project_name = name_m.group(1).strip() if name_m else _derive_name(prompt)
    slug = slugify(slug_m.group(1) if slug_m else project_name, fallback=slugify(prompt, "app"))

    brief = _strip_header(text)
    if not brief.strip():
        brief = f"# Project Brief\n\n## Interpreted Request\n{prompt.strip()}\n"
    return IntakeResult(brief=brief, project_name=project_name, slug=slug)


def _strip_header(text: str) -> str:
    """Remove the PROJECT_NAME/SLUG control lines, keep the markdown brief."""
    lines = text.splitlines()
    kept = [ln for ln in lines if not _NAME_RE.match(ln) and not _SLUG_RE.match(ln)]
    return "\n".join(kept).strip()


def _derive_name(prompt: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", prompt)[:4]
    return " ".join(words).title() if words else "Generated Project"
