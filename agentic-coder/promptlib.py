"""Jinja2 prompt template loader (spec §2 templating).

All phase prompts live as ``.j2`` files under ``prompts/``. ``render(name, **ctx)``
renders one to a string. Templates are kept trim and trailing-whitespace-stripped
so they don't waste tokens.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        autoescape=select_autoescape(enabled_extensions=(), default=False),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )


def render(name: str, **context) -> str:
    """Render ``prompts/<name>.j2`` with *context*."""
    template = _env().get_template(f"{name}.j2")
    return template.render(**context).strip()
