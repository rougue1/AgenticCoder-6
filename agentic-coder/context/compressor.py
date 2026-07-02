"""Token-budget compressor (spec §4b compression policy, §10).

Context is assembled as an ordered list of :class:`Block` objects, each with a
``priority`` (0 = never compress) and an optional one-line ``summary`` fallback.
When the assembled context exceeds the phase budget, the compressor replaces the
most distant/least-relevant compressible blocks with their summaries — never
truncating mid-file, never silently dropping content — and emits a
``compression`` event documenting every decision. If it still doesn't fit after
exhausting all compressible blocks, it emits ``compression_failure`` and raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config import AppConfig
from context.conversation import with_headroom
from tokens import estimate_tokens

if TYPE_CHECKING:
    from server.events import EventBus

# Priority bands (lower = kept first / longest). 0 is never compressed.
P_CRITICAL = 0    # steering, current subtask spec, the plan, files being edited
P_DESIGN = 10     # sdd / architecture sections
P_MANIFEST = 20   # file manifest + directory listing
P_ACTIVE_SRC = 30  # source files for the active feature
P_DISTANT_SRC = 40  # completed files from distant/unrelated features


class CompressionError(RuntimeError):
    """Raised when context cannot be made to fit even fully compressed."""


@dataclass
class Block:
    """One unit of context. Compressed by swapping ``content`` for ``summary``."""

    title: str
    content: str
    priority: int = P_CRITICAL
    summary: str | None = None  # one-liner used when compressed
    kind: str = "text"
    _compressed: bool = field(default=False, repr=False)

    def rendered(self) -> str:
        if self._compressed and self.summary is not None:
            return f"{self.title} (summarized): {self.summary}"
        body = self.content if not self._compressed else (self.summary or "")
        return f"=== {self.title} ===\n{body}".rstrip()

    def tokens(self) -> int:
        return estimate_tokens(self.rendered())

    @property
    def compressible(self) -> bool:
        return self.priority > P_CRITICAL and self.summary is not None and not self._compressed


class Compressor:
    def __init__(self, config: AppConfig, bus: "EventBus"):
        self.config = config
        self.bus = bus

    def fit(self, phase: str, blocks: list[Block]) -> str:
        """Compress *blocks* to fit the phase budget; return the joined text."""
        # Leave headroom below the literal budget: the tiktoken estimate isn't the
        # local model's exact tokenizer and tends to undercount code/JSON, so
        # packing to the ceiling risks the real prompt spilling past num_ctx.
        budget = with_headroom(self.config.usable_budget_for(phase))
        total = sum(b.tokens() for b in blocks)
        if total <= budget:
            return _join(blocks)

        summarized: list[dict] = []
        # Compress most-distant (highest priority number), largest first.
        order = sorted(
            (b for b in blocks if b.compressible),
            key=lambda b: (b.priority, b.tokens()),
            reverse=True,
        )
        for block in order:
            if total <= budget:
                break
            before = block.tokens()
            block._compressed = True
            after = block.tokens()
            total = total - before + after
            summarized.append(
                {"title": block.title, "original_tokens": before, "summary_tokens": after}
            )

        if summarized:
            self.bus.emit(
                "compression",
                phase,
                summarized_files=summarized,
                budget=budget,
                resulting_tokens=total,
            )

        if total > budget:
            self.bus.emit(
                "compression_failure",
                phase,
                budget=budget,
                resulting_tokens=total,
                message="context exceeds budget even after full compression",
            )
            raise CompressionError(
                f"{phase}: context {total} tokens exceeds budget {budget} even after compressing "
                f"{len(summarized)} block(s)."
            )
        return _join(blocks)


def _join(blocks: list[Block]) -> str:
    return "\n\n".join(b.rendered() for b in blocks if b.rendered().strip())
