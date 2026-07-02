"""Assembles a fresh prompt for each LLM call from disk (spec §10).

The builder produces an ordered list of :class:`Block` objects — steering first,
then design docs, then the manifest, then the specific source files a task
needs, then the task-specific instruction — and runs them through the compressor
so the result fits the phase's token budget. It returns a ``messages`` list ready
for :class:`llm.client.LLMClient`.
"""

from __future__ import annotations

from config import AppConfig
from context.compressor import (
    P_CRITICAL,
    P_DESIGN,
    P_DISTANT_SRC,
    P_MANIFEST,
    Block,
    Compressor,
)
from context.loader import Loader
from context.manifest import Manifest
from workspace import Workspace


class ContextBuilder:
    def __init__(
        self,
        workspace: Workspace,
        loader: Loader,
        manifest: Manifest,
        compressor: Compressor,
        config: AppConfig,
    ):
        self.workspace = workspace
        self.loader = loader
        self.manifest = manifest
        self.compressor = compressor
        self.config = config

    # ── generic assembly ──────────────────────────────────────────────────────
    def assemble(self, phase: str, system_prompt: str, blocks: list[Block], instruction: str) -> list[dict]:
        """Build a system+user message pair, compressing *blocks* to the budget.

        ``system_prompt`` is always kept verbatim. ``instruction`` is the
        task-specific ask and is appended last as a critical block.
        """
        body_blocks = list(blocks)
        body_blocks.append(Block(title="TASK", content=instruction.strip(), priority=P_CRITICAL, kind="instruction"))
        user_content = self.compressor.fit(phase, body_blocks)
        messages = [{"role": "user", "content": user_content}]
        if system_prompt.strip():
            messages.insert(0, {"role": "system", "content": system_prompt.strip()})
        return messages

    # ── block factories ───────────────────────────────────────────────────────
    def doc_block(self, name: str, priority: int = P_DESIGN, *, required: bool = False) -> Block | None:
        content = self.loader.doc(name)
        if not content.strip():
            return None
        return Block(
            title=name,
            content=content,
            priority=P_CRITICAL if required else priority,
            summary=f"(see {name} on disk)",
            kind="doc",
        )

    def manifest_blocks(self) -> list[Block]:
        blocks: list[Block] = []
        for name in ("file_manifest.md", "file-directory.txt"):
            content = self.loader.doc(name)
            if content.strip():
                blocks.append(
                    Block(
                        title=name,
                        content=content,
                        priority=P_MANIFEST,
                        summary="(project file manifest available on disk)",
                        kind="manifest",
                    )
                )
        return blocks

    def source_blocks(self, rel_paths, *, active: set[str] | None = None) -> list[Block]:
        """One block per existing source file. Files the subtask directly touches
        (*active*) are kept verbatim; others fall back to their manifest one-liner
        under compression."""
        active = {self.workspace.relative(p) for p in (active or set())}
        blocks: list[Block] = []
        for rel, content in self.loader.sources(rel_paths).items():
            is_active = rel in active
            summary = self.manifest.describe(rel) or "(file contents on disk)"
            blocks.append(
                Block(
                    title=f"FILE {rel}",
                    content=content,
                    priority=P_CRITICAL if is_active else P_DISTANT_SRC,
                    summary=summary,
                    kind="source",
                )
            )
        return blocks

    def text_block(self, title: str, content: str, priority: int = P_CRITICAL, summary: str | None = None) -> Block:
        return Block(title=title, content=content, priority=priority, summary=summary, kind="text")
