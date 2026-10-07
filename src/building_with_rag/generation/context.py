"""Bounded, labelled evidence context built from semantic retrieval results."""

from dataclasses import dataclass, field

from building_with_rag.contracts import RetrievedChunk

MAX_PASSAGES = 5
MAX_CONTEXT_CHARS = 12_000


@dataclass
class Context:
    entries: list[dict] = field(default_factory=list)
    omitted: int = 0
    chars: int = 0

    @property
    def outcome(self) -> str:
        return "assembled" if self.entries else "empty"

    @property
    def labels(self) -> dict[str, str]:
        return {e["label"]: e["chunk_id"] for e in self.entries}


def build_context(results: list[RetrievedChunk]) -> Context:
    """Take passages in score order until a budget is hit; never cut a passage mid-text."""
    ctx = Context()
    for chunk in results:
        if len(ctx.entries) >= MAX_PASSAGES or ctx.chars + len(chunk.text) > MAX_CONTEXT_CHARS:
            break
        ctx.entries.append(
            {
                "label": f"E{len(ctx.entries) + 1}",
                "chunk_id": chunk.chunk_id,
                "section_id": chunk.section_id,
                "act": chunk.act,
                "act_label": chunk.act_label,
                "heading": chunk.heading,
                "chapter": chunk.chapter,
                "section_number": chunk.section_number,
                "status": chunk.status,
                "source_pdf": chunk.source_pdf,
                "needs_review": chunk.needs_review,
                "text": chunk.text,
            }
        )
        ctx.chars += len(chunk.text)
    ctx.omitted = len(results) - len(ctx.entries)
    return ctx
