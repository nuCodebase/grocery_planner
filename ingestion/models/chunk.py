"""Domain models shared by chunkers and vector-store adapters."""

from dataclasses import dataclass, field
from typing import Any, TypeAlias


@dataclass(frozen=True, slots=True)
class Chunk:
    """A normalized text chunk produced by any chunking strategy."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Chunk id cannot be empty")
        if not self.text.strip():
            raise ValueError("Chunk text cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return the existing JSON-compatible chunk payload shape."""

        return {
            "text": self.text,
            "metadata": dict(self.metadata),
        }


ChunkCollection: TypeAlias = dict[str, Chunk]


def chunks_to_dict(chunks: ChunkCollection) -> dict[str, dict[str, Any]]:
    """Serialize a keyed chunk collection for JSON output."""

    return {chunk_id: chunk.to_dict() for chunk_id, chunk in chunks.items()}
