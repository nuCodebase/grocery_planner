"""Shared data models for the ingestion pipeline."""

from ingestion.models.chunk import Chunk, ChunkCollection, chunks_to_dict

__all__ = ["Chunk", "ChunkCollection", "chunks_to_dict"]
