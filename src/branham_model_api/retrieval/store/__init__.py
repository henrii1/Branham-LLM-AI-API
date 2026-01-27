"""
Chunk store module for SQLite-backed text retrieval.
"""

from .chunk_store import ChunkRecord, ChunkStore, SermonRecord

__all__ = [
    "ChunkStore",
    "ChunkRecord",
    "SermonRecord",
]
