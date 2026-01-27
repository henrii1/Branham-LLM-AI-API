"""
Context expansion for retrieved chunks.

Expands retrieved chunks by ±1 adjacent chunk within the same sermon.
Deduplicates after expansion to avoid duplicate context.

Flow:
1. For each retrieved chunk, fetch adjacent chunks (±1)
2. Deduplicate by chunk_id
3. Group by sermon for coherent context presentation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from branham_model_api.core.pipeline.fusion import FusedHit
from branham_model_api.retrieval.store.chunk_store import ChunkRecord, ChunkStore


@dataclass
class ExpandedChunk:
    """A chunk with its context (may include adjacent chunks)."""

    chunk_id: str
    date_id: str
    chunk_index: int
    text: str
    paragraph_start: int
    paragraph_end: int
    word_count: int
    score: float | None  # Original retrieval score (None for expanded neighbors)
    is_retrieved: bool  # True if originally retrieved, False if expansion neighbor


@dataclass
class ExpandedSermon:
    """A sermon with its expanded chunks and metadata."""

    date_id: str
    title: str | None  # Sermon title (for display/reference)
    source: str | None  # Source (e.g., PDF filename)
    language: str  # Language code (default "en")
    best_score: float  # Best score among originally retrieved chunks
    chunks: list[ExpandedChunk]  # Ordered by chunk_index


def expand_chunks(
    fused_hits: Sequence[FusedHit],
    chunk_store: ChunkStore,
    *,
    delta: int = 1,
) -> list[ExpandedChunk]:
    """
    Expand retrieved chunks by ±delta adjacent chunks.

    Args:
        fused_hits: Retrieved chunks with scores
        chunk_store: Database for looking up chunks
        delta: Number of adjacent chunks to include (default ±1)

    Returns:
        List of ExpandedChunk, deduplicated, sorted by (date_id, chunk_index)
    """
    # Track which chunks were originally retrieved (for scoring/marking)
    retrieved_chunk_ids = {hit.chunk_id for hit in fused_hits}
    chunk_scores = {hit.chunk_id: hit.score for hit in fused_hits}

    # Collect all chunk IDs we need (original + expanded)
    all_chunks: dict[str, ExpandedChunk] = {}

    # First, get all original chunks to know their positions
    original_records = chunk_store.get_chunks(list(retrieved_chunk_ids))

    # Expand each original chunk
    for chunk_id, record in original_records.items():
        # Get adjacent chunks from the same sermon
        adjacent = chunk_store.get_adjacent_chunks(
            record.date_id,
            record.chunk_index,
            delta=delta,
        )

        # Add all adjacent chunks (including original)
        for adj in adjacent:
            if adj.chunk_id not in all_chunks:
                is_retrieved = adj.chunk_id in retrieved_chunk_ids
                all_chunks[adj.chunk_id] = ExpandedChunk(
                    chunk_id=adj.chunk_id,
                    date_id=adj.date_id,
                    chunk_index=adj.chunk_index,
                    text=adj.text,
                    paragraph_start=adj.paragraph_start,
                    paragraph_end=adj.paragraph_end,
                    word_count=adj.word_count,
                    score=chunk_scores.get(adj.chunk_id),
                    is_retrieved=is_retrieved,
                )

    # Sort by (date_id, chunk_index) for coherent ordering
    result = list(all_chunks.values())
    result.sort(key=lambda c: (c.date_id, c.chunk_index))
    return result


def group_expanded_by_sermon(
    expanded_chunks: Sequence[ExpandedChunk],
    chunk_store: ChunkStore,
) -> list[ExpandedSermon]:
    """
    Group expanded chunks by sermon and fetch sermon metadata.

    Args:
        expanded_chunks: Expanded and deduplicated chunks
        chunk_store: Database for looking up sermon metadata

    Returns:
        List of ExpandedSermon (with metadata), sorted by best_score descending
    """
    # Group by date_id
    sermons: dict[str, list[ExpandedChunk]] = {}
    for chunk in expanded_chunks:
        if chunk.date_id not in sermons:
            sermons[chunk.date_id] = []
        sermons[chunk.date_id].append(chunk)

    # Fetch sermon metadata for all date_ids
    date_ids = list(sermons.keys())
    sermon_metadata = chunk_store.get_sermons(date_ids)

    # Create ExpandedSermon objects
    result = []
    for date_id, chunks in sermons.items():
        # Sort chunks by chunk_index within sermon
        chunks.sort(key=lambda c: c.chunk_index)

        # Find best score among originally retrieved chunks
        retrieved_scores = [c.score for c in chunks if c.is_retrieved and c.score is not None]
        best_score = max(retrieved_scores) if retrieved_scores else 0.0

        # Get metadata (with defaults if missing)
        meta = sermon_metadata.get(date_id)
        title = meta.title if meta else None
        source = meta.source if meta else None
        language = meta.language if meta else "en"

        result.append(ExpandedSermon(
            date_id=date_id,
            title=title,
            source=source,
            language=language,
            best_score=best_score,
            chunks=chunks,
        ))

    # Sort sermons by best score
    result.sort(key=lambda s: s.best_score, reverse=True)
    return result


def expand_and_group(
    fused_hits: Sequence[FusedHit],
    chunk_store: ChunkStore,
    *,
    delta: int = 1,
) -> list[ExpandedSermon]:
    """
    Convenience function: expand chunks and group by sermon.

    Args:
        fused_hits: Retrieved chunks with scores
        chunk_store: Database for looking up chunks
        delta: Number of adjacent chunks to include (default ±1)

    Returns:
        List of ExpandedSermon with metadata and expanded context,
        sorted by best_score descending
    """
    expanded = expand_chunks(fused_hits, chunk_store, delta=delta)
    return group_expanded_by_sermon(expanded, chunk_store)


def format_sermon_context(sermon: ExpandedSermon) -> str:
    """
    Format a sermon's chunks as context text.

    Includes sermon metadata and chunk markers for reference tracking.
    """
    # Header with date_id and title
    if sermon.title:
        header = f"[Sermon: {sermon.date_id} - {sermon.title}]"
    else:
        header = f"[Sermon: {sermon.date_id}]"

    lines = [header]
    for chunk in sermon.chunks:
        marker = "*" if chunk.is_retrieved else " "
        lines.append(f"[{marker}¶{chunk.paragraph_start}–{chunk.paragraph_end}]")
        lines.append(chunk.text)
        lines.append("")  # Blank line between chunks
    return "\n".join(lines)


def format_all_context(sermons: Sequence[ExpandedSermon]) -> str:
    """
    Format all sermon contexts for the prompt.

    Args:
        sermons: Expanded sermons sorted by relevance

    Returns:
        Formatted context string
    """
    parts = []
    for i, sermon in enumerate(sermons, 1):
        parts.append(f"--- Context {i} ---")
        parts.append(format_sermon_context(sermon))
        parts.append("")
    return "\n".join(parts)
