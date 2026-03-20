"""Simple character-based chunking with overlap (good enough for markdown/plain text)."""

from __future__ import annotations


def chunk_text(
    text: str,
    max_chars: int = 1200,
    overlap: int = 180,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0
        # avoid infinite loop on tiny overlap
        if start >= end:
            start = end
    return chunks
