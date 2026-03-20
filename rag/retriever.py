"""High-level RAG retrieve + ingest; optional when Qdrant is not configured."""

from __future__ import annotations

import os
from typing import Any

from .chunking import chunk_text
from .embeddings import embed_query, embed_texts
from .store import (
    default_collection_name,
    ensure_collection,
    get_qdrant_client,
    search_similar,
    upsert_chunks,
)


class RAGRetriever:
    """Embed with Ollama, store/search in Qdrant."""

    def __init__(
        self,
        collection_name: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ):
        self.client = get_qdrant_client()
        if self.client is None:
            raise RuntimeError("QDRANT_URL is not set")
        self.collection_name = collection_name or default_collection_name()
        self.top_k = top_k or int(os.getenv("RAG_TOP_K", "5"))
        st = os.getenv("RAG_SCORE_THRESHOLD", "").strip()
        self.score_threshold = score_threshold
        if self.score_threshold is None and st:
            try:
                self.score_threshold = float(st)
            except ValueError:
                self.score_threshold = None
        self._vector_size: int | None = None

    def _vector_dimension(self) -> int:
        if self._vector_size is not None:
            return self._vector_size
        v = os.getenv("RAG_VECTOR_SIZE", "").strip()
        if v:
            self._vector_size = int(v)
            return self._vector_size
        probe = embed_query("dimension probe")
        self._vector_size = len(probe)
        return self._vector_size

    def ensure_ready(self) -> None:
        ensure_collection(self.client, self.collection_name, self._vector_dimension())

    def ingest_text(
        self,
        text: str,
        source_id: str,
        metadata: dict[str, Any] | None = None,
        max_chars: int | None = None,
        overlap: int | None = None,
    ) -> dict[str, Any]:
        mc = max_chars or int(os.getenv("RAG_CHUNK_CHARS", "1200"))
        ov = overlap if overlap is not None else int(os.getenv("RAG_CHUNK_OVERLAP", "180"))
        chunks = chunk_text(text, max_chars=mc, overlap=ov)
        if not chunks:
            return {"chunks": 0, "source_id": source_id}
        self.ensure_ready()
        vectors = embed_texts(chunks)
        n = upsert_chunks(
            self.client,
            self.collection_name,
            vectors,
            chunks,
            source_id,
            extra_metadata=metadata,
        )
        return {"chunks": n, "source_id": source_id, "collection": self.collection_name}

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        self.ensure_ready()
        qv = embed_query(query)
        return search_similar(
            self.client,
            self.collection_name,
            qv,
            limit=self.top_k,
            score_threshold=self.score_threshold,
        )

    def format_context(self, hits: list[dict[str, Any]], max_chars: int = 6000) -> str:
        if not hits:
            return ""
        parts: list[str] = []
        used = 0
        for i, h in enumerate(hits, 1):
            src = h.get("source_id") or "unknown"
            body = (h.get("text") or "").strip()
            if not body:
                continue
            block = f"[{i}] (source: {src}, score: {h.get('score', 0):.3f})\n{body}\n"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n".join(parts).strip()


def build_rag_retriever_if_configured() -> RAGRetriever | None:
    flag = (os.getenv("RAG_ENABLED") or "true").strip().lower()
    if flag in ("0", "false", "no", "off"):
        print("RAG disabled via RAG_ENABLED")
        return None
    if not (os.getenv("QDRANT_URL") or "").strip():
        print("RAG: QDRANT_URL not set; skipping vector store")
        return None
    try:
        r = RAGRetriever()
        r.ensure_ready()
        print(
            f"RAG: connected to Qdrant collection={r.collection_name!r}, "
            f"embed_model={os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')!r}"
        )
        return r
    except Exception as e:
        print(f"RAG: failed to initialize ({e}); continuing without RAG")
        return None
