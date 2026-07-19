"""Qdrant collection helpers for Atlas RAG."""

from __future__ import annotations

import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# Qdrant Cloud UI often creates a *named* vector (e.g. "dense"); raw list queries then return 0 hits.
_VECTOR_NAME_CACHE: dict[str, str | None] = {}


def _invalidate_vector_name_cache(collection_name: str) -> None:
    _VECTOR_NAME_CACHE.pop(collection_name, None)


def _vector_name_from_collection_schema(
    client: QdrantClient, collection_name: str
) -> str | None:
    """Infer name from collection config (used when there are no points yet)."""
    try:
        info = client.get_collection(collection_name)
        vectors = getattr(info.config.params, "vectors", None)
        if vectors is None:
            return None
        if isinstance(vectors, qm.VectorParams):
            return None
        if isinstance(vectors, dict) and vectors:
            if len(vectors) == 1:
                only = next(iter(vectors.values()))
                if isinstance(only, qm.VectorParams):
                    return next(iter(vectors.keys()))
            return None
        dumped = None
        for meth in ("model_dump", "dict"):
            if hasattr(vectors, meth):
                try:
                    dumped = getattr(vectors, meth)()
                    break
                except Exception:
                    pass
        if isinstance(dumped, dict):
            if len(dumped) == 1 and "root" in dumped and isinstance(dumped["root"], dict):
                dumped = dumped["root"]
            if len(dumped) == 1:
                k, v = next(iter(dumped.items()))
                if isinstance(v, dict) and "size" in v:
                    return k
                if isinstance(v, qm.VectorParams):
                    return k
    except Exception as e:
        print(f"_vector_name_from_collection_schema({collection_name!r}): {e}")
    return None


def _collection_top_level_is_unnamed_vector_params(
    client: QdrantClient, collection_name: str
) -> bool:
    """True when collection config is a single default vector (not a named map like {\"dense\": ...})."""
    try:
        info = client.get_collection(collection_name)
        vectors = getattr(info.config.params, "vectors", None)
        if isinstance(vectors, qm.VectorParams):
            return True
    except Exception:
        pass
    return False


def resolve_vector_name(client: QdrantClient, collection_name: str) -> str | None:
    """Named vector field if points/schema use one key; else None (default unnamed vector).

    QDRANT_VECTOR_NAME is only applied when there are no points yet *and* the collection is not
    a single default VectorParams config (avoids 'Wrong input: Not existing vector name' when
    .env says e.g. dense but the collection is unnamed).
    """
    env = (os.getenv("QDRANT_VECTOR_NAME") or "").strip() or None
    if collection_name in _VECTOR_NAME_CACHE:
        return _VECTOR_NAME_CACHE[collection_name]

    name: str | None = None
    has_points = False
    try:
        # Authoritative when points exist: stored shape (list = unnamed; dict = named)
        pts, _ = client.scroll(
            collection_name=collection_name,
            limit=1,
            with_vectors=True,
            with_payload=False,
        )
        has_points = bool(pts)
        if pts:
            raw = pts[0].vector
            if isinstance(raw, dict):
                if len(raw) == 1:
                    name = next(iter(raw.keys()))
                else:
                    name = None
            else:
                name = None
    except Exception as e:
        print(f"resolve_vector_name scroll({collection_name!r}): {e}")

    if has_points:
        _VECTOR_NAME_CACHE[collection_name] = name
        if env and env != name:
            detail = (
                "the default unnamed vector"
                if name is None
                else f"named field {name!r}"
            )
            print(f"QDRANT_VECTOR_NAME={env!r} ignored; stored points use {detail}")
        return name

    # Empty collection: optional env override, unless schema is clearly single unnamed vector
    if env:
        if _collection_top_level_is_unnamed_vector_params(client, collection_name):
            print(
                f"QDRANT_VECTOR_NAME={env!r} ignored; collection {collection_name!r} "
                "uses a single default (unnamed) vector"
            )
            name = None
        else:
            name = env
    else:
        name = _vector_name_from_collection_schema(client, collection_name)
    _VECTOR_NAME_CACHE[collection_name] = name
    return name


def get_qdrant_client() -> QdrantClient | None:
    url = (os.getenv("QDRANT_URL") or "").strip()
    if not url:
        return None
    key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
    return QdrantClient(url=url, api_key=key, timeout=60.0)


def default_collection_name() -> str:
    return (os.getenv("QDRANT_COLLECTION") or "atlas_rag").strip()


def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    try:
        client.get_collection(name)
        return
    except Exception:
        pass
    _invalidate_vector_name_cache(name)
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


def collection_info(client: QdrantClient, name: str) -> dict[str, Any]:
    try:
        c = client.get_collection(name)
        return {
            "exists": True,
            "points_count": getattr(c, "points_count", None),
            "vectors_count": getattr(c, "vectors_count", None),
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    vectors: list[list[float]],
    texts: list[str],
    source_id: str,
    extra_metadata: dict[str, Any] | None = None,
) -> int:
    if len(vectors) != len(texts):
        raise ValueError("vectors and texts length mismatch")
    meta = extra_metadata or {}
    points = []
    vname = resolve_vector_name(client, collection_name)
    for i, (vec, text) in enumerate(zip(vectors, texts)):
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}:{text[:80]}"))
        vec_payload: Any = {vname: vec} if vname else vec
        points.append(
            qm.PointStruct(
                id=pid,
                vector=vec_payload,
                payload={
                    "text": text,
                    "source_id": source_id,
                    "chunk_index": i,
                    **meta,
                },
            )
        )
    if points:
        client.upsert(collection_name=collection_name, points=points)
        if vname:
            print(f"Qdrant upsert: using named vector field {vname!r}")
        # Next search must re-resolve from scroll (first points may change vector layout detection)
        _invalidate_vector_name_cache(collection_name)
    return len(points)


def search_similar(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    limit: int = 5,
    score_threshold: float | None = None,
    qdrant_filter: qm.Filter | None = None,
) -> list[dict[str, Any]]:
    vname = resolve_vector_name(client, collection_name)
    if vname:
        print(f"Qdrant search: using named vector field {vname!r}")

    # qdrant-client 1.14+ exposes query_points; older releases used search()
    if hasattr(client, "query_points"):
        resp = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vname if vname else None,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        scored = list(getattr(resp, "points", None) or [])
    else:
        qv: Any = (
            qm.NamedVector(name=vname, vector=query_vector)
            if vname
            else query_vector
        )
        scored = client.search(
            collection_name=collection_name,
            query_vector=qv,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )
    hits = []
    for p in scored:
        pl = p.payload or {}
        hits.append(
            {
                "score": p.score,
                "text": pl.get("text", ""),
                "source_id": pl.get("source_id", ""),
                "metadata": {
                    k: v
                    for k, v in pl.items()
                    if k not in ("text", "source_id", "chunk_index")
                },
            }
        )
    return hits
