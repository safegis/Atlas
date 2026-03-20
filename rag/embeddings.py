"""Ollama embedding API (local). Prefers /api/embed; falls back to /api/embeddings."""

from __future__ import annotations

import os
from typing import Sequence

import httpx


def _ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def _ollama_error_body(response: httpx.Response) -> str:
    try:
        j = response.json()
        return str(j.get("error", j))
    except Exception:
        return (response.text or "")[:500]


def _parse_embed_response(data: dict) -> list[float] | None:
    embs = data.get("embeddings")
    if isinstance(embs, list) and embs:
        first = embs[0]
        if isinstance(first, list):
            return [float(x) for x in first]
    one = data.get("embedding")
    if isinstance(one, list):
        return [float(x) for x in one]
    return None


def _embed_one(client: httpx.Client, host: str, model: str, text: str) -> list[float]:
    # Modern Ollama: POST /api/embed  {"model","input"}
    r = client.post(f"{host}/api/embed", json={"model": model, "input": text})
    if r.status_code == 200:
        parsed = _parse_embed_response(r.json())
        if parsed:
            return parsed

    # Legacy Ollama: /api/embeddings (some versions removed it → 404)
    r2 = client.post(
        f"{host}/api/embeddings",
        json={"model": model, "prompt": text},
    )
    if r2.status_code == 200:
        data2 = r2.json()
        vec = data2.get("embedding")
        if vec:
            return [float(x) for x in vec]

    hint = (
        f"Ollama embedding failed (/api/embed: {r.status_code}, /api/embeddings: {r2.status_code}). "
        f"Most often: embedding model not installed — run  ollama pull {model} "
        f"(and keep OLLAMA_EMBED_MODEL in sync). "
        f"Optional: set RAG_VECTOR_SIZE=768 to skip a probe at startup once the model works. "
        f"Details: embed={_ollama_error_body(r)} | legacy={_ollama_error_body(r2)}"
    )
    raise RuntimeError(hint)


def embed_texts(
    texts: Sequence[str],
    model: str | None = None,
    timeout: float = 120.0,
) -> list[list[float]]:
    model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    host = _ollama_host()
    if not texts:
        return []

    with httpx.Client(timeout=timeout) as client:
        if len(texts) > 1:
            r = client.post(
                f"{host}/api/embed",
                json={"model": model, "input": list(texts)},
            )
            if r.status_code == 200:
                data = r.json()
                embs = data.get("embeddings")
                if isinstance(embs, list) and len(embs) == len(texts):
                    return [[float(x) for x in e] for e in embs]

        return [_embed_one(client, host, model, t) for t in texts]


def embed_query(text: str, model: str | None = None) -> list[float]:
    return embed_texts([text], model=model)[0]
