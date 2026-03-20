<div align="center">
  <h1>SafeGIS AI</h1>
</div>

### 🧐 I. Overview

### RAG (Qdrant + Ollama embeddings)

Atlas retrieves **grounded excerpts** for the **QA agent** when `QDRANT_URL` is set in `.env`.

1. `pip install -r requirements.txt` (includes `qdrant-client`).
2. `ollama pull nomic-embed-text` — default embedding model (`OLLAMA_EMBED_MODEL`).
3. Set `QDRANT_URL`, `QDRANT_API_KEY` (cloud), `OLLAMA_HOST` — see `.env.example`.
4. Ingest: `POST /rag/ingest-text` or `POST /rag/ingest-file` (`.txt`/`.md`, `.pdf` via **pypdf**, images via **Tesseract OCR** — install `tesseract` on the host, e.g. `brew install tesseract`); status: `GET /rag/status`.
5. Ask questions that route to **qa_agent**; answers use retrieved chunks when relevant (`rag_used` / `rag_sources` in JSON).

Set `RAG_ENABLED=false` to turn off without removing Qdrant settings. **Never commit API keys.**

Map/pathfinder/spatial-connector flows do **not** require RAG.
