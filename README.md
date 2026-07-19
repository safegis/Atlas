<div align="center">
  <h1>SafeGIS AI</h1>
</div>

### 🧐 I. Overview

### Run Atlas (development)

From the directory that contains `Atlas` (e.g. the SafeGIS repo root):

```bash
cd Atlas
.venv/bin/python main.py
```

If you don’t have a venv yet (from `Atlas/`):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure as needed before running.

### RAG (Qdrant + Ollama embeddings)

Atlas retrieves **grounded excerpts** for the **QA agent** when `QDRANT_URL` is set in `.env`.

1. `pip install -r requirements.txt` (includes `qdrant-client`).
2. `ollama pull nomic-embed-text` — default embedding model (`OLLAMA_EMBED_MODEL`).
3. Set `QDRANT_URL`, `QDRANT_API_KEY` (cloud), `OLLAMA_HOST` — see `.env.example`.
4. Ingest: `POST /rag/ingest-text` or `POST /rag/ingest-file` (`.txt`/`.md`, `.pdf` via **pypdf**, images via **Tesseract OCR** — install `tesseract` on the host, e.g. `brew install tesseract`); status: `GET /rag/status`.
5. Ask questions that route to **qa_agent**; answers use retrieved chunks when relevant (`rag_used` / `rag_sources` in JSON).

### Per-conversation memory (Simulation Studio)

When the client sends **`conversation_id`** (Supabase thread UUID from Studio), Atlas:

1. **After each reply**, embeds `User: … / Assistant: …` into Qdrant with `scope=conversation_memory` and that UUID (same collection as file RAG).
2. On **QA retrieval**, search is filtered so **memory hits are only for that `conversation_id`**. Uploaded docs stay available as **`scope=knowledge_base`** (plus legacy points with no `scope`). Other chats’ memory never matches.

Requires RAG/Qdrant as above. If `conversation_id` is omitted, behavior matches older Atlas (no thread memory; KB + legacy only). New Studio threads may get an id only **after** the first save — indexing starts once the client sends that id.

Set `RAG_ENABLED=false` to turn off without removing Qdrant settings. **Never commit API keys.**

Map/pathfinder/spatial-connector flows do **not** require RAG.
