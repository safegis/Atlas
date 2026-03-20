"""
SafeGIS AI Backend - LangGraph Multi-Agent System
FastAPI server with LangGraph orchestration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import asyncio
from typing import Optional, List
import uvicorn
import websockets

from dotenv import load_dotenv
load_dotenv()
load_dotenv(".env.local")

from graph import create_agent_graph, process_message
from utils import OllamaWrapper
from rag import build_rag_retriever_if_configured

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI(title="SafeGIS AI - LangGraph Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD LLM (Ollama - local Gemma or other model)
# ============================================================================

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma")
print(f"Using Ollama model: {OLLAMA_MODEL}")
llm_wrapper = OllamaWrapper(model=OLLAMA_MODEL)
llm = None  # No GGUF; health check uses agent_graph

# Exa.ai API key for web search
EXA_API_KEY = os.getenv("EXA_API_KEY")

# Optional: Qdrant + Ollama embeddings for RAG (QA agent)
RAG_RETRIEVER = build_rag_retriever_if_configured()

# Create LangGraph agent system
print("Creating LangGraph agent system...")
agent_graph = create_agent_graph(
    llm_wrapper=llm_wrapper,
    exa_api_key=EXA_API_KEY,
    rag_retriever=RAG_RETRIEVER,
)
print("Agent system ready!")

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = None
    map_state: Optional[dict] = None
    web_search_enabled: Optional[bool] = False
    uploaded_files: Optional[List[str]] = None
    # Rich layer list from Simulation Studio: { "name", "layerName", "sourceType"? }
    spatial_context: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    response: dict
    requires_frontend: bool
    requires_clarification: bool = False
    conversation_history: List[dict]


# --- RAG (ingest / status) ---
class RAGIngestTextRequest(BaseModel):
    text: str
    source_id: str
    metadata: Optional[dict] = None


MAX_RAG_TEXT_CHARS = 1_500_000
MAX_RAG_FILE_BYTES = 5_000_000

# ElevenLabs API key for real-time speech-to-text and TTS (keep server-side only)
ELEVENLABS_API_KEY = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
ELEVENLABS_WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
ELEVENLABS_VOICE_ID = (os.getenv("ELEVENLABS_VOICE_ID") or "21m00Tcm4TlvDq8ikWAM").strip()  # Rachel

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "message": "SafeGIS AI Backend - LangGraph Multi-Agent System",
        "version": "2.0.0",
        "system": "LangGraph",
        "agents": [
            "router",
            "map_agent",
            "hazard_agent",
            "qa_agent",
            "clarification_agent",
            "spatial_data_agent",
            "pathfinder_agent",
            "ui_agent",
            "exposure_assessment_agent",
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "SafeGIS AI Backend is running",
        "llm_loaded": agent_graph is not None,
        "agent_graph_loaded": agent_graph is not None,
        "system": "LangGraph Multi-Agent",
        "rag_enabled": RAG_RETRIEVER is not None,
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint using LangGraph multi-agent system
    
    Processes user messages through specialized agents:
    - Router Agent: Determines which agent to use
    - Map Agent: Handles location, style, view mode
    - Hazard Agent: Handles earthquake/weather monitoring
    - QA Agent: Handles general questions
    - Clarification Agent: Handles ambiguous requests
    """
    try:
        # Process message through agent graph
        result = process_message(
            agent_graph,
            request.message,
            request.conversation_history,
            request.map_state,
            request.web_search_enabled,
            request.uploaded_files,
            request.spatial_context,
        )
        
        # Format conversation history safely
        formatted_history = []
        for msg in result["conversation_history"]:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                formatted_history.append({
                    "role": msg.type,
                    "content": msg.content
                })
            elif isinstance(msg, dict):
                formatted_history.append(msg)
        
        return ChatResponse(
            response=result["response"],
            requires_frontend=result.get("requires_frontend", False),
            requires_clarification=result.get("requires_clarification", False),
            conversation_history=formatted_history
        )
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/status")
def rag_status():
    """Qdrant + embedding model configuration (no secrets)."""
    from rag.store import collection_info, default_collection_name, get_qdrant_client

    if RAG_RETRIEVER is None:
        return {
            "enabled": False,
            "reason": "Set QDRANT_URL (and QDRANT_API_KEY for cloud) or fix init errors; optional RAG_ENABLED=false to skip.",
            "embed_model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            "ollama_host": os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        }
    c = get_qdrant_client()
    name = default_collection_name()
    info = collection_info(c, name) if c else {"exists": False}
    resolved_vec = None
    try:
        from rag.store import resolve_vector_name

        if c:
            resolved_vec = resolve_vector_name(c, name)
    except Exception:
        pass
    return {
        "enabled": True,
        "collection": name,
        "embed_model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        "ollama_host": os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        "top_k": int(os.getenv("RAG_TOP_K", "5")),
        "qdrant": info,
        "qdrant_vector_field": resolved_vec,
        "qdrant_vector_env": (os.getenv("QDRANT_VECTOR_NAME") or "").strip() or None,
    }


@app.get("/rag/preview")
def rag_preview(query: str = Query(..., min_length=1, description="Test retrieval; no LLM")):
    """Return raw Qdrant hits for debugging (embed query + vector search)."""
    if RAG_RETRIEVER is None:
        raise HTTPException(status_code=503, detail="RAG not configured")
    q = query.strip()
    try:
        hits = RAG_RETRIEVER.retrieve(q)
        return {
            "query": q,
            "hit_count": len(hits),
            "hits": [
                {
                    "score": h.get("score"),
                    "source_id": h.get("source_id"),
                    "text_preview": (h.get("text") or "")[:400],
                }
                for h in hits
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/ingest-text")
def rag_ingest_text(body: RAGIngestTextRequest):
    """Chunk, embed (Ollama), and upsert into Qdrant."""
    if RAG_RETRIEVER is None:
        raise HTTPException(status_code=503, detail="RAG not configured (see /rag/status)")
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    if len(body.text) > MAX_RAG_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"text too large (max {MAX_RAG_TEXT_CHARS} characters)",
        )
    sid = (body.source_id or "anonymous").strip()
    try:
        result = RAG_RETRIEVER.ingest_text(
            body.text, sid, metadata=body.metadata or None
        )
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/ingest-file")
async def rag_ingest_file(
    file: UploadFile = File(...),
    source_id: Optional[str] = Query(None, description="Logical document id; defaults to filename"),
):
    """Upload text, Markdown, PDF, or image (OCR) and ingest into Qdrant."""
    if RAG_RETRIEVER is None:
        raise HTTPException(status_code=503, detail="RAG not configured (see /rag/status)")
    raw = await file.read()
    if len(raw) > MAX_RAG_FILE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"file too large (max {MAX_RAG_FILE_BYTES} bytes)",
        )
    from rag.file_extract import extract_text_from_upload

    fname = (file.filename or "upload").strip()
    try:
        text, extract_meta = extract_text_from_upload(fname, raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    sid = (source_id or fname or "upload").strip()
    meta = {"filename": fname, **extract_meta}
    try:
        result = RAG_RETRIEVER.ingest_text(
            text,
            sid,
            metadata=meta,
        )
        return {"ok": True, **result, "extraction": extract_meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.websocket("/transcribe-ws")
async def transcribe_ws(websocket: WebSocket):
    """
    WebSocket proxy to ElevenLabs real-time speech-to-text.
    Client sends input_audio_chunk messages; server forwards partial/committed transcripts.
    """
    await websocket.accept()
    if not ELEVENLABS_API_KEY:
        await websocket.send_json({
            "message_type": "error",
            "error": "ELEVENLABS_API_KEY not configured",
        })
        await websocket.close()
        return

    additional_headers = {"xi-api-key": ELEVENLABS_API_KEY}
    # 48kHz from browser (no resampling); VAD auto-commits on silence; English only
    url = f"{ELEVENLABS_WS_URL}?audio_format=pcm_48000&commit_strategy=vad&language_code=en"

    async def forward_client_to_elevenlabs():
        try:
            async with websockets.connect(url, additional_headers=additional_headers) as eleven_ws:
                async def forward_elevenlabs_to_client():
                    try:
                        async for raw in eleven_ws:
                            try:
                                msg = json.loads(raw)
                                await websocket.send_json(msg)
                            except Exception as e:
                                print(f"ElevenLabs->client forward error: {e}")
                    except asyncio.CancelledError:
                        pass

                task = asyncio.create_task(forward_elevenlabs_to_client())
                try:
                    async for raw in websocket.iter_text():
                        try:
                            # Forward client JSON to ElevenLabs
                            await eleven_ws.send(raw)
                        except Exception as e:
                            print(f"Client->ElevenLabs forward error: {e}")
                            break
                finally:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        except websockets.exceptions.InvalidStatusCode as e:
            await websocket.send_json({
                "message_type": "error",
                "error": f"ElevenLabs connection failed: {e}",
            })
        except Exception as e:
            print(f"Transcribe WebSocket error: {e}")
            try:
                await websocket.send_json({
                    "message_type": "error",
                    "error": str(e),
                })
            except Exception:
                pass

    try:
        await forward_client_to_elevenlabs()
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ============================================================================
# ELEVENLABS TTS (conversational mode – Atlas speaks)
# ============================================================================

class TTSRequest(BaseModel):
    text: str

@app.get("/tts-verify")
async def tts_verify():
    """
    Verify ELEVENLABS_API_KEY by calling ElevenLabs /v1/user.
    Returns 200 if key is valid, 401/500 with detail if not.
    """
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY not set")
    import httpx
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://api.elevenlabs.io/v1/user", headers=headers)
            if resp.status_code == 200:
                return {"ok": True, "message": "API key is valid"}
            return Response(
                content=resp.text,
                status_code=resp.status_code,
                media_type="application/json",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech via ElevenLabs. Returns MP3 bytes for conversational mode.
    """
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY not configured")
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    import httpx
    url = f"{ELEVENLABS_TTS_URL}/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {"text": request.text.strip()}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return Response(
                content=resp.content,
                media_type="audio/mpeg",
                headers={"Content-Disposition": "inline"},
            )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            err_body = e.response.text
            print("ElevenLabs TTS 401 Unauthorized. Response:", err_body)
            print("Your key works for /v1/user but not TTS. In the key's permissions, enable any 'Speech' / 'Text-to-Speech' / 'Generate' scope.")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LEGACY ENDPOINTS (for backward compatibility during migration)
# ============================================================================

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

@app.post("/generate")
def generate_text(request: PromptRequest):
    """
    Legacy endpoint - redirects to /chat
    Kept for backward compatibility during migration
    """
    try:
        result = process_message(agent_graph, request.prompt)
        
        response_text = result["response"].get("text", "")
        if not response_text and "tool" in result["response"]:
            response_text = f"Action required: {result['response']['tool']}"
        
        return {"response": response_text}
    
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
