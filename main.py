"""
SafeGIS AI Backend - LangGraph Multi-Agent System
FastAPI server with LangGraph orchestration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import os
import tempfile
import speech_recognition as sr
from typing import Optional, List
import uvicorn

from graph import create_agent_graph, process_message

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
# LOAD LLM
# ============================================================================

MODEL_PATH = "gemma-3n-E4B-it-Q4_0.gguf"
print("Loading Gemma model...")
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,  # Increased context window for longer conversations and responses
        n_threads=max(1, os.cpu_count() // 2),
        n_gpu_layers=0,
        verbose=False
    )
    print("Gemma model loaded!")
except Exception as e:
    print(f"Failed to load Gemma model: {e}")
    raise

# Create LangGraph agent system
print("Creating LangGraph agent system...")
agent_graph = create_agent_graph(llm)
print("Agent system ready!")

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = None
    map_state: Optional[dict] = None

class ChatResponse(BaseModel):
    response: dict
    requires_frontend: bool
    requires_clarification: bool = False
    conversation_history: List[dict]

class TranscribeResponse(BaseModel):
    transcription: str
    success: bool
    error: Optional[str] = None

# ============================================================================
# SPEECH RECOGNITION
# ============================================================================

recognizer = sr.Recognizer()

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "message": "SafeGIS AI Backend - LangGraph Multi-Agent System",
        "version": "2.0.0",
        "system": "LangGraph",
        "agents": ["router", "map_agent", "hazard_agent", "qa_agent", "clarification_agent"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "SafeGIS AI Backend is running",
        "llm_loaded": llm is not None,
        "agent_graph_loaded": agent_graph is not None,
        "system": "LangGraph Multi-Agent"
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
            request.map_state
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

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio to text using Google Speech Recognition
    Falls back to offline Sphinx if Google API is unavailable
    """
    try:
        audio_content = await audio.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_original:
            temp_original.write(audio_content)
            temp_original_path = temp_original.name

        temp_wav_path = temp_original_path.replace(".tmp", ".wav")

        try:
            # Try to convert audio to WAV using pydub
            try:
                from pydub import AudioSegment
                from pydub.utils import which

                if which("ffmpeg") is None:
                    raise ImportError("ffmpeg not found")

                audio_segment = AudioSegment.from_file(temp_original_path)
                audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
                audio_segment.export(temp_wav_path, format="wav")
                print(f"Audio converted successfully: {temp_wav_path}")

            except ImportError as e:
                print(f"pydub/ffmpeg not available: {e}")
                import shutil
                shutil.copy2(temp_original_path, temp_wav_path)
                print(f"Using original file format: {temp_wav_path}")

            with sr.AudioFile(temp_wav_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio_data = recognizer.record(source)
                print(f"Audio data recorded successfully")
                
                try:
                    text = recognizer.recognize_google(audio_data)
                    print(f"Transcription successful: {text}")
                    return TranscribeResponse(transcription=text, success=True)
                
                except sr.UnknownValueError:
                    print("Speech recognition could not understand audio")
                    return TranscribeResponse(
                        transcription="",
                        success=False,
                        error="Could not understand audio"
                    )
                
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")
                    try:
                        text = recognizer.recognize_sphinx(audio_data)
                        print(f"Offline transcription successful: {text}")
                        return TranscribeResponse(transcription=text, success=True)
                    except Exception as sphinx_error:
                        print(f"Offline recognition also failed: {sphinx_error}")
                        return TranscribeResponse(
                            transcription="",
                            success=False,
                            error=f"Recognition service error: {str(e)}"
                        )
        finally:
            for temp_path in [temp_original_path, temp_wav_path]:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                        print(f"Cleaned up temporary file: {temp_path}")
                    except Exception as cleanup_error:
                        print(f"Failed to cleanup {temp_path}: {cleanup_error}")

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return TranscribeResponse(
            transcription="",
            success=False,
            error=str(e)
        )

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
