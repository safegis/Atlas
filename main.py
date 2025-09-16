# main.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from llama_cpp import Llama
import os
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
import tempfile

# Create app only once
app = FastAPI(title="Gemma FastAPI Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
MODEL_PATH = "gemma-3n-E4B-it-Q4_0.gguf"
print("Loading Gemma model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=os.cpu_count() // 2,
    n_gpu_layers=0
)
print("Model loaded!")

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Request schema
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

@app.post("/generate")
def generate_text(request: PromptRequest):
    try:
        prompt_text = str(request.prompt).strip()
        # Truncate long prompt
        max_prompt_len = 1500
        if len(prompt_text) > max_prompt_len:
            prompt_text = prompt_text[:max_prompt_len]
        
        # Use chat completion for Gemma IT
        output = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are SafeGIS AI, an expert assistant specialized in disaster management, "
                        "GIS (Geographic Information Systems), mapping, emergency response, hazard assessment, "
                        "and spatial analysis. "
                        "Always give concise, clear, and informative answers to the specific question asked. "
                        "Do not generate code, scripts, or programming examples. "
                        "If the user asks about unrelated topics, politely refuse and redirect them to your expertise."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=request.max_tokens,
            temperature=0.3,  # Lower temperature for more focused responses
            top_p=0.8,        # More focused sampling
        )
        
        print("Raw model output:", output)
        
        # Extract response robustly
        response_text = ""
        if "choices" in output and len(output["choices"]) > 0:
            choice = output["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                response_text = choice["message"]["content"].strip()
            elif "text" in choice:
                response_text = choice["text"].strip()
        
        if not response_text:
            response_text = "Model returned empty output."
        
        return {"response": response_text}
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        # Read the uploaded audio file
        audio_content = await audio.read()
        
        # Create a temporary file to store the original audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_original:
            temp_original.write(audio_content)
            temp_original_path = temp_original.name
        
        # Create a temporary WAV file path
        temp_wav_path = temp_original_path.replace(".tmp", ".wav")
        
        try:
            # Try to convert audio to WAV using pydub (if you have ffmpeg installed)
            try:
                from pydub import AudioSegment
                from pydub.utils import which
                
                # Check if ffmpeg is available
                if which("ffmpeg") is None:
                    raise ImportError("ffmpeg not found")
                
                # Load audio file and convert to WAV
                audio_segment = AudioSegment.from_file(temp_original_path)
                
                # Convert to mono, 16kHz for better speech recognition
                audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
                
                # Export as WAV
                audio_segment.export(temp_wav_path, format="wav")
                
                print(f"Audio converted successfully: {temp_wav_path}")
                
            except ImportError as e:
                print(f"pydub/ffmpeg not available: {e}")
                # Fallback: assume the file is already in the correct format
                import shutil
                shutil.copy2(temp_original_path, temp_wav_path)
                print(f"Using original file format: {temp_wav_path}")
            
            # Use speech recognition to transcribe
            with sr.AudioFile(temp_wav_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                # Record the audio
                audio_data = recognizer.record(source)
                print(f"Audio data recorded successfully")
            
            # Try to recognize speech using Google Speech Recognition
            try:
                text = recognizer.recognize_google(audio_data)
                print(f"Transcription successful: {text}")
                return {"transcription": text, "success": True}
            except sr.UnknownValueError:
                print("Speech recognition could not understand audio")
                return {"transcription": "", "success": False, "error": "Could not understand audio"}
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                # Try offline recognition as fallback
                try:
                    # You can install vosk for offline recognition
                    # pip install vosk
                    text = recognizer.recognize_sphinx(audio_data)
                    print(f"Offline transcription successful: {text}")
                    return {"transcription": text, "success": True}
                except Exception as sphinx_error:
                    print(f"Offline recognition also failed: {sphinx_error}")
                    return {"transcription": "", "success": False, "error": f"Recognition service error: {str(e)}"}
        
        finally:
            # Clean up temporary files
            for temp_path in [temp_original_path, temp_wav_path]:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                        print(f"Cleaned up temporary file: {temp_path}")
                    except Exception as cleanup_error:
                        print(f"Failed to cleanup {temp_path}: {cleanup_error}")
    
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return {"transcription": "", "success": False, "error": str(e)}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "SafeGIS AI Backend is running"}

# Root endpoint
@app.get("/")
def root():
    return {"message": "SafeGIS AI Backend", "version": "1.0.0"}
