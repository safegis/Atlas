# \SafeGIS\SafeGIS-AI\main.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from llama_cpp import Llama
import os
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
import tempfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import Dict, List, Tuple

# SentenceTransformers + fallback vectorizer
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from sklearn.feature_extraction.text import TfidfVectorizer

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

# Request schemas
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

class IntentRequest(BaseModel):
    prompt: str

# Load intent examples from JSON file (function defined early so fallback can use them)
def load_intent_examples() -> Dict[str, List[str]]:
    """Load intent examples from intent_examples.json file"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "intent_examples.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            intent_examples = json.load(f)
        print(f"Loaded intent examples from {json_path}")
        print(f"Available intents: {list(intent_examples.keys())}")
        return intent_examples
    except FileNotFoundError:
        print("intent_examples.json not found, using fallback examples")
        return {
            "map": ["show me location", "find city", "navigate to"],
            "view": ["3d mode", "2d view", "satellite"],
            "earthquake": ["earthquake data", "seismic activity"],
            "volcano": ["volcano list", "volcanic activity"],
            "activefaults": ["active faults", "fault lines"],
            "congestion": ["traffic congestion", "congestion data"],
            "qa": ["what is GIS", "explain disaster"]
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing intent_examples.json: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error loading intent examples: {e}")
        return {}

# Load intent examples (needed before building fallback TF-IDF)
INTENT_EXAMPLES = load_intent_examples()

# -------------------------
# Load main LLM model (Gemma)
# -------------------------
MODEL_PATH = "gemma-3n-E4B-it-Q4_0.gguf"
print("Loading Gemma model...")
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=max(1, os.cpu_count() // 2),
        n_gpu_layers=0
    )
    print("Gemma model loaded!")
except Exception as e:
    print(f"Failed to load Gemma model: {e}")
    raise

# -------------------------
# Load embedding model (MiniLM via SentenceTransformers) or fallback
# -------------------------
embedding_model = None
fallback_vectorizer = None
EMBEDDING_DIM = None

print("Loading MiniLM embedding model (SentenceTransformer)...")
if SentenceTransformer is not None:
    try:
        # This will download the model if not present locally (requires internet)
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
        print(f"SentenceTransformer MiniLM loaded (dim={EMBEDDING_DIM})")
    except Exception as e:
        print(f"Failed to load SentenceTransformer model: {e}")
        embedding_model = None
else:
    print("sentence-transformers package not available in this environment.")

# If SentenceTransformer failed, create a TF-IDF fallback trained on the intent examples
if embedding_model is None:
    print("Building TF-IDF fallback vectorizer from intent examples...")
    try:
        # flatten all example texts
        corpus = []
        for examples in INTENT_EXAMPLES.values():
            corpus.extend(examples)
        if len(corpus) == 0:
            corpus = [""]  # avoid empty corpus issues
        fallback_vectorizer = TfidfVectorizer()
        fallback_vectorizer.fit(corpus)
        EMBEDDING_DIM = len(fallback_vectorizer.get_feature_names_out())
        print(f"TF-IDF fallback ready (dim={EMBEDDING_DIM})")
    except Exception as e:
        print(f"Failed to build TF-IDF fallback: {e}")
        fallback_vectorizer = None
        # last resort dimension
        EMBEDDING_DIM = 384

# -------------------------
# Initialize speech recognizer
# -------------------------
recognizer = sr.Recognizer()

# -------------------------
# Embedding utility functions
# -------------------------
def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Return embeddings for the given list of texts.
    Uses SentenceTransformer if available, otherwise TF-IDF fallback.
    """
    if embedding_model is not None:
        try:
            # convert_to_numpy ensures we get a numpy array
            embs = embedding_model.encode(texts, convert_to_numpy=True)
            # Ensure shape (n_texts, EMBEDDING_DIM)
            return embs
        except Exception as e:
            print(f"Error getting embeddings from SentenceTransformer: {e}")
            # Fall through to fallback
    # Fallback path
    if fallback_vectorizer is not None:
        try:
            arr = fallback_vectorizer.transform(texts).toarray()
            # If TF-IDF dim differs from EMBEDDING_DIM, pad/truncate to EMBEDDING_DIM
            if arr.shape[1] != EMBEDDING_DIM:
                if arr.shape[1] < EMBEDDING_DIM:
                    pad_width = EMBEDDING_DIM - arr.shape[1]
                    arr = np.pad(arr, ((0,0),(0,pad_width)), mode='constant', constant_values=0.0)
                else:
                    arr = arr[:, :EMBEDDING_DIM]
            return arr
        except Exception as e:
            print(f"Error with fallback TF-IDF embeddings: {e}")
    # Last resort: simple bag-of-words on intent vocabulary (very small)
    print("Using last-resort bag-of-words fallback.")
    import re
    all_words = set()
    for examples in INTENT_EXAMPLES.values():
        for example in examples:
            words = re.findall(r'\w+', example.lower())
            all_words.update(words)
    vocab = sorted(list(all_words))
    vocab_size = max(1, len(vocab))
    embeddings = []
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        vector = [0.0] * vocab_size
        for w in words:
            if w in vocab:
                vector[vocab.index(w)] = 1.0
        # pad/truncate to EMBEDDING_DIM
        if vocab_size < EMBEDDING_DIM:
            vector.extend([0.0] * (EMBEDDING_DIM - vocab_size))
        else:
            vector = vector[:EMBEDDING_DIM]
        embeddings.append(vector)
    return np.array(embeddings)

# -------------------------
# Pre-compute intent centroids
# -------------------------
print("Computing intent embeddings...")
intent_embeddings: Dict[str, np.ndarray] = {}
try:
    for intent, examples in INTENT_EXAMPLES.items():
        print(f"Computing embeddings for {intent}...")
        if isinstance(examples, list) and len(examples) > 0:
            embs = get_embeddings(examples)
            if embs is not None and embs.shape[0] > 0:
                centroid = np.mean(embs, axis=0)
            else:
                centroid = np.zeros(EMBEDDING_DIM, dtype=float)
        else:
            centroid = np.zeros(EMBEDDING_DIM, dtype=float)
        intent_embeddings[intent] = centroid
    print("Intent embeddings computed successfully!")
except Exception as e:
    print(f"Error computing intent embeddings: {e}")
    intent_embeddings = {}

# -------------------------
# Intent classification
# -------------------------
def classify_intent(text: str, threshold: float = 0.3) -> str:
    """
    Classify intent using embedding similarity
    Returns the top intent if similarity >= threshold, otherwise 'qa'.
    """
    if not intent_embeddings:
        print("No intent embeddings available, using fallback classification")
        return classify_intent_fallback(text)
    try:
        user_embeddings = get_embeddings([text])
        if user_embeddings is None or user_embeddings.shape[0] == 0:
            return "qa"
        user_embedding = user_embeddings[0]
        similarities = {}
        for intent, intent_embedding in intent_embeddings.items():
            try:
                if len(user_embedding) != len(intent_embedding):
                    print(f"Shape mismatch for {intent}: {len(user_embedding)} vs {len(intent_embedding)}")
                    similarities[intent] = 0.0
                    continue
                sim = cosine_similarity([user_embedding], [intent_embedding])[0][0]
                similarities[intent] = float(sim)
            except Exception as e:
                print(f"Error calculating similarity for {intent}: {e}")
                similarities[intent] = 0.0
        if not similarities:
            return "qa"
        best_intent = max(similarities, key=similarities.get)
        best_score = similarities[best_intent]
        print(f"Intent classification for '{text}': {best_intent} (score: {best_score:.3f})")
        return best_intent if best_score >= threshold else "qa"
    except Exception as e:
        print(f"Error in intent classification: {e}")
        return classify_intent_fallback(text)

def classify_intent_fallback(text: str) -> str:
    """Simple keyword-based fallback intent classification"""
    text_lower = text.lower()
    if any(word in text_lower for word in ["earthquake", "seismic", "quake"]):
        return "earthquake"
    elif any(word in text_lower for word in ["volcano", "volcanic"]):
        return "volcano"
    elif any(word in text_lower for word in ["fault", "geological"]):
        return "activefaults"
    elif any(word in text_lower for word in ["congestion", "traffic jam", "traffic congestion"]):
        return "congestion"
    elif any(word in text_lower for word in ["3d", "2d", "view", "perspective", "satellite", "terrain"]):
        return "view"
    elif any(word in text_lower for word in ["show", "find", "go to", "locate", "search", "navigate"]):
        return "map"
    else:
        return "qa"

# -------------------------
# /generate endpoint (Gemma chat)
# -------------------------
@app.post("/generate")
def generate_text(request: PromptRequest):
    try:
        prompt_text = str(request.prompt).strip()
        max_prompt_len = 1500
        if len(prompt_text) > max_prompt_len:
            prompt_text = prompt_text[:max_prompt_len]

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
            temperature=0.3,
            top_p=0.8,
        )

        print("Raw model output:", output)

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

# -------------------------
# /classify-intent endpoint
# -------------------------
@app.post("/classify-intent")
def classify_user_intent(request: IntentRequest):
    try:
        intent = classify_intent(request.prompt)

        similarities = {}
        if intent_embeddings:
            try:
                user_embedding = get_embeddings([request.prompt])[0]
                for intent_name, intent_embedding in intent_embeddings.items():
                    try:
                        if len(user_embedding) == len(intent_embedding):
                            similarity = cosine_similarity([user_embedding], [intent_embedding])[0][0]
                            similarities[intent_name] = float(similarity)
                        else:
                            similarities[intent_name] = 0.0
                    except Exception:
                        similarities[intent_name] = 0.0
            except Exception as e:
                print(f"Error computing similarities for debugging: {e}")

        return {
            "intent": intent,
            "similarities": similarities,
            "prompt": request.prompt,
            "embedding_available": embedding_model is not None
        }
    except Exception as e:
        print(f"Error in classify_user_intent: {e}")
        return {
            "error": str(e),
            "intent": "qa",
            "similarities": {},
            "prompt": request.prompt,
            "embedding_available": False
        }

# -------------------------
# /transcribe endpoint (unchanged logic)
# -------------------------
@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        audio_content = await audio.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_original:
            temp_original.write(audio_content)
            temp_original_path = temp_original.name

        temp_wav_path = temp_original_path.replace(".tmp", ".wav")

        try:
            # Try to convert audio to WAV using pydub (if you have ffmpeg installed)
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
                    return {"transcription": text, "success": True}
                except sr.UnknownValueError:
                    print("Speech recognition could not understand audio")
                    return {"transcription": "", "success": False, "error": "Could not understand audio"}
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")
                    try:
                        text = recognizer.recognize_sphinx(audio_data)
                        print(f"Offline transcription successful: {text}")
                        return {"transcription": text, "success": True}
                    except Exception as sphinx_error:
                        print(f"Offline recognition also failed: {sphinx_error}")
                        return {"transcription": "", "success": False, "error": f"Recognition service error: {str(e)}"}
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
        return {"transcription": "", "success": False, "error": str(e)}

# -------------------------
# Health & Root
# -------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "SafeGIS AI Backend is running",
        "embedding_model_loaded": embedding_model is not None,
        "intent_embeddings_computed": len(intent_embeddings) > 0,
        "intent_examples_loaded": len(INTENT_EXAMPLES) > 0
    }

@app.get("/")
def root():
    return {"message": "SafeGIS AI Backend", "version": "1.0.0"}
