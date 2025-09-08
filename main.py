from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import os

from fastapi.middleware.cors import CORSMiddleware

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
MODEL_PATH = "gemma-3-4b-it-q4_0.gguf"

print("Loading Gemma model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=os.cpu_count() // 2,
    n_gpu_layers=0
)
print("Model loaded!")

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
                        "You are SafeGIS AI, specialized in disaster management, "
                        "GIS, and mapping. Answer only in these domains."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=request.max_tokens,
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
            response_text = "⚠️ Model returned empty output."

        return {"response": response_text}

    except Exception as e:
        return {"error": str(e)}
