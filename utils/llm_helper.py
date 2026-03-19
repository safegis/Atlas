"""LLM wrapper for Llama.cpp and Ollama compatibility with LangChain"""

import os


class OllamaWrapper:
    """Wrapper for Ollama (local) with the same invoke interface as LlamaCppWrapper."""

    DEFAULT_SYSTEM_PROMPT = """You are SafeGIS AI, an intelligent assistant for disaster management and GIS analysis.

You are connected to a live interactive map and can perform real actions:

MAP CONTROLS:
- Search and navigate to any location worldwide
- Change map styles (streets, satellite, terrain, dark mode)
- Switch between 2D and 3D view modes

HAZARD MONITORING:
- Enable/disable live earthquake data (Philippine PHIVOLCS or Global USGS)
- Enable/disable live weather data (by province or city)
- Control multiple hazard layers simultaneously

EXPERTISE:
- Disaster management and emergency response
- GIS (Geographic Information Systems) and spatial analysis
- Hazard assessment and risk mapping
- Geospatial technology and remote sensing

When users ask questions, you can both:
1. Provide informative answers about GIS and disaster management
2. Offer to demonstrate concepts by controlling the map (e.g., "Would you like me to show earthquake data on the map?")

Be proactive in suggesting map actions when relevant to the conversation."""

    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key="ollama")
        return self._client

    def invoke(self, messages, system_prompt=None):
        """Same interface as LlamaCppWrapper: convert messages to prompt and call Ollama."""
        if isinstance(messages, list):
            prompt_parts = []
            for msg in messages:
                if hasattr(msg, "content"):
                    prompt_parts.append(msg.content)
                else:
                    prompt_parts.append(str(msg))
            prompt_text = "\n".join(prompt_parts)
        else:
            prompt_text = str(messages)

        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        print(f"LLM Prompt: {prompt_text[:200]}...")

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                max_tokens=2048,
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()
            print(f"LLM Response: {content[:200]}...")
            return content
        except Exception as e:
            print(f"LLM Error: {e}")
            return f"Error calling Ollama ({self.model}): {e}. Is Ollama running (ollama serve) and model pulled (ollama pull {self.model})?"


class LlamaCppWrapper:
    """Wrapper to make Llama compatible with LangChain's invoke pattern"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def invoke(self, messages, system_prompt=None):
        """
        Convert messages to prompt and call Llama
        
        Args:
            messages: List of messages or single message
            system_prompt: Optional custom system prompt (if None, uses default)
        """
        # Convert messages to text
        if isinstance(messages, list):
            prompt_parts = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    prompt_parts.append(msg.content)
                else:
                    prompt_parts.append(str(msg))
            prompt_text = "\n".join(prompt_parts)
        else:
            prompt_text = str(messages)
        
        print(f"LLM Prompt: {prompt_text[:200]}...")  # Debug
        
        # Use provided system prompt or default
        if system_prompt is None:
            system_prompt = """You are SafeGIS AI, an intelligent assistant for disaster management and GIS analysis.

You are connected to a live interactive map and can perform real actions:

MAP CONTROLS:
- Search and navigate to any location worldwide
- Change map styles (streets, satellite, terrain, dark mode)
- Switch between 2D and 3D view modes

HAZARD MONITORING:
- Enable/disable live earthquake data (Philippine PHIVOLCS or Global USGS)
- Enable/disable live weather data (by province or city)
- Control multiple hazard layers simultaneously

EXPERTISE:
- Disaster management and emergency response
- GIS (Geographic Information Systems) and spatial analysis
- Hazard assessment and risk mapping
- Geospatial technology and remote sensing

When users ask questions, you can both:
1. Provide informative answers about GIS and disaster management
2. Offer to demonstrate concepts by controlling the map (e.g., "Would you like me to show earthquake data on the map?")

Be proactive in suggesting map actions when relevant to the conversation."""
        
        # Call Llama with chat completion
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=2048,  # Increased for comprehensive Q&A responses
                temperature=0.3,
                top_p=0.8,
            )
            
            # Extract text from response
            if isinstance(response, dict) and "choices" in response:
                content = response["choices"][0]["message"]["content"].strip()
                print(f"LLM Response: {content[:200]}...")  # Debug
                return content
        except Exception as e:
            print(f"LLM Error: {e}")
            # Fallback to simple completion
            response = self.llm(prompt_text, max_tokens=2048, temperature=0.3)
            return response.strip()
