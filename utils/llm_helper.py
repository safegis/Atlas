"""LLM wrapper for Llama.cpp compatibility with LangChain"""


class LlamaCppWrapper:
    """Wrapper to make Llama compatible with LangChain's invoke pattern"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def invoke(self, messages):
        """Convert messages to prompt and call Llama"""
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
        
        # Call Llama with chat completion
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant for SafeGIS."},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=1024,  # Increased for longer Q&A responses
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
            response = self.llm(prompt_text, max_tokens=1024, temperature=0.3)
            return response.strip()
