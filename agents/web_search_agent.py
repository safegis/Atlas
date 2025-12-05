"""Web search agent using Exa.ai for real-time information"""
import json
from langchain_core.messages import AIMessage
from state import AgentState
from exa_py import Exa


class WebSearchAgent:
    """Handles web search queries using Exa.ai"""
    
    def __init__(self, llm, exa_api_key: str):
        self.llm = llm
        self.exa = Exa(api_key=exa_api_key)
    
    def process(self, state: AgentState) -> AgentState:
        """Process web search requests"""
        messages = state["messages"]
        last_message = messages[-1].content
        
        print(f"Web Search Agent processing: {last_message}")
        
        # First, check if the question is relevant to our domain
        relevance_check_prompt = """Analyze if this question is related to GIS, disaster management, or geospatial topics.

RELEVANT topics include:
- GIS software and companies (ESRI, ArcGIS, QGIS, Mapbox, etc.)
- Disaster management and emergency response
- Mapping, cartography, and spatial analysis
- Hazards (earthquakes, floods, weather, climate)
- Remote sensing and satellite imagery
- Urban planning and infrastructure
- Environmental monitoring
- Geospatial technology and tools

NOT_RELEVANT topics include:
- Entertainment (anime, movies, TV shows, celebrities)
- Sports and games (unless related to disaster/GIS)
- General trivia unrelated to geography/disasters
- Politics (unless disaster-related)
- Shopping, cooking, fashion, etc.

Question: {question}

Respond with ONLY "RELEVANT" or "NOT_RELEVANT"."""
        
        try:
            relevance_response = self.llm.invoke(
                relevance_check_prompt.format(question=last_message),
                system_prompt="You are a topic classifier for GIS and disaster management. Respond with only 'RELEVANT' or 'NOT_RELEVANT'."
            ).strip().upper()
            
            print(f"Relevance check: {relevance_response}")
            
            if "NOT_RELEVANT" in relevance_response or "NOT RELEVANT" in relevance_response:
                # Question is off-topic, decline politely
                decline_message = json.dumps({
                    "text": "I'm specialized in GIS, disaster management, and geospatial analysis. I can only answer questions related to these topics.\n\nPlease ask me about:\n• Mapping and geographic information systems\n• Disaster response and emergency management\n• Hazard assessment (earthquakes, weather, floods, etc.)\n• Spatial analysis and remote sensing\n• Climate monitoring and environmental analysis",
                    "has_citations": False
                })
                state["messages"].append(AIMessage(content=decline_message))
                return state
        
        except Exception as e:
            print(f"Error in relevance check: {e}")
            # If relevance check fails, proceed with caution
        
        try:
            # Search using Exa.ai
            print(f"Searching Exa.ai for: {last_message}")
            search_results = self.exa.search_and_contents(
                last_message,
                type="neural",  # Use neural search for better relevance
                num_results=5,
                text=True
            )
            
            # Format search results for LLM
            context = self._format_search_results(search_results)
            
            # Use LLM to synthesize answer with citations
            system_prompt = """You are SafeGIS AI, an expert assistant specialized in:
- Disaster management and emergency response
- GIS (Geographic Information Systems) and mapping
- Hazard assessment and spatial analysis
- Geospatial technology and remote sensing
- Climate and weather monitoring
- Earthquake and seismic analysis

You have access to real-time web search results to provide accurate, up-to-date information.

IMPORTANT RULES:
1. ONLY answer questions related to your expertise areas listed above
2. If asked about unrelated topics (entertainment, sports, general trivia, etc.), politely decline and say:
   "I'm specialized in GIS, disaster management, and geospatial analysis. I can only answer questions related to these topics. Please ask me about mapping, hazards, emergency response, or spatial analysis."
3. Include inline citations using [1], [2], [3] etc. throughout your answer
4. DO NOT include a "Sources:" section at the end - citations will be displayed automatically below
5. Provide comprehensive, well-cited answers for relevant topics
6. CRITICAL: Respond with PLAIN TEXT ONLY. Do NOT format your response as JSON. Do NOT include {"text": ...} in your response.

Example format:
"According to recent studies [1], disaster management systems have improved significantly [2]. The technology has evolved rapidly [3]."

End your response naturally without listing sources. Write in plain text, not JSON."""

            # Combine user query with search context
            prompt = f"""User Question: {last_message}

Web Search Results:
{context}

Provide a comprehensive answer using the search results above. Include inline citations [1], [2], etc. throughout your answer. 

IMPORTANT: Write your answer in PLAIN TEXT. Do NOT format as JSON. Do NOT include {{"text": ...}}. Just write the answer directly."""

            response = self.llm.invoke(prompt, system_prompt=system_prompt)
            
            # Wrap response with citations
            json_response = json.dumps({
                "text": response,
                "has_citations": True,
                "search_results": self._format_citations(search_results)
            })
            
            state["messages"].append(AIMessage(content=json_response))
            
        except Exception as e:
            print(f"Error in web search: {e}")
            # Fallback to regular QA without search
            error_response = json.dumps({
                "text": f"I encountered an error while searching the web: {str(e)}. Let me answer based on my knowledge instead.\n\n" + 
                        "Please try your question again or ask something else.",
                "has_citations": False
            })
            state["messages"].append(AIMessage(content=error_response))
        
        return state
    
    def _format_search_results(self, results) -> str:
        """Format Exa search results for LLM context"""
        formatted = []
        
        for i, result in enumerate(results.results, 1):
            formatted.append(f"[{i}] {result.title}")
            formatted.append(f"URL: {result.url}")
            
            # Add text content preview
            if hasattr(result, 'text') and result.text:
                # Use first 800 chars of text for context
                text_preview = result.text[:800] + "..." if len(result.text) > 800 else result.text
                formatted.append(f"Content: {text_preview}")
            
            formatted.append("")  # Empty line between results
        
        return "\n".join(formatted)
    
    def _format_citations(self, results) -> list:
        """Format citations for frontend display"""
        citations = []
        
        for i, result in enumerate(results.results, 1):
            citations.append({
                "number": i,
                "title": result.title,
                "url": result.url,
                "author": getattr(result, 'author', None),
                "published_date": getattr(result, 'published_date', None)
            })
        
        return citations
