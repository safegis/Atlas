"""QA agent for handling general questions and explanations"""
import json
from langchain_core.messages import AIMessage
from state import AgentState


class QAAgent:
    """Handles general questions and explanations"""
    
    def __init__(self, llm):
        self.llm = llm
        # QA agent uses custom system prompt passed to LLM, not template
    
    def process(self, state: AgentState) -> AgentState:
        """Process Q&A requests"""
        messages = state["messages"]
        
        print(f"QA Agent processing {len(messages)} messages")
        
        # Extract system prompt from template
        system_prompt = """You are SafeGIS AI, an expert assistant specialized in:
- Disaster management and emergency response
- GIS (Geographic Information Systems) and mapping
- Hazard assessment and spatial analysis
- Geospatial technology and remote sensing

You are connected to a live interactive map and can perform real actions like:
- Searching locations and navigating the map
- Enabling earthquake data (Philippine PHIVOLCS or Global USGS)
- Enabling weather data (by province or city)
- Changing map styles and view modes (2D/3D)

When answering questions:
1. Provide concise, clear, and informative answers
2. Offer to demonstrate concepts by controlling the map when relevant
   Example: "Would you like me to show live earthquake data on the map?"
3. Be proactive in suggesting map actions that enhance understanding

Do not generate code or scripts.
If asked about unrelated topics, politely redirect to your expertise in GIS and disaster management."""
        
        # Use LLM with agent-specific system prompt
        response = self.llm.invoke(messages, system_prompt=system_prompt)
        
        # Wrap response in JSON format
        json_response = json.dumps({"text": response})
        
        state["messages"].append(AIMessage(content=json_response))
        
        return state
