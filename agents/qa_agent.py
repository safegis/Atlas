"""QA agent for handling general questions and explanations"""
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from state import AgentState


class QAAgent:
    """Handles general questions and explanations"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are SafeGIS AI, an expert assistant specialized in:
- Disaster management and emergency response
- GIS (Geographic Information Systems) and mapping
- Hazard assessment and spatial analysis
- Geospatial technology and remote sensing

Provide concise, clear, and informative answers. Do not generate code or scripts.
If asked about unrelated topics, politely redirect to your expertise."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def process(self, state: AgentState) -> AgentState:
        """Process Q&A requests"""
        messages = state["messages"]
        
        print(f"QA Agent processing {len(messages)} messages")
        
        # Use LLM to generate response
        response = self.llm.invoke(messages)
        
        # Wrap response in JSON format
        json_response = json.dumps({"text": response})
        
        state["messages"].append(AIMessage(content=json_response))
        
        return state
