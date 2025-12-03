"""Router agent for directing requests to specialized agents"""
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from state import AgentState


class RouterAgent:
    """Routes user requests to appropriate specialized agent"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a routing agent for SafeGIS, a disaster management GIS platform.
            
Your job is to analyze user requests and route them to the appropriate specialized agent:

1. **map_agent** - For location searches, navigation, map style changes, view mode switching
   Examples: "show me Tokyo", "switch to satellite view", "change to 3D mode"

2. **hazard_agent** - For earthquake/weather data, hazard monitoring, live data
   Examples: "enable earthquake data", "show weather", "turn on seismic monitoring"

3. **qa_agent** - For questions about GIS, disasters, explanations, general knowledge
   Examples: "what is GIS?", "explain earthquakes", "how does mapping work?"

4. **clarification_agent** - When the request is ambiguous or unclear
   Examples: "make it darker" (which style?), "show hazards" (which type?)

Analyze the user's message and respond with ONLY the agent name: map_agent, hazard_agent, qa_agent, or clarification_agent"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def route(self, state: AgentState) -> str:
        """Determine which agent should handle the request"""
        messages = state["messages"]
        
        # Get last user message
        last_message = messages[-1].content if messages else ""
        msg_lower = last_message.lower().strip()
        
        print(f"Router analyzing: {last_message}")
        print(f"Clarification needed: {state.get('clarification_needed')}")
        print(f"Pending action: {state.get('pending_action')}")
        
        # Check if we're in a clarification flow
        if state.get("clarification_needed") or state.get("pending_action"):
            print("Routing to: clarification_agent (has pending action)")
            return "clarification_agent"
        
        # Check if this looks like a follow-up response (yes/no or specific options)
        potential_responses = ["yes", "yeah", "yep", "sure", "ok", "okay", "no", "nope", "nah", "cancel",
                              "philippines", "philippine", "phivolcs", "local", 
                              "global", "usgs", "worldwide", "world",
                              "both", "all",
                              "province", "provincial", "region",
                              "city", "municipality", "municipal",
                              "dark", "navigation", "navigation night",
                              "1", "2", "3"]
        
        if msg_lower in potential_responses:
            # Check if previous message was a clarification
            print(f"Detected potential clarification response, checking previous messages ({len(messages)} total)")
            if len(messages) >= 2:
                # Look at the last AI message
                for i in range(len(messages) - 2, -1, -1):
                    msg = messages[i]
                    print(f"Checking message {i}: type={type(msg)}, hasattr content={hasattr(msg, 'content')}")
                    
                    # Handle both Message objects and dicts
                    content = None
                    msg_type = None
                    
                    if hasattr(msg, 'content'):
                        content = msg.content
                        msg_type = getattr(msg, 'type', None)
                    elif isinstance(msg, dict):
                        content = msg.get('content')
                        msg_type = msg.get('role')
                    
                    print(f"Message type: {msg_type}, content preview: {content[:100] if content else None}")
                    
                    if msg_type in ['ai', 'assistant'] and content:
                        try:
                            prev_content = json.loads(content)
                            print(f"Parsed content type: {prev_content.get('type')}")
                            if prev_content.get("type") == "clarification":
                                print("✓ Routing to: clarification_agent (follow-up to clarification)")
                                print(f"Setting pending action: {prev_content.get('suggested_action')}")
                                # Set the pending action from the clarification
                                state["pending_action"] = prev_content
                                state["clarification_needed"] = True
                                return "clarification_agent"
                        except Exception as e:
                            print(f"Error parsing message content: {e}")
                        break  # Only check the last AI message
        
        # Use simple keyword-based routing (more reliable than LLM for this)
        # Hazard Agent - Earthquake, weather, live hazards (CHECK FIRST - higher priority)
        hazard_keywords = ["earthquake", "seismic", "weather", "temperature", "climate",
                          "enable", "disable", "turn on", "turn off", "monitoring",
                          "hazard", "hazards", "live"]
        
        # Map Agent - Location, style, view
        map_keywords = ["show", "find", "go to", "navigate", "search", "where is", "locate", 
                       "satellite", "dark", "light", "outdoors", "navigation", "style", "default",
                       "2d", "3d", "view", "perspective", "map", "darker", "lighter"]
        
        # Check for hazard keywords FIRST (higher priority than map)
        if any(keyword in msg_lower for keyword in hazard_keywords):
            if not any(q in msg_lower for q in ["what is", "what are", "explain", "how does", "tell me about"]):
                print("Routing to: hazard_agent")
                return "hazard_agent"
        
        # Check for map keywords
        if any(keyword in msg_lower for keyword in map_keywords):
            # But not if it's asking "what is" (that's Q&A)
            if not any(q in msg_lower for q in ["what is", "what are", "explain", "how does", "tell me about"]):
                print("Routing to: map_agent")
                return "map_agent"
        
        # Default to QA agent for questions
        print("Routing to: qa_agent")
        return "qa_agent"
