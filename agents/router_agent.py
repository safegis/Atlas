"""Router agent for directing requests to specialized agents"""
import json
from state import AgentState


class RouterAgent:
    """Routes user requests to appropriate specialized agent using keyword-based logic"""
    
    def __init__(self, llm):
        self.llm = llm
        # Router uses rule-based logic, not LLM prompts
    
    def route(self, state: AgentState) -> str:
        """Determine which agent should handle the request"""
        messages = state["messages"]
        
        # Get last user message
        last_message = messages[-1].content if messages else ""
        msg_lower = last_message.lower().strip()
        
        print(f"Router analyzing: {last_message}")
        print(f"Clarification needed: {state.get('clarification_needed')}")
        print(f"Pending action: {state.get('pending_action')}")
        print(f"Web search enabled: {state.get('web_search_enabled', False)}")
        
        # Check if this is a Q&A question (not a control request)
        qa_indicators = [
            "what is", "what are", "what's", "explain", "how does", "tell me about",
            "do you know", "are you familiar", "can you tell me", "have you heard",
            "who is", "who are", "why is", "why are", "when is", "when are",
            "describe", "define", "meaning of"
        ]
        
        is_question = any(indicator in msg_lower for indicator in qa_indicators)
        
        # Check if web search is enabled - route to web search for Q&A questions
        if state.get("web_search_enabled", False):
            # Check if this is a Q&A type question (not map/hazard control)
            hazard_control_keywords = ["enable", "disable", "turn on", "turn off", "monitoring"]
            map_action_keywords = ["show", "find", "go to", "navigate", "zoom", "fly to", "take me to",
                                  "change to", "switch to", "set to"]
            
            has_hazard_control = any(keyword in msg_lower for keyword in hazard_control_keywords)
            has_map_action = any(keyword in msg_lower for keyword in map_action_keywords)
            
            # If it's a question OR not a control request, use web search
            if is_question or (not has_hazard_control and not has_map_action):
                print("Routing to: web_search_agent (web search enabled)")
                return "web_search_agent"
        
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
        
        # Map Agent - Location, style, view (ACTION keywords only)
        map_action_keywords = ["show", "find", "go to", "navigate", "search", "where is", "locate", 
                              "change to", "switch to", "set to", "use", "apply",
                              "zoom", "fly to", "take me to"]
        
        # Check for hazard keywords FIRST (higher priority than map)
        if any(keyword in msg_lower for keyword in hazard_keywords):
            if not is_question:
                print("Routing to: hazard_agent")
                return "hazard_agent"
        
        # Check for map ACTION keywords (not just mentions of map-related terms)
        if any(keyword in msg_lower for keyword in map_action_keywords):
            if not is_question:
                print("Routing to: map_agent")
                return "map_agent"
        
        # Check for specific map style/view change requests
        style_keywords = ["satellite", "dark", "light", "outdoors", "navigation", "2d", "3d"]
        change_keywords = ["change", "switch", "set", "use", "apply", "make it"]
        
        has_style = any(keyword in msg_lower for keyword in style_keywords)
        has_change = any(keyword in msg_lower for keyword in change_keywords)
        
        if has_style and has_change and not is_question:
            print("Routing to: map_agent")
            return "map_agent"
        
        # Default to QA agent for questions
        print("Routing to: qa_agent")
        return "qa_agent"
