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
        
        # Check if we're in a clarification flow - but check the tool type first
        if state.get("clarification_needed") or state.get("pending_action"):
            pending = state.get("pending_action")
            if pending and isinstance(pending, dict):
                # Check if it's a pathfinder clarification
                if pending.get('type') == 'pathfinder_awaiting_locations':
                    print("Routing to: pathfinder_agent (has pending pathfinder action)")
                    return "pathfinder_agent"
                
                suggested_action = pending.get('suggested_action', {})
                tool = suggested_action.get('tool')
                
                # Route based on the tool in the pending action
                if tool == "run_exposure_analysis" or tool == "control_exposure_assessment":
                    print("Routing to: exposure_assessment_agent (has pending exposure action)")
                    return "exposure_assessment_agent"
                elif tool == "find_route":
                    print("Routing to: pathfinder_agent (has pending route finding action)")
                    return "pathfinder_agent"
            
            # Default to clarification agent for other pending actions
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
        
        # Also check if message contains file references (for exposure assessment clarifications)
        has_file_reference = any(ext in msg_lower for ext in [".geojson", ".shp", ".kml", ".gpkg", ".json"])
        
        # Check if message is short and might be a location name (for pathfinder)
        # Location responses are typically short (1-5 words) and don't contain action keywords
        is_short_response = len(last_message.split()) <= 5
        has_no_action_keywords = not any(keyword in msg_lower for keyword in ["find", "show", "search", "enable", "disable", "switch", "change"])
        might_be_location = is_short_response and has_no_action_keywords
        
        if msg_lower in potential_responses or has_file_reference or might_be_location:
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
                                suggested_action = prev_content.get('suggested_action', {})
                                tool = suggested_action.get('tool')
                                
                                # Route based on the tool in the pending action
                                if tool == "run_exposure_analysis" or tool == "control_exposure_assessment":
                                    print("✓ Routing to: exposure_assessment_agent (follow-up to exposure clarification)")
                                    state["pending_action"] = prev_content
                                    state["clarification_needed"] = True
                                    return "exposure_assessment_agent"
                                elif tool == "find_route":
                                    print("✓ Routing to: pathfinder_agent (follow-up to pathfinder clarification)")
                                    state["pending_action"] = prev_content
                                    state["clarification_needed"] = True
                                    return "pathfinder_agent"
                                else:
                                    print("✓ Routing to: clarification_agent (follow-up to clarification)")
                                    print(f"Setting pending action: {suggested_action}")
                                    state["pending_action"] = prev_content
                                    state["clarification_needed"] = True
                                    return "clarification_agent"
                        except Exception as e:
                            print(f"Error parsing message content: {e}")
                        break  # Only check the last AI message
        
        # Use simple keyword-based routing (more reliable than LLM for this)
        # Pathfinder Agent - Route planning and navigation (CHECK FIRST - highest priority for routing)
        pathfinder_keywords = ["route", "routes", "directions", "navigate to", "navigation",
                               "find route", "find a route", "show route", "show routes",
                               "get to", "how to get", "way to", "path to", "drive to",
                               "walk to", "cycle to", "bike to", "fastest route", "safest route",
                               "best route", "quickest way", "driving directions", "walking directions",
                               "cycling directions", "pathfinder", "traffic", "avoid traffic",
                               "switch mode", "change mode", "switch to driving", "switch to walking",
                               "switch to cycling", "switch to motorcycle", "switch to car",
                               "switch to bicycle", "switch to pedestrian", "change to driving",
                               "change to walking", "change to cycling", "show driving routes",
                               "show walking routes", "show cycling routes", "show all modes",
                               "sort by fastest", "sort by safest", "sort by best balance",
                               "show fastest", "show safest", "best balance",
                               "sort routes by", "sort the routes by", "sort routes",
                               "fastest route", "safest route", "best balance route",
                               "route optimization", "optimize route", "optimize routes",
                               "perform route optimization", "route planning"]
        
        # Exposure Agent - Exposure assessment
        exposure_keywords = ["exposure", "assessment", "analyze exposure", "run analysis",
                            "exposure analysis", "exposure assessment", "assess exposure",
                            "clear steps", "select hazard", "select element"]
        
        # Hazard Agent - Earthquake, weather, live hazards
        hazard_keywords = ["earthquake", "seismic", "weather", "temperature", "climate",
                          "enable", "disable", "turn on", "turn off", "monitoring",
                          "hazard", "hazards", "live"]
        
        # Map Agent - Location, style, view, time of day, zoom (ACTION keywords only)
        map_action_keywords = ["show", "find", "go to", "navigate", "search", "where is", "locate", 
                              "change to", "switch to", "set to", "use", "apply",
                              "zoom", "zoom in", "zoom out", "fly to", "take me to",
                              "time of day", "lighting", "morning", "daytime", "evening", "nighttime",
                              "i want to see", "want to see", "see", "display", "view"]
        
        # Check for pathfinder keywords FIRST (highest priority for routing)
        if any(keyword in msg_lower for keyword in pathfinder_keywords):
            if not is_question:
                print("Routing to: pathfinder_agent")
                return "pathfinder_agent"
        
        # Check for exposure keywords (second priority for assessment)
        if any(keyword in msg_lower for keyword in exposure_keywords):
            if not is_question:
                print("Routing to: exposure_assessment_agent")
                return "exposure_assessment_agent"
        
        # Check for map ACTION keywords - prioritize over hazard if both present
        # This handles compound requests like "switch to 3D and enable earthquakes"
        has_map_action = any(keyword in msg_lower for keyword in map_action_keywords)
        has_hazard = any(keyword in msg_lower for keyword in hazard_keywords)
        
        # Check for location patterns (place names, addresses, coordinates)
        # Common patterns: "show me [place]", "I want to see [place]", "[place] in the map"
        location_indicators = ["in the map", "on the map", "in map", "on map"]
        has_location_request = any(indicator in msg_lower for indicator in location_indicators)
        
        # If message contains location indicators or map actions, route to map agent
        if (has_map_action or has_location_request) and not is_question:
            # If it's ONLY about hazards (no map actions), route to hazard agent
            # Otherwise, route to map agent (it can handle compound requests)
            if has_hazard and not any(keyword in msg_lower for keyword in ["switch", "change", "set", "show", "go to", "navigate", "zoom", "fly", "style", "view", "mode", "3d", "2d", "time of day", "lighting", "see", "display"]):
                print("Routing to: hazard_agent (hazard-only request)")
                return "hazard_agent"
            else:
                print("Routing to: map_agent (map action detected)")
                return "map_agent"
        
        # Check for hazard keywords (only if no map actions)
        if has_hazard and not is_question:
            print("Routing to: hazard_agent")
            return "hazard_agent"
        
        # Check for specific map style/view/time change requests
        style_keywords = ["satellite", "dark", "light", "outdoors", "navigation", "2d", "3d"]
        time_keywords = ["time of day", "lighting", "morning", "daytime", "evening", "nighttime", "dawn", "dusk", "noon", "midnight", "auto time"]
        change_keywords = ["change", "switch", "set", "use", "apply", "make it"]
        
        has_style = any(keyword in msg_lower for keyword in style_keywords)
        has_time = any(keyword in msg_lower for keyword in time_keywords)
        has_change = any(keyword in msg_lower for keyword in change_keywords)
        
        if (has_style or has_time) and has_change and not is_question:
            print("Routing to: map_agent")
            return "map_agent"
        
        # Default to QA agent for questions
        print("Routing to: qa_agent")
        return "qa_agent"
