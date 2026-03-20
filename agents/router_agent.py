"""Router agent for directing requests to specialized agents"""
import json
import re
from state import AgentState


def _wants_map_history_reset_or_clear(msg_lower: str) -> bool:
    """True for reset/clear/wipe map even with extra words (e.g. 'reset the the map')."""
    if "map" not in msg_lower and "global reset" not in msg_lower:
        return False
    if re.search(
        r"\b(reset|clear|wipe|erase)\b(?:\s+\w+){0,6}\bmap\b|\bmap\b\s+(?:reset|clear)\b",
        msg_lower,
    ):
        return True
    if "global reset" in msg_lower or "restore default map" in msg_lower:
        return True
    return False


def _is_affirmation(msg_lower: str) -> bool:
    """Short replies agreeing to a prior assistant suggestion (e.g. map demo)."""
    t = msg_lower.strip()
    if t in (
        "yes",
        "yeah",
        "yep",
        "sure",
        "ok",
        "okay",
        "please",
        "y",
        "do it",
        "go ahead",
        "alright",
        "sounds good",
    ):
        return True
    if t.startswith(("yes ", "yeah ", "sure ", "ok ", "okay ")):
        return True
    if "yes please" in t:
        return True
    return False


def _prev_qa_offered_map_hazard_demo(prev_content: dict) -> bool:
    """True if a QA-style JSON reply (not clarification card) offered earthquake/weather on the map."""
    if prev_content.get("type") == "clarification":
        return False
    text = prev_content.get("text")
    if not isinstance(text, str) or not text.strip():
        return False
    tl = text.lower()
    # Assistant claimed it will / did enable (common LLM hallucination)
    if "earthquake" in tl and (
        "phivolcs" in tl or "philippine" in tl or "usgs" in tl or "map" in tl or "display" in tl or "enable" in tl
    ):
        return True
    if "weather" in tl and "map" in tl:
        return True
    # Offer phrasing from QA system prompt
    if ("would you like" in tl or "demonstrate" in tl or "show you" in tl or "on the map" in tl) and any(
        k in tl for k in ("earthquake", "phivolcs", "usgs", "seismic", "quake", "weather")
    ):
        return True
    return False


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
            "what is", "what are", "what's", "what does", "what do", "explain", "how does",
            "tell me about",
            "do you know", "are you familiar", "can you tell me", "have you heard",
            "who is", "who are", "why is", "why are", "when is", "when are",
            "describe", "define", "meaning of",
        ]
        
        is_question = any(indicator in msg_lower for indicator in qa_indicators)

        # Heuristic for user-connected spatial layers (used to skip web search)
        _spatial_keywords = [
            "spatial data", "imported layer", "uploaded layer", "geojson layer",
            "connected api", "postgis", "database layer on map", "mcp layer",
            "add from url", "load geojson", "fetch geojson", "geojson url",
            "open import connect", "import connect spatial", "connect spatial data",
            "layers on the map", "layers on my map", "my map layers",
            "list my layers", "what layers are on", "which layers are on",
            "analyze imported", "analyze uploaded", "summarize layer", "summarize layers",
            "visualize my data", "visualize imported", "import / connect",
        ]
        _layer_inventory = any(
            p in msg_lower
            for p in [
                "what layer",
                "which layer",
                "list layer",
                "layers i added",
                "layers i imported",
                "what are my layers",
                "what layers do i have",
            ]
        )
        _sc = state.get("spatial_context") or []
        _spatial_kw = any(k in msg_lower for k in _spatial_keywords)
        _spatial_ctx_match = bool(_sc) and any(
            w in msg_lower
            for w in ["layer", "import", "geojson", "spatial", "feature"]
        )
        wants_spatial_agent = _layer_inventory or (
            (_spatial_kw or _spatial_ctx_match) and not is_question
        )
        
        # Check if web search is enabled - route to web search for Q&A questions
        if state.get("web_search_enabled", False):
            # Check if this is a Q&A type question (not map/hazard control)
            hazard_control_keywords = ["enable", "disable", "turn on", "turn off", "monitoring"]
            map_action_keywords = ["show", "find", "go to", "navigate", "zoom", "fly to", "take me to",
                                  "change to", "switch to", "set to"]
            
            has_hazard_control = any(keyword in msg_lower for keyword in hazard_control_keywords)
            has_map_action = any(keyword in msg_lower for keyword in map_action_keywords)
            
            # If it's a question OR not a control request, use web search
            if (
                is_question or (not has_hazard_control and not has_map_action)
            ) and not wants_spatial_agent:
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
                elif tool in ("open_spatial_data_panel", "add_spatial_layer_from_url"):
                    print("Routing to: spatial_data_agent (has pending spatial data action)")
                    return "spatial_data_agent"
            
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
        # Don't treat clear questions (e.g. "What does X mean?") as pathfinder location replies
        looks_like_question = "?" in last_message or any(
            w in msg_lower for w in ("what ", "who ", "why ", "how ", "when ", "which ")
        )
        might_be_location = (
            is_short_response
            and has_no_action_keywords
            and not looks_like_question
        )
        
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
                                elif tool in ("open_spatial_data_panel", "add_spatial_layer_from_url"):
                                    print("✓ Routing to: spatial_data_agent (follow-up to spatial clarification)")
                                    state["pending_action"] = prev_content
                                    state["clarification_needed"] = True
                                    return "spatial_data_agent"
                                else:
                                    print("✓ Routing to: clarification_agent (follow-up to clarification)")
                                    print(f"Setting pending action: {suggested_action}")
                                    state["pending_action"] = prev_content
                                    state["clarification_needed"] = True
                                    return "clarification_agent"
                            elif _is_affirmation(msg_lower) and _prev_qa_offered_map_hazard_demo(
                                prev_content
                            ):
                                print(
                                    "✓ Routing to: hazard_agent (affirmation after QA offered hazard/map demo)"
                                )
                                return "hazard_agent"
                        except Exception as e:
                            print(f"Error parsing message content: {e}")
                        break  # Only check the last AI message

        # Same as above but when the message wasn't caught by potential_responses / might_be_location
        if _is_affirmation(msg_lower) and len(messages) >= 2:
            for i in range(len(messages) - 2, -1, -1):
                msg = messages[i]
                content = None
                msg_type = None
                if hasattr(msg, "content"):
                    content = msg.content
                    msg_type = getattr(msg, "type", None)
                elif isinstance(msg, dict):
                    content = msg.get("content")
                    msg_type = msg.get("role")
                if msg_type not in ("ai", "assistant") or not content:
                    continue
                try:
                    prev_content = json.loads(content)
                except Exception:
                    continue
                if _prev_qa_offered_map_hazard_demo(prev_content):
                    print(
                        "✓ Routing to: hazard_agent (affirmation after QA offered hazard/map demo, fallback scan)"
                    )
                    return "hazard_agent"
                break
        
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

        # UI Agent - Open panel, dropdown, expand/collapse chat (CHECK BEFORE MAP so "open pathfinder" -> ui_agent)
        ui_agent_keywords = [
            "expand atlas", "atlas full", "full screen", "fullscreen", "maximize chat", "expand chat",
            "minimize chat", "collapse chat", "atlas chat bar", "chat bar to full", "chat to full",
            "map style dropdown", "map style menu", "open map style", "show map style", "style dropdown",
            "time of day dropdown", "lighting dropdown", "open time of day", "show time of day",
            "add boundary panel", "boundary panel", "boundaries panel", "open boundary panel", "show the boundary panel",
            "pathfinder panel", "open pathfinder", "show pathfinder", "route panel", "routing panel", "directions panel",
            "layers panel",
            "open layers",
            "show layers",
            "layer panel",
            "available layers",
            "list layers",
            "list my layers",
            "what layers",
            "which layers",
            "map layers",
            "import panel", "import files", "upload panel", "open import", "show import", "add files panel",
            "live hazard monitor", "hazard monitor", "open hazard monitor", "show hazard monitor",
            "select maps", "open select maps", "show select maps",
            "planning tools", "planning panel", "open planning", "show planning",
            "assessment tools", "assessment panel", "open assessment", "show assessment",
            "tool panel", "tools panel", "open tools", "show tools", "sidebar",
        ]
        if any(keyword in msg_lower for keyword in ui_agent_keywords) and not is_question:
            print("Routing to: ui_agent (panel/UI control)")
            return "ui_agent"

        # Map history: undo / redo / reset — same controls as toolbar (handled by map_agent)
        history_control_keywords = [
            "undo", "redo", "map undo", "map redo", "reset map", "map reset", "global reset",
            "clear the map", "clear map", "wipe map", "erase map", "revert last", "rollback",
            "roll back", "ctrl+z", "ctrl z", "ctrl+y", "ctrl y", "history undo", "history redo",
            "restore default map", "start over on the map", "reset the map", "clear entire map",
        ]
        if (
            any(k in msg_lower for k in history_control_keywords)
            or _wants_map_history_reset_or_clear(msg_lower)
        ) and not is_question:
            print("Routing to: map_agent (map history: undo/redo/reset)")
            return "map_agent"
        
        # Hazard Agent - Earthquake, weather, live hazards
        hazard_keywords = ["earthquake", "seismic", "weather", "temperature", "climate",
                          "enable", "disable", "turn on", "turn off", "monitoring",
                          "hazard", "hazards", "live"]
        
        # Map Agent - Location, style, view, time of day, zoom, boundaries (ACTION keywords only; panel/UI -> ui_agent)
        map_action_keywords = ["show", "find", "go to", "navigate", "search", "where is", "locate",
                              "change to", "switch to", "set to", "use", "apply",
                              "zoom", "zoom in", "zoom out", "fly to", "take me to",
                              "time of day", "lighting", "morning", "daytime", "evening", "nighttime",
                              "i want to see", "want to see", "see", "display", "view",
                              "add boundary", "add boundaries", "add border", "add borders",
                              "show boundary", "show boundaries", "display boundary", "display boundaries",
                              "load boundary", "load boundaries", "put boundary", "put boundaries",
                              "draw boundary", "draw boundaries", "open boundary", "open boundaries",
                              "boundaries for", "borders for", "boundaries of", "borders of",
                              "country boundary", "country boundaries", "national border", "national borders",
                              "administrative boundary", "administrative boundaries",
                              "by province", "by region", "province level", "regional boundary",
                              "add boundaries to the map", "add boundaries to map",
                              "clear boundary", "clear boundaries", "remove boundary", "remove boundaries",
                              "then add", "now add", "next add", "also add", "after that add",
                              "undo", "redo", "reset map", "clear map", "clear the map", "global reset"]
        
        # Check for layer panel commands (critical facility, hazard layers, etc.)
        layer_panel_keywords = ["open", "show", "display"]
        has_layer_panel_request = (
            any(keyword in msg_lower for keyword in layer_panel_keywords) and
            "panel" in msg_lower and
            ("layer" in msg_lower or "layers" in msg_lower)
        )
        
        # Check for pathfinder keywords FIRST (highest priority for routing)
        if any(keyword in msg_lower for keyword in pathfinder_keywords):
            if not is_question:
                print("Routing to: pathfinder_agent")
                return "pathfinder_agent"

        if wants_spatial_agent:
            print("Routing to: spatial_data_agent")
            return "spatial_data_agent"
        
        # Check for exposure keywords (second priority for assessment)
        if any(keyword in msg_lower for keyword in exposure_keywords):
            if not is_question:
                print("Routing to: exposure_assessment_agent")
                return "exposure_assessment_agent"
        
        # Check for layer panel requests - route based on layer type
        if has_layer_panel_request and not is_question:
            # Check if it's hazard layers or critical facility layers
            if "hazard" in msg_lower:
                print("Routing to: hazard_agent (hazard layers panel request)")
                return "hazard_agent"
            elif "critical facility" in msg_lower or "critical facilities" in msg_lower:
                print("Routing to: map_agent (critical facility layers panel request)")
                return "map_agent"
            else:
                # Generic layers panel request - route to map agent
                print("Routing to: map_agent (generic layers panel request)")
                return "map_agent"

        # Live earthquake / seismic feeds: must beat map_agent. Users say "show on the map" which
        # matches map keywords; map_agent's LLM then does search_location("earthquake") by mistake.
        if not is_question and any(
            w in msg_lower
            for w in (
                "earthquake",
                "earthquakes",
                "seismic",
                "quake",
                "phivolcs",
                "usgs",
            )
        ):
            print("Routing to: hazard_agent (earthquake / seismic monitoring)")
            return "hazard_agent"
        
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
