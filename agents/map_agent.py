"""Map agent for handling location, style, and view mode"""
import json
from langchain_core.messages import AIMessage
from state import AgentState
from tools import search_location, change_map_style, switch_view_mode


class MapAgent:
    """Handles map-related operations: location search, style, view mode"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [search_location, change_map_style, switch_view_mode]
        # Map agent uses rule-based intent parsing, not LLM prompts
    
    def process(self, state: AgentState) -> AgentState:
        """Process map-related requests"""
        messages = state["messages"]
        last_message = messages[-1].content
        map_state = state.get("map_state", {})
        current_style = map_state.get("currentMapStyle", "")
        
        print(f"Map Agent processing: {last_message}")
        print(f"Current map style: {current_style}")
        
        # Check if there are hazard requests that need clarification
        # If so, route to HazardAgent instead
        msg_lower = last_message.lower()
        has_earthquake = any(word in msg_lower for word in ["earthquake", "seismic", "quake"])
        has_earthquake_source = any(word in msg_lower for word in ["global", "usgs", "philippine", "philippines", "phivolcs"])
        
        if has_earthquake and not has_earthquake_source:
            # Earthquake mentioned without source - let HazardAgent handle clarification
            print("Earthquake without source detected - routing to HazardAgent")
            # Don't process here, return state unchanged to let router send to HazardAgent
            # But we need to signal this somehow... Actually, the router should handle this
            # For now, just don't handle it in MapAgent
            pass
        
        # Detect ambiguous patterns
        ambiguous = self._detect_ambiguous(last_message)
        if ambiguous:
            print(f"Detected ambiguous pattern: {ambiguous}")
            state["clarification_needed"] = True
            state["pending_action"] = ambiguous
            # Include suggested_action in the message so it can be retrieved later
            state["messages"].append(AIMessage(content=json.dumps({
                "type": "clarification",
                "question": ambiguous["question"],
                "options": ambiguous["options"],
                "suggested_action": ambiguous.get("suggested_action")  # Include this!
            })))
            return state
        
        # Process clear intent
        action = self._parse_intent(last_message, state.get("map_state", {}))
        print(f"Parsed action: {action}")
        
        # Check if there are also hazard-related requests in the same message
        hazard_actions = self._parse_hazard_intent(last_message)
        
        if action and hazard_actions:
            # Compound request with hazards
            print(f"Detected compound request with hazard actions: {hazard_actions}")
            if action.get("multiple_actions"):
                # Already has multiple map actions, add hazard actions
                all_actions = hazard_actions + action["multiple_actions"]
            else:
                # Single map action + hazard actions
                all_actions = hazard_actions + [action]
            
            state["messages"].append(AIMessage(content=json.dumps({
                "multiple_actions": all_actions,
                "requires_frontend": True
            })))
        elif action:
            state["messages"].append(AIMessage(content=json.dumps(action)))
        else:
            state["messages"].append(AIMessage(
                content=json.dumps({
                    "text": "I can help you with map navigation, style changes, or view modes. What would you like to do?"
                })
            ))
        
        return state
    
    def _parse_hazard_intent(self, message: str) -> list:
        """Parse hazard-related intents in map requests - returns list of actions"""
        msg_lower = message.lower()
        actions = []
        
        # Determine action
        enable_keywords = ["enable", "show", "see", "view", "display", "turn on", "activate", "want to see", "also"]
        disable_keywords = ["disable", "hide", "turn off", "deactivate", "remove"]
        
        if any(word in msg_lower for word in disable_keywords):
            action = "disable"
        elif any(word in msg_lower for word in enable_keywords):
            action = "enable"
        else:
            return actions  # No action detected
        
        # Check for earthquake requests
        if any(word in msg_lower for word in ["earthquake", "seismic", "quake"]):
            has_global = "global" in msg_lower or "usgs" in msg_lower
            has_philippine = "philippine" in msg_lower or "philippines" in msg_lower or "phivolcs" in msg_lower
            
            if has_global and has_philippine:
                # Both sources
                actions.append({"tool": "control_earthquake_data", "action": action, "source": "philippine", "requires_frontend": True})
                actions.append({"tool": "control_earthquake_data", "action": action, "source": "global", "requires_frontend": True})
            elif has_global:
                actions.append({"tool": "control_earthquake_data", "action": action, "source": "global", "requires_frontend": True})
            elif has_philippine:
                actions.append({"tool": "control_earthquake_data", "action": action, "source": "philippine", "requires_frontend": True})
            # else: Don't add anything - let HazardAgent handle the clarification
        
        # Check for weather requests
        if any(word in msg_lower for word in ["weather", "temperature", "climate"]):
            scope = "city" if "city" in msg_lower or "municipality" in msg_lower else "province"
            actions.append({"tool": "control_weather_data", "action": action, "scope": scope, "requires_frontend": True})
        
        return actions
    
    def _extract_time_preset(self, msg_lower: str) -> str:
        """Extract time of day preset from message"""
        # Check in order of specificity (most specific first to avoid false matches)
        # Check "nighttime" before "daytime" since "nighttime" contains "time"
        if "nighttime" in msg_lower or "night time" in msg_lower or "midnight" in msg_lower:
            return "nighttime"
        elif "night" in msg_lower:
            return "nighttime"
        elif "morning" in msg_lower or "dawn" in msg_lower or "sunrise" in msg_lower:
            return "morning"
        elif "evening" in msg_lower or "dusk" in msg_lower or "sunset" in msg_lower:
            return "evening"
        elif "daytime" in msg_lower or "day time" in msg_lower or "noon" in msg_lower or "afternoon" in msg_lower:
            return "daytime"
        elif "day" in msg_lower and "time of day" not in msg_lower:
            return "daytime"
        else:
            return "auto"
    
    def _detect_ambiguous(self, message: str) -> dict | None:
        """Detect ambiguous map requests"""
        msg_lower = message.lower()
        
        # Darker/lighter patterns
        if any(word in msg_lower for word in ["darker", "darken"]):
            return {
                "question": "I found a map style for a darker appearance. Would you like to switch to **Dark (Mapbox)**?",
                "options": ["yes", "no", "show all styles"],
                "suggested_action": {"tool": "change_map_style", "style": "dark"}
            }
        
        if any(word in msg_lower for word in ["lighter", "lighten", "brighter"]):
            return {
                "question": "I found a map style for a lighter appearance. Would you like to switch to **Light (Mapbox)**?",
                "options": ["yes", "no", "show all styles"],
                "suggested_action": {"tool": "change_map_style", "style": "light"}
            }
        
        return None
    
    def _parse_intent(self, message: str, map_state: dict) -> dict | None:
        """Parse map intents using LLM for natural language understanding"""
        
        msg_lower = message.lower()
        
        # Check if this is a time of day request - needs special handling for 2D/incompatible styles
        # BUT: Skip validation if user explicitly requests style/view changes in the same message
        time_keywords = ["time of day", "lighting", "morning", "daytime", "evening", "nighttime", "dawn", "dusk", "noon", "midnight", "auto time"]
        is_time_request = any(keyword in msg_lower for keyword in time_keywords)
        
        # Check if user is explicitly requesting style or view changes
        explicit_style_change = any(word in msg_lower for word in ["switch to", "change to", "set to", "use"]) and any(word in msg_lower for word in ["default", "satellite", "style"])
        explicit_view_change = any(word in msg_lower for word in ["switch to", "change to", "enable", "turn on"]) and "3d" in msg_lower
        
        # If user explicitly requests the needed changes, skip validation and let LLM parse everything
        if is_time_request and not (explicit_style_change or explicit_view_change):
            current_view = map_state.get("viewMode", "2d")
            current_style = map_state.get("currentMapStyle", "")
            
            # Only Default and Satellite support 3D
            supports_3d = "Default" in current_style or "Satellite" in current_style
            
            time_preset = self._extract_time_preset(msg_lower)
            
            # If in 2D mode, ask to switch to 3D
            if current_view == "2d":
                if supports_3d:
                    return {
                        "type": "clarification",
                        "question": f"⚠️ **Time of day control only works in 3D mode.**\n\nWould you like me to switch to 3D view and set the time to **{time_preset.title()}**?",
                        "options": ["Yes, switch to 3D and change time", "No, cancel"],
                        "suggested_action": {
                            "multiple_actions": [
                                {"tool": "switch_view_mode", "mode": "3d", "requires_frontend": True},
                                {"tool": "control_time_of_day", "preset": time_preset, "requires_frontend": True}
                            ]
                        }
                    }
                else:
                    # Current style doesn't support 3D
                    return {
                        "type": "clarification",
                        "question": f"⚠️ **Time of day control requires 3D mode, but {current_style} doesn't support 3D.**\n\nWould you like me to switch to Default style (supports 3D) or Satellite style (supports 3D)?",
                        "options": ["Default style + 3D + time change", "Satellite style + 3D + time change", "Cancel"],
                        "suggested_action": {
                            "multiple_actions": [
                                {"tool": "change_map_style", "style": "default", "requires_frontend": True},
                                {"tool": "switch_view_mode", "mode": "3d", "requires_frontend": True},
                                {"tool": "control_time_of_day", "preset": time_preset, "requires_frontend": True}
                            ],
                            "alternative_style": "satellite"  # For option 2
                        }
                    }
            
            # If in 3D but incompatible style
            if current_view == "3d" and not supports_3d:
                return {
                    "type": "clarification",
                    "question": f"⚠️ **{current_style} doesn't support time of day control.**\n\nOnly Default and Satellite styles support this feature. Which would you like?",
                    "options": ["Switch to Default style", "Switch to Satellite style", "Cancel"],
                    "suggested_action": {
                        "multiple_actions": [
                            {"tool": "change_map_style", "style": "default", "requires_frontend": True},
                            {"tool": "control_time_of_day", "preset": time_preset, "requires_frontend": True}
                        ],
                        "alternative_style": "satellite"  # For option 2
                    }
                }
        
        system_prompt = """You are a map control intent parser. Extract the user's intent from their message.

Available map actions:
1. **search_location** - Navigate to a location
   - Extract: location name/query
   - Examples: "show me Paris", "zoom to Tokyo", "fly to Eiffel Tower"

2. **change_map_style** - Change visual style
   - Options: satellite, outdoors, light, dark, default, navigation_day, navigation_night
   - Examples: "change to satellite", "make it darker" → dark

3. **switch_view_mode** - Change 2D/3D view
   - Options: 2d, 3d
   - Examples: "switch to 3D", "change orientation to 3D"

4. **control_time_of_day** - Control lighting in 3D view
   - Options: auto, morning, daytime, evening, nighttime
   - Examples: "set time to morning", "change lighting to night", "auto time of day"

5. **control_zoom** - Zoom in or out on the map
   - direction: "in" or "out"
   - amount: zoom level change (0.1 = small, 0.5 = medium, 1.0 = large)
   - is_max: true if user wants maximum zoom (100% or "max")
   - For percentages under 100%: convert to decimal (10% = 0.1, 50% = 0.5)
   - For 100% or "max": set is_max to true
   - Examples: "zoom in by 10%" → {"tool": "control_zoom", "direction": "in", "amount": 0.1, "is_max": false}
   - Examples: "zoom out by 50%" → {"tool": "control_zoom", "direction": "out", "amount": 0.5, "is_max": false}
   - Examples: "zoom in by 100%" → {"tool": "control_zoom", "direction": "in", "is_max": true}
   - Examples: "zoom to max" → {"tool": "control_zoom", "direction": "in", "is_max": true}
   - Examples: "zoom out to max" → {"tool": "control_zoom", "direction": "out", "is_max": true}
   - Examples: "zoom in a bit" → {"tool": "control_zoom", "direction": "in", "amount": 0.5, "is_max": false}

Respond with JSON array of actions. Each action has: {"tool": "...", "param": "value"}

Examples:
- "zoom to Paris" → [{"tool": "search_location", "query": "Paris"}]
- "show Tokyo in satellite view" → [{"tool": "search_location", "query": "Tokyo"}, {"tool": "change_map_style", "style": "satellite"}]
- "switch to 3D" → [{"tool": "switch_view_mode", "mode": "3d"}]
- "make it darker" → [{"tool": "change_map_style", "style": "dark"}]
- "set time to morning" → [{"tool": "control_time_of_day", "preset": "morning"}]
- "change lighting to nighttime" → [{"tool": "control_time_of_day", "preset": "nighttime"}]
- "zoom in by 10%" → [{"tool": "control_zoom", "direction": "in", "amount": 0.1, "is_max": false}]
- "zoom out by 50%" → [{"tool": "control_zoom", "direction": "out", "amount": 0.5, "is_max": false}]
- "zoom in by 100%" → [{"tool": "control_zoom", "direction": "in", "is_max": true}]
- "zoom to max" → [{"tool": "control_zoom", "direction": "in", "is_max": true}]
- "zoom in a bit" → [{"tool": "control_zoom", "direction": "in", "amount": 0.5, "is_max": false}]

Return ONLY the JSON array, no explanation."""

        try:
            # Use LLM to parse intent
            response = self.llm.invoke(message, system_prompt=system_prompt)
            print(f"LLM parsed intent: {response[:200]}")
            
            # Clean response - extract JSON if wrapped in markdown
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            parsed_actions = json.loads(response)
            
            if not isinstance(parsed_actions, list):
                parsed_actions = [parsed_actions]
            
            # Convert to internal format and add requires_frontend
            actions = []
            for action in parsed_actions:
                tool = action.get("tool")
                if tool == "search_location":
                    actions.append({"tool": "search_location", "query": action.get("query", ""), "requires_frontend": True})
                elif tool == "change_map_style":
                    actions.append({"tool": "change_map_style", "style": action.get("style", ""), "requires_frontend": True})
                elif tool == "switch_view_mode":
                    mode = action.get("mode", "")
                    # Check 3D compatibility
                    if mode == "3d":
                        current_style = map_state.get("currentMapStyle", "")
                        non_3d_styles = ["Dark", "Light", "Outdoors", "Navigation"]
                        
                        # Check if user already requested a style change in this message
                        has_style_change = any(a.get("tool") == "change_map_style" for a in actions)
                        
                        if any(style in current_style for style in non_3d_styles) and not has_style_change:
                            # Only add style change if user didn't already request one
                            info_msg = f"Note: {current_style} doesn't support 3D terrain. Switching to Default style."
                            actions.append({"tool": "change_map_style", "style": "default", "requires_frontend": True})
                            actions.append({"tool": "switch_view_mode", "mode": "3d", "requires_frontend": True, "info_message": info_msg})
                        else:
                            actions.append({"tool": "switch_view_mode", "mode": mode, "requires_frontend": True})
                    else:
                        actions.append({"tool": "switch_view_mode", "mode": mode, "requires_frontend": True})
                elif tool == "control_time_of_day":
                    preset = action.get("preset", "auto")
                    actions.append({"tool": "control_time_of_day", "preset": preset, "requires_frontend": True})
                elif tool == "control_zoom":
                    direction = action.get("direction", "in")
                    amount = action.get("amount", 1.0)
                    is_max = action.get("is_max", False)
                    actions.append({"tool": "control_zoom", "direction": direction, "amount": amount, "is_max": is_max, "requires_frontend": True})
            
            # Prioritize actions for best UX: style → view → time/other → location
            if len(actions) > 1:
                style_actions = [a for a in actions if a.get("tool") == "change_map_style"]
                view_actions = [a for a in actions if a.get("tool") == "switch_view_mode"]
                time_actions = [a for a in actions if a.get("tool") == "control_time_of_day"]
                location_actions = [a for a in actions if a.get("tool") == "search_location"]
                other_actions = [a for a in actions if a.get("tool") not in ["switch_view_mode", "change_map_style", "search_location", "control_time_of_day"]]
                
                # Order: style first (needed for 3D), then view mode, then time of day, then other, location last
                sorted_actions = style_actions + view_actions + time_actions + other_actions + location_actions
                return {"multiple_actions": sorted_actions, "requires_frontend": True}
            elif len(actions) == 1:
                return actions[0]
            
            return None
            
        except Exception as e:
            print(f"Error parsing intent with LLM: {e}")
            # Fallback to simple keyword matching for critical failures
            msg_lower = message.lower()
            if any(word in msg_lower for word in ["show", "zoom", "fly", "navigate", "find"]):
                # Try to extract location after common keywords
                for keyword in ["show", "zoom to", "fly to", "navigate to"]:
                    if keyword in msg_lower:
                        location = msg_lower.split(keyword, 1)[1].strip().split()[0:3]
                        return {"tool": "search_location", "query": " ".join(location), "requires_frontend": True}
            return None


