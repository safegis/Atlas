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

Respond with JSON array of actions. Each action has: {"tool": "...", "param": "value"}

Examples:
- "zoom to Paris" → [{"tool": "search_location", "query": "Paris"}]
- "show Tokyo in satellite view" → [{"tool": "search_location", "query": "Tokyo"}, {"tool": "change_map_style", "style": "satellite"}]
- "switch to 3D" → [{"tool": "switch_view_mode", "mode": "3d"}]
- "make it darker" → [{"tool": "change_map_style", "style": "dark"}]

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
                        
                        if any(style in current_style for style in non_3d_styles):
                            info_msg = f"Note: {current_style} doesn't support 3D terrain. Switching to Default style."
                            actions.append({"tool": "change_map_style", "style": "default", "requires_frontend": True})
                            actions.append({"tool": "switch_view_mode", "mode": "3d", "requires_frontend": True, "info_message": info_msg})
                        else:
                            actions.append({"tool": "switch_view_mode", "mode": mode, "requires_frontend": True})
                    else:
                        actions.append({"tool": "switch_view_mode", "mode": mode, "requires_frontend": True})
            
            # Prioritize actions for best UX
            if len(actions) > 1:
                view_actions = [a for a in actions if a.get("tool") == "switch_view_mode"]
                style_actions = [a for a in actions if a.get("tool") == "change_map_style"]
                location_actions = [a for a in actions if a.get("tool") == "search_location"]
                other_actions = [a for a in actions if a.get("tool") not in ["switch_view_mode", "change_map_style", "search_location"]]
                
                sorted_actions = view_actions + style_actions + other_actions + location_actions
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


