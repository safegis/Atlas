"""Hazard agent for handling earthquake and weather data"""
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from state import AgentState
from tools import control_earthquake_data, control_weather_data


class HazardAgent:
    """Handles hazard monitoring: earthquake and weather data"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [control_earthquake_data, control_weather_data]
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Hazard Monitoring Agent for SafeGIS. You handle:
- Earthquake data (Philippine PHIVOLCS or Global USGS)
- Weather data (province or city level)

Available tools: control_earthquake_data, control_weather_data, ask_clarification

Respond with tool calls in JSON format."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def process(self, state: AgentState) -> AgentState:
        """Process hazard monitoring requests"""
        messages = state["messages"]
        last_message = messages[-1].content
        map_state = state.get("map_state", {})
        
        print(f"Hazard Agent processing: {last_message}")
        
        # Check if there are map-related requests that might need clarification
        map_action = self._parse_map_intent(last_message, map_state)
        
        # If map action needs clarification, handle that first
        if map_action and map_action.get("type") == "clarification":
            print(f"Map action needs clarification, will handle hazard after")
            
            # Check if hazard also needs clarification
            hazard_ambiguous = self._detect_ambiguous(last_message)
            
            # Store the hazard intent to execute after map clarification
            # Only parse if not ambiguous, otherwise store the clarification
            if hazard_ambiguous:
                # Hazard also needs clarification - store it to ask after map clarification
                hazard_action = hazard_ambiguous
            else:
                hazard_action = self._parse_intent(last_message)
            
            # Also check if there are other map actions (like 3D) that should be preserved
            # Parse view mode separately since it wasn't ambiguous
            msg_lower = last_message.lower()
            additional_map_actions = []
            
            # Check for 3D mode request
            has_3d_request = ("3d" in msg_lower or "three dimensional" in msg_lower) and ("view" in msg_lower or "mode" in msg_lower or "orientation" in msg_lower or "switch" in msg_lower)
            has_2d_request = ("2d" in msg_lower or "two dimensional" in msg_lower) and ("view" in msg_lower or "mode" in msg_lower or "orientation" in msg_lower or "switch" in msg_lower)
            
            # Check if the requested style conflicts with 3D mode
            # Dark, Light, Outdoors, Navigation styles don't support 3D terrain
            non_3d_styles = ["dark", "light", "outdoors", "navigation"]
            requested_style = map_action.get("suggested_action", {}).get("style", "")
            style_conflicts_with_3d = any(style in requested_style for style in non_3d_styles)
            
            if has_3d_request and style_conflicts_with_3d:
                # Conflict detected - dark/light/outdoors/navigation styles don't support 3D
                # Inform user and skip 3D mode
                map_action["conflict_note"] = f"Note: {requested_style.title()} style doesn't support 3D terrain. Keeping in 2D mode."
            elif has_3d_request:
                additional_map_actions.append({"tool": "switch_view_mode", "mode": "3d", "requires_frontend": True})
            elif has_2d_request:
                additional_map_actions.append({"tool": "switch_view_mode", "mode": "2d", "requires_frontend": True})
            
            map_action["pending_hazard_action"] = hazard_action
            if additional_map_actions:
                map_action["additional_map_actions"] = additional_map_actions
            
            state["clarification_needed"] = True
            state["pending_action"] = map_action
            state["messages"].append(AIMessage(content=json.dumps(map_action)))
            return state
        
        # Check for ambiguous hazard requests
        ambiguous = self._detect_ambiguous(last_message)
        if ambiguous:
            print(f"Detected ambiguous hazard request: {ambiguous}")
            
            state["clarification_needed"] = True
            state["pending_action"] = ambiguous
            
            # Store map actions to execute after clarification
            clarification_response = {
                "type": "clarification",
                "question": ambiguous["question"],
                "options": ambiguous["options"],
                "suggested_action": ambiguous.get("suggested_action")
            }
            
            # If there are map actions (not clarifications), store them as pending
            if map_action and map_action.get("type") != "clarification":
                clarification_response["pending_map_action"] = map_action
            
            state["messages"].append(AIMessage(content=json.dumps(clarification_response)))
            return state
        
        # Parse hazard intent
        action = self._parse_intent(last_message)
        print(f"Hazard action: {action}")
        
        # Check if there are also map-related requests in the same message
        map_action = self._parse_map_intent(last_message, map_state)
        
        # Check if map action needs clarification
        if map_action and map_action.get("type") == "clarification":
            print(f"Map action needs clarification in compound request")
            # Store the hazard action as pending, ask for map clarification
            state["clarification_needed"] = True
            state["pending_action"] = {
                **map_action,
                "pending_hazard_action": action  # Store the hazard action to execute after clarification
            }
            state["messages"].append(AIMessage(content=json.dumps({
                "type": "clarification",
                "question": map_action["question"],
                "options": map_action["options"],
                "suggested_action": map_action.get("suggested_action"),
                "pending_hazard_action": action
            })))
            return state
        
        if action and map_action:
            # Compound request - return both actions
            print(f"Detected compound request with map action: {map_action}")
            state["messages"].append(AIMessage(content=json.dumps({
                "multiple_actions": [action, map_action],
                "requires_frontend": True
            })))
        elif action:
            state["messages"].append(AIMessage(content=json.dumps(action)))
        else:
            state["messages"].append(AIMessage(
                content=json.dumps({
                    "text": "I can help you enable earthquake or weather monitoring. What would you like to see?"
                })
            ))
        
        return state
    
    def _parse_map_intent(self, message: str, map_state: dict = None) -> dict | None:
        """Parse map-related intents in hazard requests - returns action(s) or clarification"""
        msg_lower = message.lower()
        actions = []
        if map_state is None:
            map_state = {}
        
        # Check for ambiguous style requests first
        if ("style" in msg_lower or "map" in msg_lower or "change" in msg_lower):
            # Check for ambiguous "dark" or "night" request - could be Dark or Navigation Night
            if ("dark" in msg_lower or "night" in msg_lower) and "navigation night" not in msg_lower and "navigation day" not in msg_lower:
                # Ambiguous - could be Dark or Navigation Night
                return {
                    "type": "clarification",
                    "question": "I found 2 map styles with dark appearance. Which one would you like?",
                    "options": [
                        "Dark (Mapbox) - Dark color scheme for general use",
                        "Navigation Night (Mapbox) - Navigation optimized for nighttime driving"
                    ],
                    "suggested_action": {"tool": "change_map_style", "style": "dark"},
                    "requires_clarification": True
                }
            
            # Check for ambiguous "darker" request
            if ("darker" in msg_lower or "darken" in msg_lower):
                return {
                    "type": "clarification",
                    "question": "I found a map style for a darker appearance. Would you like to switch to **Dark (Mapbox)**?",
                    "options": ["yes", "no", "show all styles"],
                    "suggested_action": {"tool": "change_map_style", "style": "dark"},
                    "requires_clarification": True
                }
        
        # Map style detection - specific matches only
        style_map = {
            "satellite": "satellite",
            "outdoors": "outdoors",
            "light": "light",
            "dark": "dark",
            "default": "default",
            "navigation day": "navigation_day",
            "navigation night": "navigation_night"
        }
        
        for style_name, style_value in style_map.items():
            if style_name in msg_lower and ("style" in msg_lower or "map" in msg_lower or "change" in msg_lower):
                actions.append({"tool": "change_map_style", "style": style_value, "requires_frontend": True})
                break  # Only one style at a time
        
        # View mode detection
        if ("3d" in msg_lower or "three dimensional" in msg_lower) and ("view" in msg_lower or "mode" in msg_lower or "orientation" in msg_lower or "switch" in msg_lower):
            # Check if current style doesn't support 3D
            current_style = map_state.get("currentMapStyle", "")
            print(f"HazardAgent: Checking 3D compatibility - Current style: '{current_style}'")
            non_3d_styles = ["Dark", "Light", "Outdoors", "Navigation"]
            
            if any(style in current_style for style in non_3d_styles):
                # Need to change style to default first, then switch to 3D
                info_msg = f"Note: {current_style} doesn't support 3D terrain. Switching to Default style."
                print(f"HazardAgent: Style conflict detected, adding style change action")
                
                # Add style change action first
                actions.append({"tool": "change_map_style", "style": "default", "requires_frontend": True})
                # Then add view mode change with info message
                actions.append({"tool": "switch_view_mode", "mode": "3d", "requires_frontend": True, "info_message": info_msg})
            else:
                print(f"HazardAgent: Style '{current_style}' supports 3D, no style change needed")
                actions.append({"tool": "switch_view_mode", "mode": "3d", "requires_frontend": True})
        elif ("2d" in msg_lower or "two dimensional" in msg_lower or "flat" in msg_lower) and ("view" in msg_lower or "mode" in msg_lower or "orientation" in msg_lower or "switch" in msg_lower):
            actions.append({"tool": "switch_view_mode", "mode": "2d", "requires_frontend": True})
        
        # Return multiple actions if found
        if len(actions) > 1:
            return {"multiple_actions": actions, "requires_frontend": True}
        elif len(actions) == 1:
            return actions[0]
        
        return None
    
    def _detect_ambiguous(self, message: str) -> dict | None:
        """Detect ambiguous hazard requests that need clarification"""
        msg_lower = message.lower()
        
        # Determine if this is an enable or disable request
        disable_keywords = ["disable", "hide", "turn off", "deactivate", "remove"]
        is_disable = any(word in msg_lower for word in disable_keywords)
        
        # Check if requesting earthquake data without specifying source
        if any(word in msg_lower for word in ["earthquake", "seismic", "quake"]):
            # If they explicitly mention global/usgs or philippine/phivolcs, it's not ambiguous
            has_global = "global" in msg_lower or "usgs" in msg_lower or "worldwide" in msg_lower or "world" in msg_lower
            has_philippine = "philippine" in msg_lower or "philippines" in msg_lower or "phivolcs" in msg_lower or "local" in msg_lower
            
            # If neither mentioned, ask for clarification (if both mentioned, it's clear - enable/disable both)
            # Only ask for clarification on enable requests - for disable, we can disable all
            if not has_global and not has_philippine and not is_disable:
                return {
                    "type": "clarification",
                    "question": "I found 2 earthquake data sources. Which one would you like to see?",
                    "options": [
                        "Philippines - Latest earthquake data from PHIVOLCS (Philippine Institute of Volcanology and Seismology)",
                        "Global - Worldwide earthquake data from USGS (U.S. Geological Survey)",
                        "Both - Show both Philippine and Global earthquake data"
                    ],
                    "suggested_action": {"tool": "control_earthquake_data", "action": "enable", "source": "philippine"}
                }
        
        # Check if requesting weather data - always list all available sources
        if any(word in msg_lower for word in ["weather", "temperature", "climate"]) and not is_disable:
            return {
                "type": "clarification",
                "question": "I found weather data at different levels. Which scope would you like?",
                "options": [
                    "Province - Weather by province (Philippines)",
                    "City - Weather by city/municipality (Abra)",
                    "City - Weather by city/municipality (Agusan del Norte)",
                    "City - Weather by city/municipality (Agusan del Sur)",
                    "City - Weather by city/municipality (Aklan)",
                    "All - Show all available weather data"
                ],
                "suggested_action": {"tool": "control_weather_data", "action": "enable", "scope": "all"}
            }
        
        return None
    
    def _parse_intent(self, message: str) -> dict | None:
        """Parse hazard monitoring intents"""
        msg_lower = message.lower()
        
        # Determine action - check for enable keywords (including "see", "view", "display")
        enable_keywords = ["enable", "show", "see", "view", "display", "turn on", "activate", "want to see", "also"]
        disable_keywords = ["disable", "hide", "turn off", "deactivate", "remove"]
        
        # Check disable first (more specific), then enable, default to enable for viewing requests
        if any(word in msg_lower for word in disable_keywords):
            action = "disable"
        elif any(word in msg_lower for word in enable_keywords):
            action = "enable"
        else:
            action = "enable"  # Default to enable for hazard viewing
        
        # Check for location-specific requests (Philippines/Global) even without "earthquake" keyword
        # This handles follow-up requests like "also enable the Philippines"
        has_global = "global" in msg_lower or "usgs" in msg_lower
        has_philippine = "philippine" in msg_lower or "philippines" in msg_lower or "phivolcs" in msg_lower
        
        # If location is mentioned with an action word, assume earthquake context
        if (has_global or has_philippine) and any(word in msg_lower for word in enable_keywords + disable_keywords):
            # If both mentioned, enable both
            if has_global and has_philippine:
                return {
                    "tool": "control_earthquake_data",
                    "action": action,
                    "source": "philippine",
                    "requires_frontend": True,
                    "also_enable": {"tool": "control_earthquake_data", "action": action, "source": "global"}
                }
            
            # Otherwise, determine which one
            source = "global" if has_global else "philippine"
            return {"tool": "control_earthquake_data", "action": action, "source": source, "requires_frontend": True}
        
        # Earthquake - explicit mention
        if any(word in msg_lower for word in ["earthquake", "seismic", "quake"]):
            # If both mentioned, enable/disable both
            if has_global and has_philippine:
                return {
                    "tool": "control_earthquake_data",
                    "action": action,
                    "source": "philippine",
                    "requires_frontend": True,
                    "also_enable": {"tool": "control_earthquake_data", "action": action, "source": "global"}
                }
            
            # If disabling without specifying source, disable both
            if action == "disable" and not has_global and not has_philippine:
                return {
                    "tool": "control_earthquake_data",
                    "action": "disable",
                    "source": "philippine",
                    "requires_frontend": True,
                    "also_enable": {"tool": "control_earthquake_data", "action": "disable", "source": "global"}
                }
            
            # Otherwise, determine which one
            source = "global" if has_global else "philippine"
            return {"tool": "control_earthquake_data", "action": action, "source": source, "requires_frontend": True}
        
        # Weather
        if any(word in msg_lower for word in ["weather", "temperature", "climate"]):
            # Check for specific province names (check these first, even without "city" keyword)
            if "abra" in msg_lower:
                return {"tool": "control_weather_data", "action": action, "scope": "city", "province": "Abra", "requires_frontend": True}
            elif "agusan del norte" in msg_lower:
                return {"tool": "control_weather_data", "action": action, "scope": "city", "province": "Agusan del Norte", "requires_frontend": True}
            elif "agusan del sur" in msg_lower:
                return {"tool": "control_weather_data", "action": action, "scope": "city", "province": "Agusan del Sur", "requires_frontend": True}
            elif "aklan" in msg_lower:
                return {"tool": "control_weather_data", "action": action, "scope": "city", "province": "Aklan", "requires_frontend": True}
            elif "all" in msg_lower:
                return {"tool": "control_weather_data", "action": action, "scope": "all", "requires_frontend": True}
            elif "city" in msg_lower or "municipality" in msg_lower:
                # Generic city request - default to all cities
                return {"tool": "control_weather_data", "action": action, "scope": "all_cities", "requires_frontend": True}
            else:
                # Province level (Philippines-wide)
                return {"tool": "control_weather_data", "action": action, "scope": "province", "requires_frontend": True}
        
        return None


