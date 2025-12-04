"""Map agent for handling location, style, and view mode"""
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from state import AgentState
from tools import search_location, change_map_style, switch_view_mode


class MapAgent:
    """Handles map-related operations: location search, style, view mode"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [search_location, change_map_style, switch_view_mode]
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Map Agent for SafeGIS. You handle:
- Location searches and navigation
- Map style changes (default, satellite, outdoors, light, dark, navigation_day, navigation_night)
- View mode switching (2D/3D)

When the user's request is ambiguous (e.g., "make it darker"), use the ask_clarification tool.

Available tools: search_location, change_map_style, switch_view_mode, ask_clarification

Respond with tool calls in JSON format or a helpful message."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
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
        """Parse clear map intents - can return multiple actions"""
        msg_lower = message.lower()
        actions = []
        
        # Location search - extract location name
        location_keywords = ["show me", "show", "find", "go to", "navigate to", "search for", "where is", "locate", "take me to", "fly to", "fly the map to", "fly"]
        
        for keyword in location_keywords:
            if keyword in msg_lower:
                # Extract location by finding text after the keyword and before style/view keywords
                location_part = msg_lower.split(keyword, 1)[1] if keyword in msg_lower else ""
                
                # Stop at common separators (commas, "and", "then", style/view keywords)
                separators = [",", " and ", " then ", " change ", " switch ", " satellite", " outdoors", " light", " dark", 
                             " 2d", " 3d", " style", " mode", " view", " orientation"]
                
                for sep in separators:
                    if sep in location_part:
                        location_part = location_part.split(sep)[0]
                        break
                
                location = location_part.strip()
                
                # Clean up common suffixes and prefixes
                location = location.replace("on the map", "").replace("in the map", "").strip()
                
                # Remove leading "the" only if it's at the start
                if location.startswith("the "):
                    location = location[4:]
                
                # Keep location descriptors like "in paris" - they help geocoding accuracy
                # Just clean up the query
                location = location.strip()
                
                # Only add if we have a meaningful location (not empty and not a style/view keyword)
                if location and len(location) > 2 and location not in ["map", "style", "view", "mode"]:
                    actions.append({"tool": "search_location", "query": location, "requires_frontend": True})
                    break  # Only extract one location
        
        # Map style
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
            if style_name in msg_lower:
                actions.append({"tool": "change_map_style", "style": style_value, "requires_frontend": True})
                break  # Only one style at a time
        
        # View mode
        if "3d" in msg_lower or "three dimensional" in msg_lower:
            # Check if current style doesn't support 3D
            current_style = map_state.get("currentMapStyle", "")
            print(f"Checking 3D compatibility - Current style: '{current_style}'")
            non_3d_styles = ["Dark", "Light", "Outdoors", "Navigation"]
            
            if any(style in current_style for style in non_3d_styles):
                # Need to change style to default first, then switch to 3D
                info_msg = f"Note: {current_style} doesn't support 3D terrain. Switching to Default style."
                print(f"Style conflict detected, adding style change action")
                
                # Add style change action first
                actions.append({"tool": "change_map_style", "style": "default", "requires_frontend": True})
                # Then add view mode change with info message
                actions.append({"tool": "switch_view_mode", "mode": "3d", "requires_frontend": True, "info_message": info_msg})
            else:
                print(f"Style '{current_style}' supports 3D, no style change needed")
                actions.append({"tool": "switch_view_mode", "mode": "3d", "requires_frontend": True})
        elif "2d" in msg_lower or "two dimensional" in msg_lower or "flat" in msg_lower:
            actions.append({"tool": "switch_view_mode", "mode": "2d", "requires_frontend": True})
        
        # Return multiple actions if found
        if len(actions) > 1:
            # Prioritize actions in this order for best UX:
            # 1. View mode changes (2D/3D) - affects rendering
            # 2. Style changes - visual appearance
            # 3. Location searches - camera movement (should be last to not be interrupted)
            view_actions = [a for a in actions if a.get("tool") == "switch_view_mode"]
            style_actions = [a for a in actions if a.get("tool") == "change_map_style"]
            location_actions = [a for a in actions if a.get("tool") == "search_location"]
            other_actions = [a for a in actions if a.get("tool") not in ["switch_view_mode", "change_map_style", "search_location"]]
            
            sorted_actions = view_actions + style_actions + other_actions + location_actions
            return {"multiple_actions": sorted_actions, "requires_frontend": True}
        elif len(actions) == 1:
            return actions[0]
        
        return None


