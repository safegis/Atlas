"""Hazard agent for handling earthquake and weather data"""
import json
from typing import Any, Optional

from langchain_core.messages import AIMessage

from agents.router_agent import _is_affirmation, _prev_qa_offered_map_hazard_demo
from state import AgentState
from tools import (
    control_earthquake_data,
    control_tsunami_data,
    control_weather_data,
)

# Tsunami-only clarification labels. Keep in sync with
# Simulation-Studio/frontend/src/components/controls/Main/TsunamiDataConfig.ts (tsunamiData).
LIVE_TSUNAMI_FEED_OPTIONS = [
    "1 — PHIVOLCS Tsunami Information (Philippines / Pacific — official bulletins)",
]


class HazardAgent:
    """Handles hazard monitoring: earthquake and weather data"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [control_earthquake_data, control_weather_data]
        # Hazard agent uses rule-based intent parsing, not LLM prompts

    @staticmethod
    def _last_assistant_json_payload(messages: list) -> Optional[dict[str, Any]]:
        """Most recent assistant message parsed as JSON (QA / clarification cards)."""
        for i in range(len(messages) - 1, -1, -1):
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
                return json.loads(content)
            except Exception:
                continue
        return None

    def _expand_affirmation_after_qa_demo(self, messages: list) -> Optional[str]:
        """Turn 'yes' after a QA earthquake/weather demo offer into an explicit enable command."""
        if len(messages) < 2:
            return None
        raw_last = messages[-1].content if hasattr(messages[-1], "content") else ""
        if not isinstance(raw_last, str) or not _is_affirmation(raw_last.lower()):
            return None
        prev = self._last_assistant_json_payload(messages[:-1])
        if not prev or not _prev_qa_offered_map_hazard_demo(prev):
            return None
        text = prev.get("text")
        if not isinstance(text, str):
            return None
        # Model sometimes double-encodes JSON inside "text"
        if text.strip().startswith("{"):
            try:
                inner = json.loads(text)
                if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                    text = inner["text"]
            except Exception:
                pass
        tl = text.lower()
        if "tsunami" in tl and not any(
            x in tl for x in ("earthquake", "quake", "weather", "temperature")
        ):
            return "enable phivolcs tsunami data on the map"
        if "weather" in tl and not any(
            x in tl for x in ("earthquake", "quake", "phivolcs", "usgs", "seismic")
        ):
            return "enable weather data for the Philippines"
        if "phivolcs" in tl or (
            "philippine" in tl and ("earthquake" in tl or "quake" in tl)
        ):
            return "enable philippine earthquake data from PHIVOLCS"
        if "usgs" in tl or ("global" in tl and "earthquake" in tl):
            return "enable global earthquake data from USGS"
        return "enable earthquake data"

    def process(self, state: AgentState) -> AgentState:
        """Process hazard monitoring requests"""
        messages = state["messages"]
        last_message = messages[-1].content
        expanded = self._expand_affirmation_after_qa_demo(messages)
        if expanded:
            print(f"Hazard Agent: affirmation expanded to hazard command: {expanded!r}")
            last_message = expanded
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
                "suggested_action": ambiguous.get("suggested_action"),
            }
            ck = ambiguous.get("clarification_kind")
            if ck:
                clarification_response["clarification_kind"] = ck
            
            # If there are map actions (not clarifications), store them as pending
            if map_action and map_action.get("type") != "clarification":
                if self._should_chain_map_with_hazard_clarification(map_action):
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
            # Compound request - flatten and prioritize actions for best UX
            print(f"Detected compound request with map action: {map_action}")
            
            hazard_actions = []
            map_actions_list = []
            
            # Collect hazard action(s)
            if action.get("also_enable"):
                # Dual source earthquake (both Philippine and Global)
                hazard_actions.append(action)
            else:
                hazard_actions.append(action)
            
            # Collect map action(s) - flatten if nested
            if map_action.get("multiple_actions"):
                # Map action is already multiple actions - flatten them
                map_actions_list = map_action["multiple_actions"]
            else:
                # Single map action
                map_actions_list = [map_action]
            
            # Prioritize actions: view mode > style > hazards > location (location LAST to avoid interruption)
            view_actions = [a for a in map_actions_list if a.get("tool") == "switch_view_mode"]
            style_actions = [a for a in map_actions_list if a.get("tool") == "change_map_style"]
            location_actions = [a for a in map_actions_list if a.get("tool") == "search_location"]
            other_map_actions = [a for a in map_actions_list if a.get("tool") not in ["switch_view_mode", "change_map_style", "search_location"]]
            
            # Final order: view > style > other map > hazards > location
            all_actions = view_actions + style_actions + other_map_actions + hazard_actions + location_actions
            
            state["messages"].append(AIMessage(content=json.dumps({
                "multiple_actions": all_actions,
                "requires_frontend": True
            })))
        elif action:
            state["messages"].append(AIMessage(content=json.dumps(action)))
        else:
            state["messages"].append(AIMessage(
                content=json.dumps({
                    "text": "I can help you enable earthquake, tsunami, or weather monitoring. What would you like to see?"
                })
            ))
        
        return state

    @staticmethod
    def _should_chain_map_with_hazard_clarification(map_action: dict) -> bool:
        """Avoid chaining bogus search_location(earthquake...) onto earthquake source clarifications."""
        if not isinstance(map_action, dict):
            return True
        bad_q = (
            "earthquake",
            "seismic",
            "quake",
            "hazard",
            "phivolcs",
            "usgs",
            "tsunami",
        )

        def _bad_search(d: dict) -> bool:
            if d.get("tool") != "search_location":
                return False
            q = (d.get("query") or "").lower()
            return any(w in q for w in bad_q)

        if map_action.get("multiple_actions"):
            return not any(
                _bad_search(a) for a in map_action["multiple_actions"] if isinstance(a, dict)
            )
        if _bad_search(map_action):
            return False
        return True
    
    def _parse_map_intent(self, message: str, map_state: dict = None) -> Optional[dict]:
        """Parse map-related intents in hazard requests - returns action(s) or clarification"""
        msg_lower = message.lower()
        actions = []
        if map_state is None:
            map_state = {}
        
        # Check for location search keywords
        location_keywords = ["show", "go to", "fly to", "fly the map to", "navigate to", "zoom to", "zoom the map to", "search for", "find", "locate"]
        has_location_request = any(keyword in msg_lower for keyword in location_keywords)
        # "show ... earthquake ... on the map" is hazard wording, not search_location("earthquake")
        hazard_location_false_positive = any(
            w in msg_lower
            for w in ("earthquake", "seismic", "quake", "phivolcs", "usgs", "tsunami")
        )
        
        # If location request detected, extract the location query
        if has_location_request and not hazard_location_false_positive:
            # Try to extract location from common patterns
            for keyword in location_keywords:
                if keyword in msg_lower:
                    # Split by the keyword and take the part after it
                    parts = msg_lower.split(keyword, 1)
                    if len(parts) > 1:
                        location_part = parts[1].strip()
                        # Remove common trailing phrases
                        location_part = location_part.split(" in the map")[0].strip()
                        location_part = location_part.split(" on the map")[0].strip()
                        location_part = location_part.split(", then")[0].strip()
                        location_part = location_part.split(" then")[0].strip()
                        location_part = location_part.split(" and ")[0].strip()
                        
                        if location_part and not any(
                            w in location_part
                            for w in ("earthquake", "seismic", "quake", "hazard", "phivolcs", "usgs")
                        ):
                            actions.append({"tool": "search_location", "query": location_part, "requires_frontend": True})
                            break
        
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
    
    @staticmethod
    def _explicit_tsunami_enable(msg_lower: str) -> bool:
        """User named the actual tsunami data product or layer — OK to enable without a picker."""
        return (
            "phivolcs tsunami" in msg_lower
            or "tsunami bulletin" in msg_lower
            or "tsunami information" in msg_lower
            or "tsunami layer" in msg_lower
            or "tsunami feed" in msg_lower
        )

    @staticmethod
    def _needs_tsunami_feed_clarification(msg_lower: str, is_disable: bool) -> bool:
        """User mentioned tsunami but not a specific product/layer — list tsunami feeds only."""
        if is_disable:
            return False
        has_tsunami = "tsunami" in msg_lower
        has_quake = any(
            w in msg_lower
            for w in ("earthquake", "earthquakes", "seismic", "quake")
        )
        return (
            has_tsunami
            and not has_quake
            and not HazardAgent._explicit_tsunami_enable(msg_lower)
        )

    @staticmethod
    def _needs_generic_live_hazard_feed_clarification(
        msg_lower: str, is_disable: bool
    ) -> bool:
        """
        Generic 'live hazard(s)' with no feed type named — full hazard picker
        (earthquake / tsunami / weather).
        """
        if is_disable:
            return False
        has_tsunami = "tsunami" in msg_lower
        has_quake = any(
            w in msg_lower
            for w in ("earthquake", "earthquakes", "seismic", "quake")
        )
        if "live" not in msg_lower or "hazard" not in msg_lower:
            return False
        if has_tsunami or has_quake:
            return False
        if any(
            w in msg_lower
            for w in ("weather", "temperature", "climate")
        ):
            return False
        if "monitor" in msg_lower or "panel" in msg_lower:
            return False
        return True

    def _detect_ambiguous(self, message: str) -> Optional[dict]:
        """Detect ambiguous hazard requests that need clarification"""
        msg_lower = message.lower()
        
        # Determine if this is an enable or disable request
        disable_keywords = ["disable", "hide", "turn off", "deactivate", "remove"]
        is_disable = any(word in msg_lower for word in disable_keywords)
        
        # Vague tsunami — same pattern as earthquake: only list tsunami feeds (not weather/quake)
        if self._needs_tsunami_feed_clarification(msg_lower, is_disable):
            return {
                "type": "clarification",
                "clarification_kind": "tsunami_sources",
                "question": (
                    "These tsunami data feeds are available. Which would you like to show on the map?"
                ),
                "options": list(LIVE_TSUNAMI_FEED_OPTIONS),
                "suggested_action": {
                    "tool": "control_tsunami_data",
                    "action": "enable",
                    "requires_frontend": True,
                },
            }

        # Generic live hazard(s) — user did not name earthquake, tsunami, or weather
        if self._needs_generic_live_hazard_feed_clarification(msg_lower, is_disable):
            return {
                "type": "clarification",
                "clarification_kind": "live_hazard_feeds",
                "question": (
                    "Several live hazard feeds are available. Which would you like to show on the map?"
                ),
                "options": [
                    "1 — Philippines: earthquakes (PHIVOLCS)",
                    "2 — Global: earthquakes (USGS)",
                    "3 — PHIVOLCS tsunami bulletin / information",
                    "4 — Weather (province, city, or all — you can narrow in the Live Hazard Monitor)",
                ],
                "suggested_action": {
                    "tool": "control_earthquake_data",
                    "action": "enable",
                    "source": "philippine",
                },
            }
        
        # Check if requesting earthquake data without specifying source
        if any(word in msg_lower for word in ["earthquake", "seismic", "quake"]):
            # If they explicitly mention global/usgs or philippine/phivolcs, it's not ambiguous
            has_global = "global" in msg_lower or "usgs" in msg_lower or "worldwide" in msg_lower or "world" in msg_lower
            has_philippine = "philippine" in msg_lower or "philippines" in msg_lower or "phivolcs" in msg_lower or "local" in msg_lower
            has_both = "both" in msg_lower or "all" in msg_lower
            
            # If neither mentioned AND not "both", ask for clarification
            # If "both" is mentioned, it's clear - enable/disable both (not ambiguous)
            # Only ask for clarification on enable requests - for disable, we can disable all
            if not has_global and not has_philippine and not has_both and not is_disable:
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
    
    def _parse_intent(self, message: str) -> Optional[dict]:
        """Parse hazard monitoring intents using LLM for natural language understanding"""
        
        # Check if this is just an "open panel" command
        msg_lower = message.lower()
        open_keywords = ["open", "show", "display"]
        
        is_open_command = any(keyword in msg_lower for keyword in open_keywords)
        
        # Check for Live Hazard Monitor (live earthquake/weather monitoring)
        mentions_live_monitor = ("live" in msg_lower and "hazard" in msg_lower) or "live hazard monitor" in msg_lower
        
        # Check for Hazard Layers panel (static layers in map layers)
        mentions_hazard_layers = ("hazard" in msg_lower and "layer" in msg_lower) or "hazard layers" in msg_lower
        
        if is_open_command and mentions_live_monitor and "panel" in msg_lower:
            # User wants to open the live hazard monitor panel
            return {
                "tool": "open_live_hazard_monitor",
                "action": "show_panel",
                "requires_frontend": True,
                "text": "Opening the Live Hazard Monitor panel. You can now enable earthquake, tsunami, and weather monitoring."
            }
        elif is_open_command and mentions_hazard_layers and "panel" in msg_lower:
            # User wants to open the hazard layers in the map layers panel
            return {
                "tool": "open_layers_panel",
                "action": "show_hazard_layers",
                "requires_frontend": True,
                "text": "Opening the Hazard Layers panel. You can now view and manage hazard layers on the map."
            }

        # Tsunami — disable can stay broad; enable only when the product/layer is explicit.
        # Vague tsunami is clarified in _detect_ambiguous before we get here.
        has_quake = any(
            w in msg_lower
            for w in ("earthquake", "earthquakes", "seismic", "quake")
        )
        if "tsunami" in msg_lower and not has_quake:
            wants_disable = any(
                w in msg_lower
                for w in (
                    "disable",
                    "hide",
                    "turn off",
                    "remove",
                    "clear",
                )
            )
            if wants_disable:
                return {
                    "tool": "control_tsunami_data",
                    "action": "disable",
                    "requires_frontend": True,
                }
            if self._explicit_tsunami_enable(msg_lower):
                return {
                    "tool": "control_tsunami_data",
                    "action": "enable",
                    "requires_frontend": True,
                }

        system_prompt = """You are a hazard monitoring intent parser. Extract the user's intent from their message.

Available hazard actions:
1. **control_earthquake_data** - Enable/disable earthquake monitoring
   - action: "enable" or "disable"
   - source: "philippine" (PHIVOLCS), "global" (USGS), or "both"
   - Examples: "show earthquakes" → enable, "Philippine earthquakes" → philippine

2. **control_tsunami_data** - Enable/disable PHIVOLCS tsunami bulletin layer (Live Hazard Monitor)
   - action: "enable" or "disable"
   - Enable only when the user names the bulletin/layer/PHIVOLCS tsunami (vague "tsunami hazard" is clarified elsewhere).
   - Examples: "enable PHIVOLCS tsunami bulletin" → enable, "hide tsunami" → disable

3. **control_weather_data** - Enable/disable weather monitoring
   - action: "enable" or "disable"
   - scope: "province" (Philippines-wide), "city" (specific), or "all"
   - province: (if scope=city) "Abra", "Agusan del Norte", "Agusan del Sur", "Aklan"
   - Examples: "show weather" → province, "weather in Abra" → city + Abra

Default to "enable" if action is unclear (e.g., "show", "display").

Respond with JSON object: {"tool": "...", "action": "...", "source/scope": "...", "province": "..."}

Examples:
- "enable earthquake data" → {"tool": "control_earthquake_data", "action": "enable", "source": "philippine"}
- "show global earthquakes" → {"tool": "control_earthquake_data", "action": "enable", "source": "global"}
- "disable all earthquakes" → {"tool": "control_earthquake_data", "action": "disable", "source": "both"}
- "show PHIVOLCS tsunami bulletin" → {"tool": "control_tsunami_data", "action": "enable"}
- "disable tsunami layer" → {"tool": "control_tsunami_data", "action": "disable"}
- "show weather" → {"tool": "control_weather_data", "action": "enable", "scope": "province"}
- "weather in Aklan" → {"tool": "control_weather_data", "action": "enable", "scope": "city", "province": "Aklan"}

Return ONLY the JSON object, no explanation."""

        try:
            # Use LLM to parse intent
            response = self.llm.invoke(message, system_prompt=system_prompt)
            print(f"LLM parsed hazard intent: {response[:200]}")
            
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            parsed = json.loads(response)
            
            # Convert to internal format
            tool = parsed.get("tool")
            action = parsed.get("action", "enable")
            
            if tool == "control_earthquake_data":
                source = parsed.get("source", "philippine")
                
                if source == "both":
                    return {
                        "tool": "control_earthquake_data",
                        "action": action,
                        "source": "philippine",
                        "requires_frontend": True,
                        "also_enable": {"tool": "control_earthquake_data", "action": action, "source": "global"}
                    }
                else:
                    return {"tool": "control_earthquake_data", "action": action, "source": source, "requires_frontend": True}
            
            elif tool == "control_tsunami_data":
                return {
                    "tool": "control_tsunami_data",
                    "action": action,
                    "requires_frontend": True,
                }

            elif tool == "control_weather_data":
                scope = parsed.get("scope", "province")
                province = parsed.get("province")
                
                result = {"tool": "control_weather_data", "action": action, "scope": scope, "requires_frontend": True}
                if province:
                    result["province"] = province
                return result
            
            return None
            
        except Exception as e:
            print(f"Error parsing hazard intent with LLM: {e}")
            # Fallback to simple keyword matching
            msg_lower = message.lower()
            
            if "earthquake" in msg_lower or "seismic" in msg_lower:
                action = "disable" if "disable" in msg_lower or "hide" in msg_lower else "enable"
                source = "global" if "global" in msg_lower or "usgs" in msg_lower else "philippine"
                return {"tool": "control_earthquake_data", "action": action, "source": source, "requires_frontend": True}

            if any(
                t in msg_lower
                for t in ("tsunami", "tsunami bulletin", "phivolcs tsunami")
            ):
                action = "disable" if "disable" in msg_lower or "hide" in msg_lower else "enable"
                return {
                    "tool": "control_tsunami_data",
                    "action": action,
                    "requires_frontend": True,
                }
            
            if "weather" in msg_lower:
                action = "disable" if "disable" in msg_lower or "hide" in msg_lower else "enable"
                return {"tool": "control_weather_data", "action": action, "scope": "province", "requires_frontend": True}
            
            return None


