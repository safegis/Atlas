"""Map agent for handling location, style, and view mode"""
import json
import re
from typing import Optional
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
        has_earthquake_source = any(
            word in msg_lower
            for word in [
                "global",
                "usgs",
                "philippine",
                "philippines",
                "phivolcs",
                "worldwide",
                "world",
                "both",
                "all",
            ]
        )
        
        if has_earthquake and not has_earthquake_source:
            # Never run location LLM on "earthquake" — it returns search_location("earthquake").
            print(
                "Earthquake without source — hazard clarification (skipping map location LLM)"
            )
            ambiguous_eq = {
                "type": "clarification",
                "question": "I found 2 earthquake data sources. Which one would you like to see?",
                "options": [
                    "Philippines - Latest earthquake data from PHIVOLCS (Philippine Institute of Volcanology and Seismology)",
                    "Global - Worldwide earthquake data from USGS (U.S. Geological Survey)",
                    "Both - Show both Philippine and Global earthquake data",
                ],
                "suggested_action": {
                    "tool": "control_earthquake_data",
                    "action": "enable",
                    "source": "philippine",
                },
            }
            state["clarification_needed"] = True
            state["pending_action"] = ambiguous_eq
            state["messages"].append(
                AIMessage(
                    content=json.dumps(
                        {
                            "type": "clarification",
                            "question": ambiguous_eq["question"],
                            "options": ambiguous_eq["options"],
                            "suggested_action": ambiguous_eq["suggested_action"],
                        }
                    )
                )
            )
            return state
        
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
    
    def _normalize_step_message(self, message: str) -> str:
        """Extract the main intent from step-by-step prompts (e.g. 'then add Philippines boundaries' -> 'add Philippines boundaries')."""
        msg_lower = message.lower().strip()
        for prefix in ("then ", "now ", "next ", "after that ", "and then ", "and also ", "also "):
            if msg_lower.startswith(prefix):
                return msg_lower[len(prefix):].strip()
        return msg_lower

    def _parse_history_control_intent(self, message: str) -> Optional[dict]:
        """
        Undo / redo / reset map — rule-based so Atlas handles many phrasings without LLM drift.
        Router should send map questions elsewhere; avoid matching 'don't undo' style negation.
        """
        raw = message.strip()
        msg_lower = raw.lower()
        nl = self._normalize_step_message(message).lower()
        blob = f"{msg_lower} {nl}"

        # Negations: user is refusing undo/redo, not requesting it
        if re.search(r"\b(don't|do not|dont|never|without)\s+undo\b", blob):
            return None
        if re.search(r"\b(don't|do not|dont|never)\s+redo\b", blob):
            return None

        # --- Undo BEFORE redo so phrases like "undo the redo" prefer undo ---
        undo_phrases = (
            "undo that",
            "undo the last",
            "undo last",
            "undo my last",
            "undo it",
            "please undo",
            "can you undo",
            "revert",
            "revert that",
            "revert last",
            "revert the last",
            "rollback",
            "roll back",
            "roll-back",
            "take that back",
            "go back one step",
            "previous map state",
            "last map state",
            "step back",
            "ctrl+z",
            "ctrl z",
            "control+z",
            "control z",
            "keyboard undo",
            "history undo",
            "map undo",
            "undo the map",
            "undo map change",
        )
        if any(p in blob for p in undo_phrases) or re.search(r"\bundo\b", msg_lower):
            return {
                "tool": "map_undo",
                "requires_frontend": True,
                "text": "Undoing the last map change.",
            }

        # --- Redo ---
        redo_phrases = (
            "redo that",
            "redo the last",
            "redo last",
            "redo my last",
            "redo it",
            "please redo",
            "can you redo",
            "repeat the last action",
            "repeat last action",
            "restore what i undid",
            "restore what i undone",
            "bring that back",
            "bring it back",
            "go forward",
            "step forward",
            "ctrl+y",
            "ctrl y",
            "control+y",
            "control y",
            "keyboard redo",
            "history redo",
            "map redo",
            "redo the map",
            "redo map change",
            "reapply",
            "re-apply",
        )
        if any(p in blob for p in redo_phrases) or re.search(r"\bredo\b", msg_lower):
            return {
                "tool": "map_redo",
                "requires_frontend": True,
                "text": "Redoing the last undone map change.",
            }

        # --- Reset map (opens confirm modal by default; immediate skips modal) ---
        reset_phrases = (
            "reset map",
            "reset the map",
            "map reset",
            "global reset",
            "reset everything on the map",
            "clear the map",
            "clear map",
            "clear entire map",
            "wipe the map",
            "wipe map",
            "erase the map",
            "erase map",
            "empty the map",
            "blank the map",
            "start over on the map",
            "start fresh on the map",
            "fresh map",
            "default map",
            "restore default map",
            "restore map to default",
            "clear all map layers",
            "remove everything from the map",
            "remove all from the map",
            "reset map view",
            "reset the canvas",
            "clear canvas",
            "factory reset map",
            "hard reset map",
            "hit reset",
            "press reset",
            "click reset",
            "use the reset button",
            "tap reset",
            "reset button",
            "clear drawings",
            "clear drawing on map",
            "clear map content",
        )
        immediate_kw = (
            "immediately",
            "right away",
            "without asking",
            "without confirmation",
            "skip confirmation",
            "skip the dialog",
            "skip dialog",
            "no confirmation",
            "don't ask",
            "dont ask",
            "do not ask",
            "just reset",
            "force reset",
            "confirm reset for me",
            "auto confirm",
        )
        has_reset_word = bool(re.search(r"\breset\b", msg_lower))
        map_context = any(
            w in blob
            for w in (
                "map",
                "canvas",
                "layer",
                "layers",
                "drawing",
                "view",
                "gis",
                "simulation",
            )
        )
        reset_hit = any(p in blob for p in reset_phrases) or (
            has_reset_word and map_context
        )
        if reset_hit:
            immediate = any(k in blob for k in immediate_kw)
            if immediate:
                txt = "Resetting the map now (clearing content and restoring default basemap)."
            else:
                txt = "Opening the map reset confirmation — confirm to clear the map and restore defaults."
            return {
                "tool": "map_reset",
                "requires_frontend": True,
                "immediate": immediate,
                "text": txt,
            }

        return None

    def _parse_boundary_intent(self, message: str) -> Optional[dict]:
        """Parse add boundary commands. Supports step-by-step prompts and many phrasings."""
        # Normalize step-by-step: "then add Philippines boundaries" -> "add Philippines boundaries"
        msg_normalized = self._normalize_step_message(message)
        msg_lower = msg_normalized.lower()

        # Extract parameters
        source = None
        country = None
        admin_level = None

        # Default source for Simulation-Studio is geoBoundaries (only supported source in UI)
        source = "geoBoundaries"

        # Override if user explicitly mentions another source (for future use)
        if "gadm" in msg_lower:
            source = "GADM"
        elif "natural earth" in msg_lower or "naturalearth" in msg_lower:
            source = "Natural Earth"
        elif "geoboundaries" in msg_lower or "geo boundaries" in msg_lower:
            source = "geoBoundaries"
        
        # Detect country - common patterns
        country_patterns = {
            "afghanistan": "Afghanistan",
            "philippines": "Philippines",
            "philippine": "Philippines",
            "ph": "Philippines",
            "japan": "Japan",
            "china": "China",
            "indonesia": "Indonesia",
            "thailand": "Thailand",
            "vietnam": "Vietnam",
            "malaysia": "Malaysia",
            "singapore": "Singapore",
            "united states": "United States",
            "usa": "United States",
            "us": "United States",
            "america": "United States",
            "canada": "Canada",
            "mexico": "Mexico",
            "uk": "United Kingdom",
            "united kingdom": "United Kingdom",
            "australia": "Australia",
            "india": "India",
            "brazil": "Brazil",
            "france": "France",
            "germany": "Germany",
            "italy": "Italy", 
            "spain": "Spain",
        }
        
        # Check patterns with word boundaries to avoid false matches
        # Sort by length (longest first) to match more specific patterns first
        for pattern, country_name in sorted(country_patterns.items(), key=lambda x: len(x[0]), reverse=True):
            # Use word boundaries to ensure exact matches
            import re
            if re.search(r'\b' + re.escape(pattern) + r'\b', msg_lower):
                country = country_name
                break
        
        # Detect admin level
        # Patterns: "admin level 3", "level 3", "adm3", "admin 3", "ADM3"
        import re
        
        # Try "admin level X" or "level X"
        level_match = re.search(r'(?:admin\s+)?level\s+(\d)', msg_lower)
        if level_match:
            admin_level = int(level_match.group(1))
        else:
            # Try "admX" or "admin X"
            adm_match = re.search(r'adm(?:in)?\s*(\d)', msg_lower)
            if adm_match:
                admin_level = int(adm_match.group(1))
        
        # Map admin level descriptions to numbers (admin 0 = country, 1 = state/province/region, etc.)
        level_descriptions = {
            "country": 0,
            "national": 0,
            "province": 1,
            "provinces": 1,
            "provincial": 1,
            "region": 1,
            "regions": 1,
            "regional": 1,
            "state": 1,
            "states": 1,
            "admin level 1": 1,
            "level 1": 1,
            "district": 2,
            "districts": 2,
            "admin level 2": 2,
            "level 2": 2,
            "municipality": 3,
            "municipalities": 3,
            "city": 3,
            "cities": 3,
            "admin level 3": 3,
            "level 3": 3,
            "barangay": 4,
            "admin level 4": 4,
            "level 4": 4,
        }
        if admin_level is None:
            for desc, level in level_descriptions.items():
                if desc in msg_lower:
                    admin_level = level
                    break
        
        # Build response (source is always geoBoundaries by default)
        return {
            "tool": "add_boundary",
            "source": source,
            "country": country,
            "admin_level": admin_level,
            "requires_frontend": True,
            "text": f"Adding boundaries{f' for {country}' if country else ''}{f' at admin level {admin_level}' if admin_level is not None else ''}. Opening the Add Boundaries panel."
            if (country or admin_level is not None)
            else "Opening the Add Boundaries panel. Select a country and admin level (e.g. by province).",
        }
    
    def _detect_ambiguous(self, message: str) -> Optional[dict]:
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

    def _parse_intent(self, message: str, map_state: dict) -> Optional[dict]:
        """Parse map intents using LLM for natural language understanding"""
        
        msg_lower = message.lower()

        # Map history: undo / redo / reset (before boundary clear — different from "clear boundaries")
        hc = self._parse_history_control_intent(message)
        if hc:
            return hc
        
        # IMPORTANT: Check for "clear boundaries" commands FIRST, before any LLM calls
        # This prevents the LLM from misinterpreting the command
        # Check for "clear" + "boundary/boundaries/border/borders" (with optional words in between like "all")
        has_clear = any(word in msg_lower for word in ["clear", "remove", "delete"])
        has_boundary = any(word in msg_lower for word in ["boundary", "boundaries", "border", "borders"])
        has_clear_boundary = has_clear and has_boundary
        
        if has_clear_boundary:
            return {
                "tool": "clear_boundary",
                "requires_frontend": True,
                "text": "Clearing all boundaries from the map."
            }

        # Check if this is an "open panel" command for layers
        open_keywords = ["open", "show", "display"]
        is_open_command = any(keyword in msg_lower for keyword in open_keywords)
        
        # Check for Critical Facility Layers panel
        mentions_critical_facility = ("critical facility" in msg_lower or "critical facilities" in msg_lower) and "layer" in msg_lower
        
        if is_open_command and mentions_critical_facility and "panel" in msg_lower:
            # User wants to open the critical facility layers panel
            return {
                "tool": "open_layers_panel",
                "action": "show_critical_facility_layers",
                "requires_frontend": True,
                "text": "Opening the Critical Facility Layers panel. You can now view and manage critical facility layers on the map."
            }
        
        # Check for "add/show/display/load boundaries" commands (including step-by-step: "then add ...")
        add_boundary_keywords = [
            "add boundary", "add boundaries", "add border", "add borders",
            "show boundary", "show boundaries", "show border", "show borders",
            "display boundary", "display boundaries", "display border", "display borders",
            "load boundary", "load boundaries", "load border", "load borders",
            "put boundary", "put boundaries", "put border", "put borders",
            "draw boundary", "draw boundaries", "open boundary", "open boundaries",
            "add boundaries to the map", "add boundaries to map", "boundaries panel",
            "boundaries for", "borders for", "boundaries of", "borders of",
            "country boundary", "country boundaries", "national border", "national borders",
            "administrative boundary", "administrative boundaries",
            "by province", "by region", "by municipality", "province level", "regional boundary",
        ]
        # Use normalized message so "then add Philippines boundaries" is detected
        msg_for_boundary = self._normalize_step_message(message)
        msg_boundary_lower = msg_for_boundary.lower()
        has_add_boundary = any(kw in msg_boundary_lower for kw in add_boundary_keywords)
        # Also trigger if message clearly asks for boundaries of a place (e.g. "Philippines boundaries", "show me Japan's borders")
        if not has_add_boundary and ("boundary" in msg_boundary_lower or "boundaries" in msg_boundary_lower or "border" in msg_boundary_lower or "borders" in msg_boundary_lower):
            action_verbs = ["add", "show", "display", "load", "put", "draw", "open", "see", "view", "get", "want"]
            if any(v in msg_boundary_lower for v in action_verbs) or "for " in msg_boundary_lower or " of " in msg_boundary_lower:
                has_add_boundary = True

        if has_add_boundary:
            boundary_action = self._parse_boundary_intent(message)
            if boundary_action:
                return boundary_action
        
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

6. **map_undo** - Undo last map change (camera, layers, style, etc.)
7. **map_redo** - Redo last undone change
8. **map_reset** - Clear map / reset to default basemap. Set immediate: true only if user explicitly wants no confirmation dialog (e.g. "reset map immediately", "skip confirmation").

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
- "undo" / "go back" (map) → [{"tool": "map_undo"}]
- "redo" → [{"tool": "map_redo"}]
- "reset the map" → [{"tool": "map_reset", "immediate": false}]
- "reset map without asking" → [{"tool": "map_reset", "immediate": true}]

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
                elif tool == "map_undo":
                    actions.append({"tool": "map_undo", "requires_frontend": True, "text": "Undoing the last map change."})
                elif tool == "map_redo":
                    actions.append({"tool": "map_redo", "requires_frontend": True, "text": "Redoing the last undone map change."})
                elif tool == "map_reset":
                    actions.append({
                        "tool": "map_reset",
                        "requires_frontend": True,
                        "immediate": bool(action.get("immediate", False)),
                        "text": action.get("text") or "",
                    })
            
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


