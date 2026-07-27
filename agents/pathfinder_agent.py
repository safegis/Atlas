"""Pathfinder agent for route planning and navigation"""
import json
import re
from typing import Any, Optional

from langchain_core.messages import AIMessage
from agents.router_agent import (
    _wants_pathfinder_clear_routes,
    _wants_pathfinder_tab_switch,
    _looks_like_pathfinder_radius_followup,
    _parse_radius_km_from_message,
    _previous_evacuation_find_route,
)
from tools.pathfinder_tools import PATHFINDER_TOOLS
from state import AgentState

# Phrases where start/destination order is unclear (ask user to pick).
_BETWEEN_AMBIGUOUS_SUBSTRINGS = (
    "between 2 locations",
    "between two locations",
    "between these locations",
    "between the following",
    "between the two",
    "between both",
    "between them",
    "between this and that",
    "two locations",
    "two places",
    "either location",
)

_CANCEL_PHRASES = frozenset(
    {
        "cancel",
        "never mind",
        "nevermind",
        "stop",
        "forget it",
        "abort",
        "no thanks",
        "nvm",
        "quit",
        "dismiss",
    }
)

_SWAP_PHRASES = (
    "swap",
    "flip start",
    "reverse route",
    "the other way",
    "opposite direction",
    "invert",
    "switch start and end",
    "swap origin",
)


def _wants_shelter_list(msg_lower: str) -> bool:
    """
    User wants names of shelters already loaded in Pathfinder (chat list),
    not a new OSM search / find_route.
    """
    # "Find/show shelters near <place>" is a new search — not listing the dropdown
    if re.search(
        r"\b(find|show|search|open|get)\b.{0,120}\bshelters?\b.{0,80}\b(near|around|close to|by)\b",
        msg_lower,
    ):
        return False
    has_list_intent = any(
        p in msg_lower
        for p in (
            "list ",
            "list all",
            " list ",
            "enumerate",
            "what shelters",
            "which shelters",
            "name the",
            "all nearest",
            "all the nearest",
            "show me all",
            "tell me all",
        )
    ) or msg_lower.startswith("list ")
    if not has_list_intent:
        return False
    return any(
        w in msg_lower
        for w in (
            "shelter",
            "shelters",
            "evacuation",
            "evacuate",
            "dropdown",
            "starting point",
            "current starting",
            "pathfinder",
        )
    )


def _wants_select_evac_destination_control(msg_lower: str) -> bool:
    """True when user is asking to pick / change evacuation or shelter destination (mirrors router)."""
    if "shelter" in msg_lower or "evacuation" in msg_lower:
        if any(
            p in msg_lower
            for p in (
                "select ",
                "choose ",
                "switch to ",
                "use ",
                "pick ",
                "set ",
                "change to ",
                "change ",
                "update ",
            )
        ):
            return True
    if "destination" in msg_lower:
        if any(
            p in msg_lower
            for p in (
                "change the destination",
                "change destination",
                "set the destination",
                "set destination",
                "switch the destination",
                "switch destination",
                "update destination",
                "select destination",
                "pick destination",
                "current destination",
                "destination to ",
                "destination as ",
                "evacuation destination",
            )
        ):
            return True
    return False


def _heuristic_select_evacuation_destination(message: str) -> Optional[dict]:
    """Extract place name from 'destination to \"X\"' / quotes / trailing 'to X'."""
    raw = (message or "").strip()
    if len(raw) < 4:
        return None
    m = re.search(r'["\u201c]([^"\u201d]+)["\u201d]', raw)
    if m:
        name = m.group(1).strip()
        if len(name) >= 2:
            return {"tool": "select_evacuation_destination", "place_name": name, "mode": "all"}
    m = re.search(
        r'(?is)\bdestination\b.{0,80}\bto\s+(.+?)\s*(?:[.!?]|$)',
        raw,
    )
    if m:
        name = m.group(1).strip().strip('"').strip("'").strip()
        if len(name) >= 2 and len(name) <= 220:
            return {"tool": "select_evacuation_destination", "place_name": name, "mode": "all"}
    m = re.search(
        r'(?is)\bdestination\b.{0,40}\bas\s+(.+?)\s*(?:[.!?]|$)',
        raw,
    )
    if m:
        name = m.group(1).strip().strip('"').strip("'").strip()
        if len(name) >= 2 and len(name) <= 220:
            return {"tool": "select_evacuation_destination", "place_name": name, "mode": "all"}
    m = re.search(
        r'(?is)\bshelter\b.{0,80}\bto\s+(.+?)\s*(?:[.!?]|$)',
        raw,
    )
    if m:
        name = m.group(1).strip().strip('"').strip("'").strip()
        if len(name) >= 2 and len(name) <= 220:
            return {"tool": "select_evacuation_destination", "place_name": name, "mode": "all"}
    return None


def _heuristic_change_route_mode(message: str) -> Optional[dict]:
    """Regex fallback for 'travel mode to driving', 'mode to walk', etc."""
    raw = (message or "").strip()
    if len(raw) < 4:
        return None
    ml = raw.lower()
    if any(p in ml for p in ("what is", "what are", "how does", "explain ", "define ")):
        return None
    m = re.search(
        r"(?i)\b(?:travel\s+mode|transport(?:ation)?\s+mode|route\s+mode)\s+to\s+(\w+)",
        raw,
    )
    if not m:
        m = re.search(r"(?i)\bmode\s+to\s+(\w+)", raw)
    if not m:
        m = re.search(
            r"(?i)\b(?:switch|change|set)\s+to\s+(driving|walking|cycling|motorcycle|walk|bike|bicycle|car|all)\b",
            raw,
        )
    if not m:
        return None
    token = m.group(1).lower().rstrip(".,!?")
    mode_map = {
        "driving": "driving",
        "car": "driving",
        "vehicle": "driving",
        "auto": "driving",
        "walking": "walking",
        "walk": "walking",
        "pedestrian": "walking",
        "foot": "walking",
        "cycling": "cycling",
        "cycle": "cycling",
        "bicycle": "cycling",
        "bike": "cycling",
        "biking": "cycling",
        "motorcycle": "motorcycle",
        "motorbike": "motorcycle",
        "all": "all",
        "modes": "all",
    }
    std = mode_map.get(token)
    if not std:
        return None
    return {"tool": "change_route_mode", "mode": std}


def _heuristic_set_pathfinder_tab(message: str) -> Optional[dict]:
    """Infer destination vs evacuation tab from natural language (when LLM returns prose)."""
    ml = (message or "").lower()
    if "pathfinder" not in ml:
        return None
    dest_ui = any(
        p in ml
        for p in (
            "set destination mode",
            "destination mode",
            "set destination tab",
            "point to point",
            "point-to-point",
            "address to address",
        )
    )
    evac_ui = any(
        p in ml
        for p in (
            "find shelter",
            "shelter mode",
            "evacuation mode",
            "evacuation tab",
            "shelter/s",
        )
    )
    if dest_ui and not evac_ui:
        return {"tool": "set_pathfinder_tab", "pathfinder_tab": "destination", "mode": "all"}
    if evac_ui and not dest_ui:
        return {"tool": "set_pathfinder_tab", "pathfinder_tab": "evacuation", "mode": "all"}
    return None


class PathfinderAgent:
    """Agent specialized in route planning and pathfinding"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = PATHFINDER_TOOLS
        
        self.system_prompt = """You are a pathfinding and route planning assistant. Your role is to help users:
1. Find routes between locations
2. Change transportation modes (driving, walking, cycling, motorcycle)
3. Sort routes by different criteria (fastest, safest, best balance)
4. Open/close the pathfinder interface

When users ask for routes:
- Extract the start and destination locations clearly
- Determine the transportation mode (default to "all" if not specified)
- Use find_route tool to calculate routes
- Provide helpful context about the route options

When users want to change modes or sorting:
- Use change_route_mode or change_route_sort tools
- Explain what the change will do

Common user requests:
- "Find a route from X to Y"
- "Show me the fastest route to X"
- "What's the safest way to get to Y"
- "Find walking directions to X"
- "Show me cycling routes"
- "Change to driving mode"
- "Sort by safest routes"

Always be clear and concise. If locations are ambiguous, ask for clarification.
"""

    @staticmethod
    def _heuristic_find_route(message: str) -> Optional[dict]:
        """
        Regex fallback for common 'from A to B' / 'between A and B' patterns when the LLM
        omits locations or JSON parsing fails.
        """
        raw = (message or "").strip()
        if len(raw) < 5:
            return None
        m = re.search(
            r"(?is)\bfrom\s+(.+?)\s+(?:to|towards?|toward|into)\s+(.+?)(?:\.|$)",
            raw,
        )
        if m:
            start, dest = m.group(1).strip().strip(",.;"), m.group(2).strip().strip(",.;")
            if len(start) >= 2 and len(dest) >= 2 and start.lower() != dest.lower():
                return {"tool": "find_route", "start": start, "destination": dest, "mode": "all"}
        m = re.search(
            r"(?is)\b(?:get|go|drive|walk|bike|cycle|ride)\s+(?:me\s+)?from\s+(.+?)\s+to\s+(.+?)(?:\.|$)",
            raw,
        )
        if m:
            start, dest = m.group(1).strip().strip(",.;"), m.group(2).strip().strip(",.;")
            if len(start) >= 2 and len(dest) >= 2:
                return {"tool": "find_route", "start": start, "destination": dest, "mode": "all"}
        m = re.search(r"(?is)\bbetween\s+(.+?)\s+and\s+(.+?)(?:\.|$)", raw)
        if m:
            a, b = m.group(1).strip().strip(",.;"), m.group(2).strip().strip(",.;")
            if len(a) >= 2 and len(b) >= 2 and a.lower() != b.lower():
                return {"tool": "find_route", "start": a, "destination": b, "mode": "all"}
        return None

    def _merge_heuristic_route(self, action: Optional[dict], message: str) -> Optional[dict]:
        """Fill missing start/destination on find_route using heuristics."""
        if not action or action.get("tool") != "find_route":
            return action
        # Evacuation/shelter flow intentionally leaves destination empty; do not let
        # "between X and Y" heuristics paste prose into destination (see long user prompts).
        ptab = (action.get("pathfinder_tab") or action.get("pathfinder_mode") or "").strip().lower()
        if ptab in ("evacuation", "shelter", "shelters") or action.get("evacuation_mode") is True:
            return action
        start = (action.get("start") or "").strip()
        dest = (action.get("destination") or "").strip()
        if start and dest and start.lower() not in ("current location", "here") and dest.lower() not in (
            "current location",
            "here",
        ):
            return action
        h = self._heuristic_find_route(message)
        if not h:
            return action
        if not start or start.lower() in ("current location", "here", "my location"):
            action["start"] = h.get("start") or start
        if not dest or dest.lower() in ("current location", "here", "my location"):
            action["destination"] = h.get("destination") or dest
        return action

    def process(self, state: AgentState) -> AgentState:
        """Process pathfinder requests"""
        messages = state["messages"]
        last_message = messages[-1].content
        
        print(f"Pathfinder Agent processing: {last_message}")
        
        # Check if we have a pending pathfinder action waiting for locations
        pending_action = state.get("pending_action")
        print(f"Pending action from state: {pending_action}")

        # Radius-only follow-up: reuse last evacuation start ("now within 2 km")
        msg_lower_early = last_message.lower()
        if _looks_like_pathfinder_radius_followup(msg_lower_early):
            radius_km = _parse_radius_km_from_message(msg_lower_early)
            prev_evac = _previous_evacuation_find_route(messages)
            start_reuse = ((prev_evac or {}).get("start") or "").strip()
            if radius_km is not None and start_reuse:
                mode = (prev_evac or {}).get("mode") or "all"
                action = {
                    "tool": "find_route",
                    "start": start_reuse,
                    "destination": "",
                    "pathfinder_tab": "evacuation",
                    "mode": mode,
                    "radius": radius_km,
                    "requires_frontend": True,
                    "action": self._get_action_type("find_route"),
                    "text": (
                        f"Updating shelter search near **{start_reuse}** to "
                        f"**{radius_km:g} km** — refreshing OSM schools and shelters."
                    ),
                }
                print(f"Radius follow-up action: {action}")
                state["messages"].append(AIMessage(content=json.dumps(action)))
                return state
            if radius_km is not None and not start_reuse:
                # No prior shelter search — ask for a place instead of falling through to LLM
                return self._request_locations(
                    state,
                    "",
                    "",
                    "all",
                    pathfinder_tab="evacuation",
                )
        
        # If no pending action in state, check conversation history for recent clarification
        if not pending_action or not isinstance(pending_action, dict):
            print("No pending action in state, checking conversation history")
            # Look for the last AI message with a clarification
            for i in range(len(messages) - 2, -1, -1):
                msg = messages[i]
                content = None
                msg_type = None
                
                if hasattr(msg, 'content'):
                    content = msg.content
                    msg_type = getattr(msg, 'type', None)
                elif isinstance(msg, dict):
                    content = msg.get('content')
                    msg_type = msg.get('role')
                
                if msg_type in ['ai', 'assistant'] and content:
                    try:
                        prev_content = json.loads(content)
                        if prev_content.get("type") == "clarification":
                            suggested_action = prev_content.get('suggested_action', {})
                            if suggested_action.get('tool') == 'find_route' and suggested_action.get('awaiting'):
                                print(f"Found pathfinder clarification in history: {suggested_action}")
                                pending_action = suggested_action
                                break
                    except:
                        pass
        
        if pending_action and isinstance(pending_action, dict):
            # Check if it's a pathfinder clarification
            if pending_action.get("type") == "pathfinder_awaiting_locations":
                print("Handling location response from pending action")
                return self._handle_location_response(state, last_message, pending_action)
            
            # Also check suggested_action for pathfinder awaiting
            suggested_action = pending_action.get("suggested_action", {})
            if suggested_action.get("awaiting") in ["start", "destination", "direction_choice"]:
                print("Handling location response from suggested_action")
                return self._handle_location_response(state, last_message, suggested_action)
            
            # Check if pending_action itself has awaiting field
            if pending_action.get("awaiting") in ["start", "destination", "direction_choice"]:
                print("Handling location response from pending_action directly")
                return self._handle_location_response(state, last_message, pending_action)
        
        # Parse the intent from the message (LLM + regex fallback / merge)
        action = self._parse_intent(last_message)
        if action:
            action = self._merge_heuristic_route(action, last_message)
        else:
            h = self._heuristic_find_route(last_message)
            if h:
                h["requires_frontend"] = True
                h["action"] = self._get_action_type("find_route")
                action = h
        if not action and _wants_shelter_list(last_message.lower()):
            action = {
                "tool": "list_evacuation_shelters",
                "requires_frontend": True,
                "action": "list_evacuation_shelters",
                "text": "Here are the shelters currently loaded in Pathfinder:",
            }
        print(f"Parsed action: {action}")

        # "List shelters in chat" uses data already in the Pathfinder UI (frontend snapshot)
        if action and _wants_shelter_list(last_message.lower()):
            if action.get("tool") != "list_evacuation_shelters":
                action = {
                    "tool": "list_evacuation_shelters",
                    "requires_frontend": True,
                    "action": "list_evacuation_shelters",
                    "text": "Here are the shelters currently loaded in Pathfinder:",
                }

        msg_lower_pf = last_message.lower()
        if _wants_select_evac_destination_control(msg_lower_pf):
            hs = _heuristic_select_evacuation_destination(last_message)
            if hs:
                if not action or action.get("tool") != "select_evacuation_destination":
                    action = hs
                elif not (action.get("place_name") or "").strip():
                    action["place_name"] = hs.get("place_name", "")
                action["requires_frontend"] = True
                action["action"] = self._get_action_type(
                    "select_evacuation_destination"
                )
                action.setdefault("mode", "all")

        if _wants_pathfinder_clear_routes(msg_lower_pf):
            action = {
                "tool": "clear_pathfinder_routes",
                "requires_frontend": True,
                "action": self._get_action_type("clear_pathfinder_routes"),
                "text": "Clearing routes from the map and resetting Pathfinder inputs.",
            }

        if _wants_pathfinder_tab_switch(msg_lower_pf):
            ht = _heuristic_set_pathfinder_tab(last_message)
            if ht and (
                not action
                or action.get("tool")
                in ("open_pathfinder", "close_pathfinder")
            ):
                action = dict(ht)
                action["requires_frontend"] = True
                action["action"] = self._get_action_type("set_pathfinder_tab")
                if action.get("pathfinder_tab") == "destination":
                    action.setdefault(
                        "text",
                        "Switching Pathfinder to **Set Destination** (point-to-point).",
                    )
                else:
                    action.setdefault(
                        "text",
                        "Switching Pathfinder to **Find Shelter/s** (evacuation / OSM).",
                    )

        if action:
            if action.get("tool") == "list_evacuation_shelters":
                action["requires_frontend"] = True
                action["action"] = "list_evacuation_shelters"
                action.setdefault(
                    "text",
                    "Here are the shelters currently loaded in Pathfinder:",
                )
                state["messages"].append(AIMessage(content=json.dumps(action)))
                return state

            if action.get("tool") == "select_evacuation_destination":
                place = (
                    action.get("place_name")
                    or action.get("destination")
                    or action.get("shelter_name")
                    or ""
                ).strip()
                mode = str(action.get("mode") or "all").strip().lower()
                if mode not in ("all", "driving", "walking", "cycling", "motorcycle"):
                    mode = "all"
                if not place:
                    state["messages"].append(
                        AIMessage(
                            content=json.dumps(
                                {
                                    "text": "Which shelter or school should I select? Use the name from your Pathfinder list (you can paste it from the chat).",
                                    "requires_frontend": False,
                                }
                            )
                        )
                    )
                    return state
                action["place_name"] = place
                action["mode"] = mode
                action["requires_frontend"] = True
                action["action"] = "select_evacuation_destination"
                action.setdefault(
                    "text",
                    f"Selecting **{place}** as the evacuation destination and refreshing routes.",
                )
                state["messages"].append(AIMessage(content=json.dumps(action)))
                return state

            # Check if it's a route finding request
            if action.get("tool") == "find_route":
                start = action.get("start", "")
                destination = action.get("destination", "")
                ptab = (action.get("pathfinder_tab") or action.get("pathfinder_mode") or "").strip().lower()
                if ptab in ("shelter", "shelters"):
                    ptab = "evacuation"
                is_evac = ptab == "evacuation" or action.get("evacuation_mode") is True

                if is_evac:
                    sl = (start or "").strip().lower()
                    if not start or sl in ("current location", "here", "my location"):
                        return self._request_locations(
                            state,
                            "",
                            "",
                            action.get("mode", "all"),
                            pathfinder_tab="evacuation",
                        )
                    action["pathfinder_tab"] = "evacuation"
                    action["destination"] = ""
                    action["requires_frontend"] = True
                    action["action"] = self._get_action_type("find_route")
                    radius = action.get("radius") or action.get("radius_km")
                    if radius not in (None, ""):
                        radius_note = f" (within **{radius} km**)" if not str(radius).lower().endswith("km") else f" (within **{radius}**)"
                    else:
                        radius_note = ""
                    action["text"] = action.get("text") or (
                        f"Opening shelter / evacuation mode near **{start}**{radius_note} — "
                        "nearby schools and shelters load from OpenStreetMap."
                    )
                    state["messages"].append(AIMessage(content=json.dumps(action)))
                    return state

                # Check if the message is ambiguous (e.g., "find routes between X and Y" without clear direction)
                # Look for patterns like "between 2 locations", "between X and Y", etc.
                msg_lower = last_message.lower()
                is_ambiguous = any(p in msg_lower for p in _BETWEEN_AMBIGUOUS_SUBSTRINGS)
                if (
                    not is_ambiguous
                    and start
                    and destination
                    and re.search(r"(?i)\bbetween\s+.+\s+and\s+", last_message)
                ):
                    is_ambiguous = True
                
                # If it's ambiguous and both locations are provided, ask for clarification
                if is_ambiguous and start and destination:
                    return self._request_direction_clarification(state, start, destination, action.get("mode", "all"))
                
                # If locations are missing or generic, ask for them
                if not start or not destination or start == "current location" or destination == "current location":
                    return self._request_locations(state, start, destination, action.get("mode", "all"))
            
            # Add response text
            action["text"] = self._generate_response_text(action)
            state["messages"].append(AIMessage(content=json.dumps(action)))
        else:
            state["messages"].append(AIMessage(
                content=json.dumps({
                    "text": "I can help you find routes between locations. Just tell me where you want to go!"
                })
            ))
        
        return state
    
    def _request_direction_clarification(self, state: AgentState, location1: str, location2: str, mode: str) -> AgentState:
        """Ask user to clarify which location is start and which is destination"""
        
        response = {
            "type": "clarification",
            "question": f"I found two locations:\n\n1. **{location1}**\n2. **{location2}**\n\nWhich one would you like to start from?",
            "options": [location1, location2],
            "suggested_action": {
                "tool": "find_route",
                "awaiting": "direction_choice",
                "location1": location1,
                "location2": location2,
                "mode": mode
            }
        }
        
        # Mark that we're awaiting direction clarification
        state["pending_action"] = {
            "type": "pathfinder_awaiting_locations",
            **response["suggested_action"]
        }
        state["clarification_needed"] = True
        
        state["messages"].append(AIMessage(content=json.dumps(response)))
        return state
    
    def _request_locations(
        self,
        state: AgentState,
        start: str,
        destination: str,
        mode: str,
        pathfinder_tab: Optional[str] = None,
    ) -> AgentState:
        """Request missing location information from user"""
        sa_extra: dict[str, Any] = {}
        if pathfinder_tab:
            sa_extra["pathfinder_tab"] = pathfinder_tab

        if not start or start == "current location":
            q = "I'd be happy to help you find routes! Where would you like to start from?"
            if pathfinder_tab == "evacuation":
                q = "Where should I search for **nearby shelters and schools**? (Your starting area / location)"
            response = {
                "type": "clarification",
                "question": q,
                "options": ["Enter your starting location"],
                "suggested_action": {
                    "tool": "find_route",
                    "awaiting": "start",
                    "destination": destination if destination and destination != "current location" else "",
                    "mode": mode,
                    **sa_extra,
                },
            }
        elif not destination or destination == "current location":
            response = {
                "type": "clarification",
                "question": f"Great! Starting from **{start}**. Where would you like to go?",
                "options": ["Enter your destination"],
                "suggested_action": {
                    "tool": "find_route",
                    "start": start,
                    "awaiting": "destination",
                    "mode": mode,
                    **sa_extra,
                },
            }
        else:
            return state

        state["pending_action"] = {
            "type": "pathfinder_awaiting_locations",
            **response["suggested_action"],
        }
        state["clarification_needed"] = True
        
        state["messages"].append(AIMessage(content=json.dumps(response)))
        return state
    
    def _handle_location_response(self, state: AgentState, user_input: str, pending: dict) -> AgentState:
        """Handle user's response with location information"""
        user_raw = (user_input or "").strip()
        user_input_lower = user_raw.lower()

        if user_input_lower in _CANCEL_PHRASES or user_input_lower.startswith(
            ("cancel ", "stop route", "abort route", "forget route")
        ):
            state["pending_action"] = None
            state["clarification_needed"] = False
            state["messages"].append(
                AIMessage(
                    content=json.dumps(
                        {
                            "text": "Okay — I cancelled the route request. Say when you want directions between two places.",
                            "requires_frontend": False,
                        }
                    )
                )
            )
            return state

        awaiting = pending.get("awaiting")

        # Swap start/end when both are already known (e.g. user said destinations in wrong order)
        if awaiting in ("start", "destination"):
            if any(s in user_input_lower for s in _SWAP_PHRASES):
                ps, pd = (pending.get("start") or "").strip(), (pending.get("destination") or "").strip()
                if ps and pd:
                    mode = pending.get("mode", "all")
                    action = {
                        "tool": "find_route",
                        "start": pd,
                        "destination": ps,
                        "mode": mode,
                        "requires_frontend": True,
                        "action": "calculate_routes",
                        "text": f"Swapped — finding routes from {pd} to {ps} using {self._get_mode_text(mode)}.",
                    }
                    state["pending_action"] = None
                    state["clarification_needed"] = False
                    state["messages"].append(AIMessage(content=json.dumps(action)))
                    return state

        # Handle direction choice (when user picks which location is start)
        if awaiting == "direction_choice":
            location1 = pending.get("location1", "")
            location2 = pending.get("location2", "")
            mode = pending.get("mode", "all")
            location1_lower = location1.lower()
            location2_lower = location2.lower()
            
            # Check if user input matches location1 or location2 (or contains it)
            if location1_lower in user_input_lower or user_input_lower in location1_lower:
                # User chose location1 as start
                start = location1
                destination = location2
            elif location2_lower in user_input_lower or user_input_lower in location2_lower:
                # User chose location2 as start
                start = location2
                destination = location1
            else:
                # User might have said "1" or "2" or "first" or "second"
                tnorm = user_input_lower.strip()
                pick_first = (
                    tnorm in ("1", "first", "one")
                    or re.search(r"\b(option\s*1|#1|the\s+first|first\s+one)\b", user_input_lower)
                    or (location1_lower in user_input_lower and location2_lower not in user_input_lower)
                )
                pick_second = (
                    tnorm in ("2", "second", "two")
                    or re.search(r"\b(option\s*2|#2|the\s+second|second\s+one)\b", user_input_lower)
                    or (location2_lower in user_input_lower and location1_lower not in user_input_lower)
                )
                if pick_first and not pick_second:
                    start = location1
                    destination = location2
                elif pick_second and not pick_first:
                    start = location2
                    destination = location1
                else:
                    # Unclear response, ask again
                    response = {
                        "type": "clarification",
                        "question": f"I'm not sure which location you meant. Please choose:\n\n1. **{location1}**\n2. **{location2}**\n\nWhich one is your starting point?",
                        "options": [location1, location2],
                        "suggested_action": {
                            "tool": "find_route",
                            "awaiting": "direction_choice",
                            "location1": location1,
                            "location2": location2,
                            "mode": mode
                        }
                    }
                    state["messages"].append(AIMessage(content=json.dumps(response)))
                    return state
            
            # Now we have both start and destination
            action = {
                "tool": "find_route",
                "start": start,
                "destination": destination,
                "mode": mode,
                "requires_frontend": True,
                "action": "calculate_routes",
                "text": f"Perfect! Finding routes from {start} to {destination} using {self._get_mode_text(mode)}. I'll show you the available options with traffic information."
            }
            state["pending_action"] = None
            state["clarification_needed"] = False
            state["messages"].append(AIMessage(content=json.dumps(action)))
            return state
        
        # Extract just the location name from the user's response
        location = user_raw

        prefixes_to_remove = [
            "here's the starting location:",
            "the starting location is:",
            "starting location:",
            "start location:",
            "start:",
            "from:",
            "here's the destination:",
            "the destination should be:",
            "the destination is:",
            "destination:",
            "to:",
            "it's",
            "it is",
        ]

        location_lower = location.lower()
        for prefix in prefixes_to_remove:
            if location_lower.startswith(prefix):
                location = location[len(prefix) :].strip()
                location_lower = location.lower()
                break

        if awaiting in ("start", "destination") and len(location.strip()) < 2:
            mode = pending.get("mode", "all")
            if awaiting == "start":
                sa = {
                    "tool": "find_route",
                    "awaiting": "start",
                    "destination": pending.get("destination", ""),
                    "mode": mode,
                }
            else:
                sa = {
                    "tool": "find_route",
                    "awaiting": "destination",
                    "start": pending.get("start", ""),
                    "mode": mode,
                }
            response = {
                "type": "clarification",
                "question": "I need a clearer place name (city, neighborhood, or landmark). Could you type it again?",
                "options": ["Try again"],
                "suggested_action": sa,
            }
            state["pending_action"] = {
                "type": "pathfinder_awaiting_locations",
                **sa,
            }
            state["messages"].append(AIMessage(content=json.dumps(response)))
            return state

        if awaiting == "start":
            # User provided start location
            start = location
            destination = pending.get("destination", "")
            mode = pending.get("mode", "all")

            if pending.get("pathfinder_tab") == "evacuation":
                action = {
                    "tool": "find_route",
                    "start": start,
                    "destination": "",
                    "pathfinder_tab": "evacuation",
                    "mode": mode,
                    "requires_frontend": True,
                    "action": "calculate_routes",
                    "text": f"Opening shelter / evacuation mode near **{start}** — nearby schools and shelters load from OpenStreetMap.",
                }
                state["pending_action"] = None
                state["clarification_needed"] = False
                state["messages"].append(AIMessage(content=json.dumps(action)))
                return state

            if not destination:
                # Still need destination
                response = {
                    "type": "clarification",
                    "question": f"Perfect! Starting from **{start}**. Where would you like to go?",
                    "options": ["Enter your destination"],
                    "suggested_action": {
                        "tool": "find_route",
                        "start": start,
                        "awaiting": "destination",
                        "mode": mode,
                    },
                }
                state["pending_action"] = {
                    "type": "pathfinder_awaiting_locations",
                    **response["suggested_action"],
                }
                state["messages"].append(AIMessage(content=json.dumps(response)))
            else:
                # Have both locations now
                action = {
                    "tool": "find_route",
                    "start": start,
                    "destination": destination,
                    "mode": mode,
                    "requires_frontend": True,
                    "action": "calculate_routes",
                    "text": f"Finding routes from {start} to {destination} using {self._get_mode_text(mode)}. I'll show you the available options with traffic information."
                }
                state["pending_action"] = None
                state["clarification_needed"] = False
                state["messages"].append(AIMessage(content=json.dumps(action)))
        
        elif awaiting == "destination":
            # User provided destination
            destination = location
            start = pending.get("start", "")
            mode = pending.get("mode", "all")
            
            # Have both locations now
            action = {
                "tool": "find_route",
                "start": start,
                "destination": destination,
                "mode": mode,
                "requires_frontend": True,
                "action": "calculate_routes",
                "text": f"Finding routes from {start} to {destination} using {self._get_mode_text(mode)}. I'll show you the available options with traffic information."
            }
            state["pending_action"] = None
            state["clarification_needed"] = False
            state["messages"].append(AIMessage(content=json.dumps(action)))
        
        return state
    
    def _get_mode_text(self, mode: str) -> str:
        """Get human-readable mode text"""
        return {
            "all": "all available modes",
            "driving": "driving",
            "walking": "walking",
            "cycling": "cycling",
            "motorcycle": "motorcycle"
        }.get(mode, mode)
    
    def _parse_intent(self, message: str) -> Optional[dict]:
        """Parse pathfinder intents using LLM"""
        
        msg_lower = message.lower()
        
        system_prompt = """You are a pathfinder intent parser. Extract the user's routing intent from their message.

Available pathfinder actions:
1. **find_route** - Point-to-point routes OR shelter/evacuation search
   - Extract: start location, destination location, mode (optional)
   - For **shelters / schools / evacuation** (nearby safe places, OSM): set **pathfinder_tab** to **"evacuation"**, provide **start** (search origin), leave **destination** as **""** (empty). Do NOT copy explanatory phrases like "the starting point which is…" into destination — the UI loads shelters and can route start→nearest shelter automatically.
   - When the user specifies a search distance (e.g. "within 2 km", "5km radius"), include **radius** as a number of kilometers (e.g. `"radius": 2` or `"radius": "2km"`). UI slider range is 0.5–30 km; omit radius if not mentioned (UI keeps its current/default radius).
   - Follow-ups that only change distance after a shelter search (e.g. "now within 2 km", "make it 5km") → same **find_route** with **pathfinder_tab** "evacuation", reuse the previous **start**, set the new **radius**, leave **destination** "".
   - Modes: all, driving, walking, cycling, motorcycle
   - Mode synonyms: car/driving, bicycle/bike/cycling, pedestrian/walking, motorcycle/motorbike
   - If locations are not provided or unclear, return empty strings for start/destination
   - Examples: "find route from SF to LA" → {"tool": "find_route", "start": "San Francisco", "destination": "Los Angeles", "mode": "all"}
   - Examples: "find shelters near Makati" → {"tool": "find_route", "start": "Makati", "destination": "", "pathfinder_tab": "evacuation", "mode": "all"}
   - Examples: "show shelters near Gateway Mall Cubao with a 2 km radius" → {"tool": "find_route", "start": "Gateway Mall Cubao", "destination": "", "pathfinder_tab": "evacuation", "mode": "all", "radius": 2}
   - Examples: "now within 2 km" (after a shelter search near Gateway Mall) → {"tool": "find_route", "start": "Gateway Mall Cubao", "destination": "", "pathfinder_tab": "evacuation", "mode": "all", "radius": 2}
   - Examples: "evacuation routes from here" → {"tool": "find_route", "start": "current location", "destination": "", "pathfinder_tab": "evacuation", "mode": "all"}
   - Examples: "drive to the airport" → {"tool": "find_route", "start": "current location", "destination": "airport", "mode": "driving"}
   - Examples: "walking directions to the park" → {"tool": "find_route", "start": "current location", "destination": "park", "mode": "walking"}
   - Examples: "route optimization" → {"tool": "find_route", "start": "", "destination": "", "mode": "all"}
   - Examples: "perform route optimization between 2 locations" → {"tool": "find_route", "start": "", "destination": "", "mode": "all"}
   - Examples: "optimize route" → {"tool": "find_route", "start": "", "destination": "", "mode": "all"}

2. **change_route_mode** - Change transportation mode
   - Modes: all, driving, walking, cycling, motorcycle
   - Mode synonyms: 
     * car/driving → "driving"
     * bicycle/bike/cycling → "cycling"
     * pedestrian/walking/walk → "walking"
     * motorcycle/motorbike → "motorcycle"
     * all/show all → "all"
   - Examples: "switch to cycling" → {"tool": "change_route_mode", "mode": "cycling"}
   - Examples: "change to car" → {"tool": "change_route_mode", "mode": "driving"}
   - Examples: "switch mode to bicycle" → {"tool": "change_route_mode", "mode": "cycling"}
   - Examples: "show pedestrian routes" → {"tool": "change_route_mode", "mode": "walking"}
   - Examples: "switch to motorcycle" → {"tool": "change_route_mode", "mode": "motorcycle"}
   - Examples: "show all modes" → {"tool": "change_route_mode", "mode": "all"}
   - Examples: "change travel mode to driving" → {"tool": "change_route_mode", "mode": "driving"}
   - Examples: "set transportation mode to walking" → {"tool": "change_route_mode", "mode": "walking"}

3. **change_route_sort** - Change route sorting
   - Options: fastest, safest, best_balance
   - Examples: "show safest routes" → {"tool": "change_route_sort", "sort_by": "safest"}
   - Examples: "sort by fastest" → {"tool": "change_route_sort", "sort_by": "fastest"}
   - Examples: "sort routes by safest" → {"tool": "change_route_sort", "sort_by": "safest"}
   - Examples: "sort the routes by fastest" → {"tool": "change_route_sort", "sort_by": "fastest"}
   - Examples: "best balance" → {"tool": "change_route_sort", "sort_by": "best_balance"}
   - Examples: "show me the fastest route" → {"tool": "change_route_sort", "sort_by": "fastest"}
   - Examples: "which is the safest" → {"tool": "change_route_sort", "sort_by": "safest"}

3b. **set_pathfinder_tab** - Switch Pathfinder **sub-tab**: **destination** = "Set Destination" (point-to-point, two addresses); **evacuation** = "Find Shelter/s" (OSM shelters/schools). Use when the user mentions switching Pathfinder **mode/tab**, not when they name a specific shelter to select.
   - Examples: "switch pathfinder to set destination mode" → {"tool": "set_pathfinder_tab", "pathfinder_tab": "destination"}
   - Examples: "pathfinder point to point" (with switch/change) → {"tool": "set_pathfinder_tab", "pathfinder_tab": "destination"}
   - Examples: "use find shelter mode in pathfinder" → {"tool": "set_pathfinder_tab", "pathfinder_tab": "evacuation"}
   - Do NOT use **open_pathfinder** alone for these; return **set_pathfinder_tab** when they specify which tab.

4. **open_pathfinder** - Open pathfinder panel
   - Examples: "open pathfinder" → {"tool": "open_pathfinder"}

5. **close_pathfinder** - Hide/collapse the pathfinder **sidebar/panel only**. Do NOT use when the user wants to erase route lines from the map.
   - Examples: "close pathfinder" → {"tool": "close_pathfinder"}
   - Examples: "hide the routing panel" → {"tool": "close_pathfinder"}

5b. **clear_pathfinder_routes** - Remove drawn routes from the **map** and reset Pathfinder trip inputs (same as in-app **Clear Routes**). Use when they mention clearing/removing/wiping **routes** on or from the **map**, or **displayed** routes.
   - Do NOT confuse with **close_pathfinder** (panel visibility).
   - Examples: "clear all routes on the map" → {"tool": "clear_pathfinder_routes"}
   - Examples: "remove routes displayed on the map" → {"tool": "clear_pathfinder_routes"}
   - Examples: "wipe pathfinder routes from the map" → {"tool": "clear_pathfinder_routes"}

6. **list_evacuation_shelters** - User wants a **text list** of shelters/schools **already loaded** in the Pathfinder evacuation dropdown (not a new search). Use when they say list/enumerate/name all shelters for the current start, Pathfinder, or dropdown.
   - Examples: "list all nearest shelters" → {"tool": "list_evacuation_shelters"}
   - Examples: "what shelters are shown for my start point" → {"tool": "list_evacuation_shelters"}
   - Do NOT use for "find shelters near X" (that is **find_route** with pathfinder_tab evacuation).

7. **select_evacuation_destination** - User wants to **change the selected shelter/school** in evacuation mode or set the evacuation **destination** to a name from the loaded OSM list (must drive the Pathfinder UI).
   - Extract **place_name** (full name as the user wrote it).
   - Examples: "change destination to Open Hand School" → {"tool": "select_evacuation_destination", "place_name": "Open Hand School", "mode": "all"}
   - Examples: "select Molave Youth Home as shelter" → {"tool": "select_evacuation_destination", "place_name": "Molave Youth Home", "mode": "all"}
   - Do NOT use for point-to-point "route from A to B" with two addresses (use **find_route** without pathfinder_tab evacuation).

IMPORTANT: Always normalize mode synonyms to the standard modes:
- car/driving → "driving"
- bicycle/bike/cycling → "cycling"
- pedestrian/walking/walk → "walking"
- motorcycle/motorbike → "motorcycle"
- all/show all → "all"

Edge cases:
- Multi-word cities and countries: extract full place names (e.g. "Quezon City", "Metro Manila", "Los Angeles, CA").
- "from A to B via C": use A as start, C or final segment as destination if clear; otherwise start=A, destination=last major place.
- "I'm at / we're at / currently at [place]": treat as start; "going to / headed to [place]" as destination.
- Typos: infer obvious place names when possible.
- "Round trip" / "return route": still return find_route with start and destination as given; UI may only show one leg.
- If user only asks to open routing UI with no places: open_pathfinder OR find_route with empty start/destination.
- "Show fastest" after no route context → change_route_sort fastest; "fastest route from X to Y" → find_route (not sort only).

Respond with JSON object for the action. Return ONLY the JSON, no explanation.

Examples:
- "find route from San Francisco to Los Angeles" → {"tool": "find_route", "start": "San Francisco", "destination": "Los Angeles", "mode": "all"}
- "how do I get to the airport" → {"tool": "find_route", "start": "current location", "destination": "airport", "mode": "all"}
- "drive to downtown" → {"tool": "find_route", "start": "current location", "destination": "downtown", "mode": "driving"}
- "walking directions to Central Park" → {"tool": "find_route", "start": "current location", "destination": "Central Park", "mode": "walking"}
- "show me cycling routes" → {"tool": "change_route_mode", "mode": "cycling"}
- "switch to car" → {"tool": "change_route_mode", "mode": "driving"}
- "change mode to bicycle" → {"tool": "change_route_mode", "mode": "cycling"}
- "show pedestrian routes" → {"tool": "change_route_mode", "mode": "walking"}
- "switch to motorcycle" → {"tool": "change_route_mode", "mode": "motorcycle"}
- "show all modes" → {"tool": "change_route_mode", "mode": "all"}
- "now change the travel mode to driving" → {"tool": "change_route_mode", "mode": "driving"}
- "clear all routes displayed on the map" → {"tool": "clear_pathfinder_routes"}
- "switch pathfinder to set destination mode" → {"tool": "set_pathfinder_tab", "pathfinder_tab": "destination"}
- "what's the safest way" → {"tool": "change_route_sort", "sort_by": "safest"}
"""

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
            parsed_action = json.loads(response)
            
            # Add requires_frontend flag
            parsed_action["requires_frontend"] = True
            parsed_action["action"] = self._get_action_type(parsed_action.get("tool", ""))
            
            return parsed_action
            
        except Exception as e:
            print(f"Error parsing intent with LLM: {e}")
            if _wants_select_evac_destination_control(msg_lower):
                hs = _heuristic_select_evacuation_destination(message)
                if hs:
                    hs["requires_frontend"] = True
                    hs["action"] = self._get_action_type(
                        "select_evacuation_destination"
                    )
                    return hs
            if _wants_shelter_list(msg_lower):
                return {
                    "tool": "list_evacuation_shelters",
                    "requires_frontend": True,
                    "action": "list_evacuation_shelters",
                }
            if _wants_pathfinder_clear_routes(msg_lower):
                return {
                    "tool": "clear_pathfinder_routes",
                    "requires_frontend": True,
                    "action": self._get_action_type("clear_pathfinder_routes"),
                    "text": "Clearing routes from the map and resetting Pathfinder inputs.",
                }
            if _wants_pathfinder_tab_switch(msg_lower):
                ht = _heuristic_set_pathfinder_tab(message)
                if ht:
                    ht["requires_frontend"] = True
                    ht["action"] = self._get_action_type("set_pathfinder_tab")
                    if ht.get("pathfinder_tab") == "destination":
                        ht.setdefault(
                            "text",
                            "Switching Pathfinder to **Set Destination** (point-to-point).",
                        )
                    else:
                        ht.setdefault(
                            "text",
                            "Switching Pathfinder to **Find Shelter/s**.",
                        )
                    return ht
            hm = _heuristic_change_route_mode(message)
            if hm:
                hm["requires_frontend"] = True
                hm["action"] = self._get_action_type("change_route_mode")
                return hm
            h = self._heuristic_find_route(message)
            if h:
                h["requires_frontend"] = True
                h["action"] = self._get_action_type(h.get("tool", ""))
                return h
            if any(
                word in msg_lower
                for word in [
                    "route",
                    "directions",
                    "get to",
                    "way to",
                    "drive to",
                    "walk to",
                    "navigate",
                    "pathfinder",
                ]
            ):
                return {
                    "tool": "open_pathfinder",
                    "requires_frontend": True,
                    "action": "show_pathfinder",
                }
            return None
    
    def _get_action_type(self, tool: str) -> str:
        """Map tool name to action type"""
        action_map = {
            "find_route": "calculate_routes",
            "change_route_mode": "update_route_mode",
            "change_route_sort": "update_route_sort",
            "open_pathfinder": "show_pathfinder",
            "close_pathfinder": "hide_pathfinder",
            "set_pathfinder_tab": "set_pathfinder_tab",
            "clear_pathfinder_routes": "clear_pathfinder_routes",
            "list_evacuation_shelters": "list_evacuation_shelters",
            "select_evacuation_destination": "select_evacuation_destination",
        }
        return action_map.get(tool, "")
    
    def _generate_response_text(self, action: dict) -> str:
        """Generate human-readable response text based on action"""
        
        tool_name = action.get("tool", "")
        
        if tool_name == "find_route":
            start = action.get("start", "")
            destination = action.get("destination", "")
            mode = action.get("mode", "all")
            
            mode_text = {
                "all": "all available modes",
                "driving": "driving",
                "walking": "walking",
                "cycling": "cycling",
                "motorcycle": "motorcycle"
            }.get(mode, mode)
            
            return f"Finding routes from {start} to {destination} using {mode_text}. I'll show you the available options with traffic information."
        
        elif tool_name == "change_route_mode":
            mode = action.get("mode", "")
            mode_text = {
                "all": "all modes",
                "driving": "driving (car)",
                "walking": "walking (pedestrian)",
                "cycling": "cycling (bicycle)",
                "motorcycle": "motorcycle"
            }.get(mode, mode)
            
            return f"Switching to {mode_text}. If you haven't found routes yet, please ask me to find routes between two locations first (e.g., 'Find routes from A to B')."
        
        elif tool_name == "change_route_sort":
            sort_by = action.get("sort_by", "")
            sort_text = {
                "fastest": "fastest routes",
                "safest": "safest routes (least traffic incidents)",
                "best_balance": "best balance (time + safety)"
            }.get(sort_by, sort_by)
            
            return f"Sorting routes by {sort_text}. If you haven't found routes yet, please ask me to find routes between two locations first (e.g., 'Find routes from A to B')."
        
        elif tool_name == "open_pathfinder":
            return "Opening the pathfinder panel. You can now plan your route."
        
        elif tool_name == "close_pathfinder":
            return "Closing the pathfinder panel."

        elif tool_name == "set_pathfinder_tab":
            tab = (action.get("pathfinder_tab") or "").strip().lower()
            if tab == "destination":
                return "Switching Pathfinder to **Set Destination** (point-to-point)."
            if tab == "evacuation":
                return "Switching Pathfinder to **Find Shelter/s** (evacuation / OSM)."
            return "Updating the Pathfinder tab."

        elif tool_name == "clear_pathfinder_routes":
            return "Clearing routes from the map and resetting Pathfinder inputs."

        elif tool_name == "select_evacuation_destination":
            place = action.get("place_name") or action.get("destination") or ""
            return f"Selecting **{place}** as the evacuation destination and refreshing routes."

        return "Action completed."
