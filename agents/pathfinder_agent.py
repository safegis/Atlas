"""Pathfinder agent for route planning and navigation"""
import json
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from tools.pathfinder_tools import PATHFINDER_TOOLS
from state import AgentState


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
    
    def process(self, state: AgentState) -> AgentState:
        """Process pathfinder requests"""
        messages = state["messages"]
        last_message = messages[-1].content
        
        print(f"Pathfinder Agent processing: {last_message}")
        
        # Check if we have a pending pathfinder action waiting for locations
        pending_action = state.get("pending_action")
        print(f"Pending action from state: {pending_action}")
        
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
        
        # Parse the intent from the message
        action = self._parse_intent(last_message)
        print(f"Parsed action: {action}")
        
        if action:
            # Check if it's a route finding request
            if action.get("tool") == "find_route":
                start = action.get("start", "")
                destination = action.get("destination", "")
                
                # Check if the message is ambiguous (e.g., "find routes between X and Y" without clear direction)
                # Look for patterns like "between 2 locations", "between X and Y", etc.
                msg_lower = last_message.lower()
                is_ambiguous = any(pattern in msg_lower for pattern in [
                    "between 2 locations",
                    "between two locations",
                    "between these locations",
                    "between the following",
                ])
                
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
    
    def _request_locations(self, state: AgentState, start: str, destination: str, mode: str) -> AgentState:
        """Request missing location information from user"""
        
        if not start or start == "current location":
            # Ask for start location
            response = {
                "type": "clarification",
                "question": "I'd be happy to help you find routes! Where would you like to start from?",
                "options": ["Enter your starting location"],
                "suggested_action": {
                    "tool": "find_route",
                    "awaiting": "start",
                    "destination": destination if destination and destination != "current location" else "",
                    "mode": mode
                }
            }
        elif not destination or destination == "current location":
            # Ask for destination
            response = {
                "type": "clarification",
                "question": f"Great! Starting from **{start}**. Where would you like to go?",
                "options": ["Enter your destination"],
                "suggested_action": {
                    "tool": "find_route",
                    "start": start,
                    "awaiting": "destination",
                    "mode": mode
                }
            }
        else:
            # Both provided, shouldn't reach here
            return state
        
        # Mark that we're awaiting location info
        state["pending_action"] = {
            "type": "pathfinder_awaiting_locations",
            **response["suggested_action"]
        }
        state["clarification_needed"] = True
        
        state["messages"].append(AIMessage(content=json.dumps(response)))
        return state
    
    def _handle_location_response(self, state: AgentState, user_input: str, pending: dict) -> AgentState:
        """Handle user's response with location information"""
        
        awaiting = pending.get("awaiting")
        
        # Handle direction choice (when user picks which location is start)
        if awaiting == "direction_choice":
            location1 = pending.get("location1", "")
            location2 = pending.get("location2", "")
            mode = pending.get("mode", "all")
            
            # Check which location the user chose
            user_input_lower = user_input.lower().strip()
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
                if any(word in user_input_lower for word in ["1", "first", "one"]):
                    start = location1
                    destination = location2
                elif any(word in user_input_lower for word in ["2", "second", "two"]):
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
        # Remove common prefixes like "Here's the starting location:", "The destination should be:", etc.
        location = user_input.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "here's the starting location:",
            "the starting location is:",
            "starting location:",
            "start location:",
            "here's the destination:",
            "the destination should be:",
            "the destination is:",
            "destination:",
            "it's",
            "it is"
        ]
        
        location_lower = location.lower()
        for prefix in prefixes_to_remove:
            if location_lower.startswith(prefix):
                location = location[len(prefix):].strip()
                break
        
        if awaiting == "start":
            # User provided start location
            start = location
            destination = pending.get("destination", "")
            mode = pending.get("mode", "all")
            
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
                        "mode": mode
                    }
                }
                state["pending_action"] = {
                    "type": "pathfinder_awaiting_locations",
                    **response["suggested_action"]
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
1. **find_route** - Find routes between two locations
   - Extract: start location, destination location, mode (optional)
   - Modes: all, driving, walking, cycling, motorcycle
   - Mode synonyms: car/driving, bicycle/bike/cycling, pedestrian/walking, motorcycle/motorbike
   - If locations are not provided or unclear, return empty strings for start/destination
   - Examples: "find route from SF to LA" → {"tool": "find_route", "start": "San Francisco", "destination": "Los Angeles", "mode": "all"}
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

3. **change_route_sort** - Change route sorting
   - Options: fastest, safest, best_balance
   - Examples: "show safest routes" → {"tool": "change_route_sort", "sort_by": "safest"}
   - Examples: "sort by fastest" → {"tool": "change_route_sort", "sort_by": "fastest"}
   - Examples: "sort routes by safest" → {"tool": "change_route_sort", "sort_by": "safest"}
   - Examples: "sort the routes by fastest" → {"tool": "change_route_sort", "sort_by": "fastest"}
   - Examples: "best balance" → {"tool": "change_route_sort", "sort_by": "best_balance"}
   - Examples: "show me the fastest route" → {"tool": "change_route_sort", "sort_by": "fastest"}
   - Examples: "which is the safest" → {"tool": "change_route_sort", "sort_by": "safest"}

4. **open_pathfinder** - Open pathfinder panel
   - Examples: "open pathfinder" → {"tool": "open_pathfinder"}

5. **close_pathfinder** - Close pathfinder panel
   - Examples: "close pathfinder" → {"tool": "close_pathfinder"}

IMPORTANT: Always normalize mode synonyms to the standard modes:
- car/driving → "driving"
- bicycle/bike/cycling → "cycling"
- pedestrian/walking/walk → "walking"
- motorcycle/motorbike → "motorcycle"
- all/show all → "all"

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
            # Fallback to simple keyword matching
            if any(word in msg_lower for word in ["route", "directions", "get to", "way to", "drive to", "walk to"]):
                # Try to extract locations
                return {"tool": "open_pathfinder", "requires_frontend": True, "action": "show_pathfinder"}
            return None
    
    def _get_action_type(self, tool: str) -> str:
        """Map tool name to action type"""
        action_map = {
            "find_route": "calculate_routes",
            "change_route_mode": "update_route_mode",
            "change_route_sort": "update_route_sort",
            "open_pathfinder": "show_pathfinder",
            "close_pathfinder": "hide_pathfinder"
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
        
        return "Action completed."
