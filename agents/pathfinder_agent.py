"""Pathfinder agent for route planning and navigation"""
import json
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
        
        # Parse the intent from the message
        action = self._parse_intent(last_message)
        print(f"Parsed action: {action}")
        
        if action:
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
    
    def _parse_intent(self, message: str) -> dict | None:
        """Parse pathfinder intents using LLM"""
        
        msg_lower = message.lower()
        
        system_prompt = """You are a pathfinder intent parser. Extract the user's routing intent from their message.

Available pathfinder actions:
1. **find_route** - Find routes between two locations
   - Extract: start location, destination location, mode (optional)
   - Modes: all, driving, walking, cycling, motorcycle
   - Mode synonyms: car/driving, bicycle/bike/cycling, pedestrian/walking, motorcycle/motorbike
   - Examples: "find route from SF to LA" → {"tool": "find_route", "start": "San Francisco", "destination": "Los Angeles", "mode": "all"}
   - Examples: "drive to the airport" → {"tool": "find_route", "start": "current location", "destination": "airport", "mode": "driving"}
   - Examples: "walking directions to the park" → {"tool": "find_route", "start": "current location", "destination": "park", "mode": "walking"}

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
