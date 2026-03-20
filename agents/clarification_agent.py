"""Clarification agent for handling ambiguous requests"""
import json
from langchain_core.messages import AIMessage
from state import AgentState


class ClarificationAgent:
    """Handles clarification flows and follow-up responses"""
    
    def process(self, state: AgentState) -> AgentState:
        """Process clarification responses"""
        messages = state["messages"]
        last_message = messages[-1].content.lower().strip()
        # "Show both" / "Display PHIVOLCS" etc. — strip common prefixes for option matching
        option_norm = last_message
        for prefix in (
            "show ",
            "display ",
            "i want ",
            "pick ",
            "select ",
            "choose ",
            "use ",
        ):
            if option_norm.startswith(prefix):
                option_norm = option_norm[len(prefix) :].strip()
        pending_action = state.get("pending_action")
        
        print(f"Clarification Agent processing: {last_message}")
        print(f"Pending action from state: {pending_action}")
        
        if not pending_action:
            # Try to find pending action from previous messages
            print(f"No pending action in state, searching {len(messages)} messages")
            for i, msg in enumerate(reversed(messages[:-1])):
                content = None
                if hasattr(msg, 'content'):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get('content')
                
                if content:
                    try:
                        parsed = json.loads(content)
                        print(f"Message {i}: type={parsed.get('type')}, has suggested_action={('suggested_action' in parsed)}")
                        if parsed.get("type") == "clarification" and "suggested_action" in parsed:
                            pending_action = parsed
                            print(f"Found pending action in message: {parsed.get('suggested_action')}")
                            break
                    except:
                        pass
        
        if not pending_action:
            state["clarification_needed"] = False
            state["messages"].append(AIMessage(content=json.dumps({
                "text": "I'm not sure what you're confirming. Could you please rephrase your request?"
            })))
            return state
        
        # Handle time of day clarifications (yes/no or option selection)
        suggested = pending_action.get("suggested_action", {})
        if suggested.get("multiple_actions"):
            # Check if this is a time of day clarification
            has_time_action = any(action.get("tool") == "control_time_of_day" for action in suggested.get("multiple_actions", []))
            
            if has_time_action:
                # Yes responses (option 1 or affirmative)
                if (last_message in ["yes", "yeah", "yep", "sure", "ok", "okay", "1", "option 1", "option one", "default"] or 
                    "yes" in last_message or 
                    "option 1" in last_message or 
                    "option one" in last_message or
                    "default" in last_message or
                    "first" in last_message):
                    # Execute the suggested multiple actions
                    state["messages"].append(AIMessage(content=json.dumps({
                        "multiple_actions": suggested["multiple_actions"],
                        "requires_frontend": True
                    })))
                    state["clarification_needed"] = False
                    state["pending_action"] = None
                    return state
                
                # Option 2: Alternative style (satellite instead of default)
                if (last_message in ["2", "satellite", "option 2", "option two"] or 
                    "satellite" in last_message or 
                    "option 2" in last_message or 
                    "option two" in last_message or
                    "second" in last_message):
                    alternative_style = suggested.get("alternative_style", "satellite")
                    # Replace the style in the first action
                    modified_actions = []
                    for action in suggested["multiple_actions"]:
                        if action.get("tool") == "change_map_style":
                            modified_actions.append({**action, "style": alternative_style})
                        else:
                            modified_actions.append(action)
                    
                    state["messages"].append(AIMessage(content=json.dumps({
                        "multiple_actions": modified_actions,
                        "requires_frontend": True
                    })))
                    state["clarification_needed"] = False
                    state["pending_action"] = None
                    return state
                
                # No/Cancel responses
                if (last_message in ["no", "nope", "nah", "cancel", "3", "option 3", "option three"] or 
                    "cancel" in last_message or 
                    "option 3" in last_message or
                    "option three" in last_message or
                    "nevermind" in last_message or
                    "never mind" in last_message):
                    state["messages"].append(AIMessage(content=json.dumps({
                        "text": "Okay, I've cancelled the time of day change."
                    })))
                    state["clarification_needed"] = False
                    state["pending_action"] = None
                    return state
        
        # Helper function to merge actions with pending map actions
        def merge_with_pending_map(action, pending_action):
            pending_map = pending_action.get("pending_map_action")
            pending_map_actions = pending_action.get("pending_map_actions")
            
            if pending_map_actions:
                return [action] + pending_map_actions
            elif pending_map:
                # Check if pending_map itself has multiple actions
                if pending_map.get("multiple_actions"):
                    return [action] + pending_map["multiple_actions"]
                else:
                    return [action, pending_map]
            return [action]
        
        # Check for earthquake source selection
        # Option 1: Philippines
        if (
            option_norm in ["philippines", "philippine", "phivolcs", "local", "1"]
            or option_norm.startswith("philippines")
            or last_message in ["philippines", "philippine", "phivolcs", "local", "1"]
            or last_message.startswith("philippines")
        ):
            suggested = pending_action.get("suggested_action", {})
            if suggested.get("tool") == "control_earthquake_data":
                action = {"tool": "control_earthquake_data", "action": "enable", "source": "philippine", "requires_frontend": True}
                all_actions = merge_with_pending_map(action, pending_action)
                
                if len(all_actions) > 1:
                    state["messages"].append(AIMessage(content=json.dumps({
                        "multiple_actions": all_actions,
                        "requires_frontend": True
                    })))
                else:
                    state["messages"].append(AIMessage(content=json.dumps(action)))
                
                state["clarification_needed"] = False
                state["pending_action"] = None
                return state
        
        # Option 2: Global
        if (
            option_norm in ["global", "usgs", "worldwide", "world", "2"]
            or option_norm.startswith("global")
            or last_message in ["global", "usgs", "worldwide", "world", "2"]
            or last_message.startswith("global")
        ):
            suggested = pending_action.get("suggested_action", {})
            if suggested.get("tool") == "control_earthquake_data":
                action = {"tool": "control_earthquake_data", "action": "enable", "source": "global", "requires_frontend": True}
                all_actions = merge_with_pending_map(action, pending_action)
                
                if len(all_actions) > 1:
                    state["messages"].append(AIMessage(content=json.dumps({
                        "multiple_actions": all_actions,
                        "requires_frontend": True
                    })))
                else:
                    state["messages"].append(AIMessage(content=json.dumps(action)))
                
                state["clarification_needed"] = False
                state["pending_action"] = None
                return state
        
        # Option 3: Both
        if (
            option_norm in ["both", "all", "3"]
            or option_norm.startswith("both")
            or last_message in ["both", "all", "3"]
            or last_message.startswith("both")
        ):
            suggested = pending_action.get("suggested_action", {})
            if suggested.get("tool") == "control_earthquake_data":
                # Enable both sources
                both_action = {
                    "tool": "control_earthquake_data",
                    "action": "enable",
                    "source": "philippine",
                    "requires_frontend": True,
                    "also_enable": {"tool": "control_earthquake_data", "action": "enable", "source": "global"}
                }
                all_actions = merge_with_pending_map(both_action, pending_action)
                conflict_note = pending_action.get("conflict_note")
                
                if len(all_actions) > 1:
                    response = {
                        "multiple_actions": all_actions,
                        "requires_frontend": True
                    }
                    if conflict_note:
                        response["info_message"] = conflict_note
                    state["messages"].append(AIMessage(content=json.dumps(response)))
                else:
                    response = both_action.copy()
                    if conflict_note:
                        response["info_message"] = conflict_note
                    state["messages"].append(AIMessage(content=json.dumps(response)))
                
                state["clarification_needed"] = False
                state["pending_action"] = None
                return state
        
        # Check for weather scope selection
        suggested = pending_action.get("suggested_action", {})
        if suggested.get("tool") == "control_weather_data":
            # Determine action from user message (enable or disable)
            disable_keywords = ["disable", "hide", "turn off", "deactivate", "remove", "stop"]
            enable_keywords = ["enable", "show", "see", "view", "display", "turn on", "activate", "start"]
            
            # Check if user explicitly wants to disable or enable
            user_action = "enable"  # Default
            if any(word in last_message for word in disable_keywords):
                user_action = "disable"
            elif any(word in last_message for word in enable_keywords):
                user_action = "enable"
            
            # Option 1: Province level (Philippines)
            if last_message in ["province", "provincial", "region", "1"] or "philippines" in last_message:
                action = {"tool": "control_weather_data", "action": user_action, "scope": "province", "requires_frontend": True}
                state["messages"].append(AIMessage(content=json.dumps(action)))
                state["clarification_needed"] = False
                state["pending_action"] = None
                return state
            
            # Option 2: City - Abra
            if "abra" in last_message or last_message == "2":
                action = {"tool": "control_weather_data", "action": user_action, "scope": "city", "province": "Abra", "requires_frontend": True}
                state["messages"].append(AIMessage(content=json.dumps(action)))
                state["clarification_needed"] = False
                state["pending_action"] = None
                return state
            
            # Option 3: City - Agusan del Norte
            if "agusan del norte" in last_message or last_message == "3":
                action = {"tool": "control_weather_data", "action": user_action, "scope": "city", "province": "Agusan del Norte", "requires_frontend": True}
                state["messages"].append(AIMessage(content=json.dumps(action)))
                state["clarification_needed"] = False
                state["pending_action"] = None
                return state
            
            # Option 4: City - Agusan del Sur
            if "agusan del sur" in last_message or last_message == "4":
                action = {"tool": "control_weather_data", "action": user_action, "scope": "city", "province": "Agusan del Sur", "requires_frontend": True}
                state["messages"].append(AIMessage(content=json.dumps(action)))
                state["clarification_needed"] = False
                state["pending_action"] = None
                return state
            
            # Option 5: City - Aklan
            if "aklan" in last_message or last_message == "5":
                action = {"tool": "control_weather_data", "action": user_action, "scope": "city", "province": "Aklan", "requires_frontend": True}
                state["messages"].append(AIMessage(content=json.dumps(action)))
                state["clarification_needed"] = False
                state["pending_action"] = None
                return state
            
            # Option 6: All weather data
            if last_message in ["all", "6"] or last_message.startswith("all"):
                action = {"tool": "control_weather_data", "action": user_action, "scope": "all", "requires_frontend": True}
                state["messages"].append(AIMessage(content=json.dumps(action)))
                state["clarification_needed"] = False
                state["pending_action"] = None
                return state
        
        # Helper to check if hazard action needs clarification and trigger second clarification
        def check_hazard_clarification(hazard_action, map_actions):
            # If hazard action is already a clarification object, use it directly
            if hazard_action and hazard_action.get("question"):
                # It's already a clarification - add pending map actions to it
                return {
                    **hazard_action,
                    "pending_map_actions": map_actions
                }
            # If hazard action exists and has default source, it might need clarification
            elif hazard_action and hazard_action.get("tool") == "control_earthquake_data":
                # Ask for earthquake source clarification as second step
                return {
                    "type": "clarification",
                    "question": "I found 2 earthquake data sources. Which one would you like to see?",
                    "options": [
                        "Philippines - Latest earthquake data from PHIVOLCS (Philippine Institute of Volcanology and Seismology)",
                        "Global - Worldwide earthquake data from USGS (U.S. Geological Survey)",
                        "Both - Show both Philippine and Global earthquake data"
                    ],
                    "suggested_action": hazard_action,
                    "pending_map_actions": map_actions
                }
            return None
        
        # Check for map style clarification responses (Dark vs Navigation Night)
        # More flexible matching for natural language
        if "dark" in last_message and "navigation" not in last_message:
            suggested = pending_action.get("suggested_action", {})
            if suggested.get("tool") == "change_map_style":
                map_action = {"tool": "change_map_style", "style": "dark", "requires_frontend": True}
                
                # Check if there's a pending hazard action
                hazard_action = pending_action.get("pending_hazard_action")
                
                # Collect all map actions
                map_actions = [map_action]
                additional_map = pending_action.get("additional_map_actions")
                conflict_note = pending_action.get("conflict_note")
                
                if additional_map:
                    map_actions.extend(additional_map)
                
                # Check if hazard needs clarification
                second_clarification = check_hazard_clarification(hazard_action, map_actions)
                if second_clarification:
                    # Pass conflict note to second clarification
                    if conflict_note:
                        second_clarification["conflict_note"] = conflict_note
                    state["clarification_needed"] = True
                    state["pending_action"] = second_clarification
                    state["messages"].append(AIMessage(content=json.dumps(second_clarification)))
                    return state
                
                # No second clarification needed, execute all
                actions = ([hazard_action] if hazard_action else []) + map_actions
                
                if len(actions) > 1:
                    response = {
                        "multiple_actions": actions,
                        "requires_frontend": True
                    }
                    if conflict_note:
                        response["info_message"] = conflict_note
                    state["messages"].append(AIMessage(content=json.dumps(response)))
                else:
                    response = map_action.copy()
                    if conflict_note:
                        response["info_message"] = conflict_note
                    state["messages"].append(AIMessage(content=json.dumps(response)))
                
                state["clarification_needed"] = False
                state["pending_action"] = None
                return state
        
        if "navigation" in last_message or ("night" in last_message and last_message not in ["dark", "1"]):
            suggested = pending_action.get("suggested_action", {})
            if suggested.get("tool") == "change_map_style":
                map_action = {"tool": "change_map_style", "style": "navigation_night", "requires_frontend": True}
                
                # Check if there's a pending hazard action
                hazard_action = pending_action.get("pending_hazard_action")
                
                # Collect all map actions
                map_actions = [map_action]
                additional_map = pending_action.get("additional_map_actions")
                if additional_map:
                    map_actions.extend(additional_map)
                
                # Check if hazard needs clarification
                second_clarification = check_hazard_clarification(hazard_action, map_actions)
                if second_clarification:
                    state["clarification_needed"] = True
                    state["pending_action"] = second_clarification
                    state["messages"].append(AIMessage(content=json.dumps(second_clarification)))
                    return state
                
                # No second clarification needed, execute all
                actions = ([hazard_action] if hazard_action else []) + map_actions
