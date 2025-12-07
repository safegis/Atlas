"""Tool definitions for map and hazard control"""
from langchain_core.tools import tool


@tool
def search_location(query: str) -> dict:
    """
    Search for a location on the map using geocoding.
    
    Args:
        query: Location name, address, or landmark to search for
        
    Returns:
        dict with location details and coordinates
    """
    return {
        "tool": "search_location",
        "query": query,
        "action": "geocode_and_fly",
        "requires_frontend": True
    }


@tool
def change_map_style(style: str) -> dict:
    """
    Change the map visual style.
    
    Args:
        style: One of: "default", "satellite", "outdoors", "light", "dark", 
               "navigation_day", "navigation_night"
               
    Returns:
        dict with style change action
    """
    valid_styles = ["default", "satellite", "outdoors", "light", "dark", 
                   "navigation_day", "navigation_night"]
    
    if style.lower() not in valid_styles:
        return {
            "error": f"Invalid style. Choose from: {', '.join(valid_styles)}",
            "requires_clarification": True,
            "available_options": valid_styles
        }
    
    return {
        "tool": "change_map_style",
        "style": style,
        "action": "update_style",
        "requires_frontend": True
    }


@tool
def switch_view_mode(mode: str) -> dict:
    """
    Switch between 2D and 3D map view.
    
    Args:
        mode: Either "2d" or "3d"
        
    Returns:
        dict with view mode change action
    """
    if mode.lower() not in ["2d", "3d"]:
        return {
            "error": "Invalid mode. Choose '2d' or '3d'",
            "requires_clarification": True
        }
    
    return {
        "tool": "switch_view_mode",
        "mode": mode.lower(),
        "action": "update_view",
        "requires_frontend": True
    }


@tool
def control_earthquake_data(action: str, source: str = "philippine") -> dict:
    """
    Enable or disable earthquake data monitoring.
    
    Args:
        action: Either "enable" or "disable"
        source: Either "philippine" (PHIVOLCS) or "global" (USGS)
        
    Returns:
        dict with earthquake control action
    """
    return {
        "tool": "control_earthquake_data",
        "action": action,
        "source": source,
        "requires_frontend": True
    }


@tool
def control_weather_data(action: str, scope: str = "province") -> dict:
    """
    Enable or disable weather data monitoring.
    
    Args:
        action: Either "enable" or "disable"
        scope: Either "province" or "city"
        
    Returns:
        dict with weather control action
    """
    return {
        "tool": "control_weather_data",
        "action": action,
        "scope": scope,
        "requires_frontend": True
    }


@tool
def control_time_of_day(preset: str) -> dict:
    """
    Control the time of day lighting in 3D view.
    
    Args:
        preset: One of: "auto", "morning", "daytime", "evening", "nighttime"
        
    Returns:
        dict with time of day control action
    """
    valid_presets = ["auto", "morning", "daytime", "evening", "nighttime"]
    
    if preset.lower() not in valid_presets:
        return {
            "error": f"Invalid preset. Choose from: {', '.join(valid_presets)}",
            "requires_clarification": True,
            "available_options": valid_presets
        }
    
    return {
        "tool": "control_time_of_day",
        "preset": preset.lower(),
        "action": "update_lighting",
        "requires_frontend": True
    }


@tool
def ask_clarification(question: str, options: list[str]) -> dict:
    """
    Ask the user for clarification when intent is ambiguous.
    
    Args:
        question: The clarification question to ask
        options: List of available options for the user
        
    Returns:
        dict with clarification request
    """
    return {
        "tool": "ask_clarification",
        "question": question,
        "options": options,
        "requires_user_response": True
    }


# All available tools
TOOLS = [
    search_location,
    change_map_style,
    switch_view_mode,
    control_earthquake_data,
    control_weather_data,
    control_time_of_day,
    ask_clarification
]
