"""Tool definitions for pathfinder/routing functionality"""
from langchain_core.tools import tool


@tool
def find_route(start: str, destination: str, mode: str = "all") -> dict:
    """
    Find routes between two locations using the pathfinder.
    
    Args:
        start: Starting location (address, place name, or landmark)
        destination: Destination location (address, place name, or landmark)
        mode: Transportation mode - one of: "all", "driving", "walking", "cycling", "motorcycle"
        
    Returns:
        dict with route finding action
    """
    valid_modes = ["all", "driving", "walking", "cycling", "motorcycle"]
    
    if mode.lower() not in valid_modes:
        return {
            "error": f"Invalid mode. Choose from: {', '.join(valid_modes)}",
            "requires_clarification": True,
            "available_options": valid_modes
        }
    
    return {
        "tool": "find_route",
        "start": start,
        "destination": destination,
        "mode": mode.lower(),
        "action": "calculate_routes",
        "requires_frontend": True
    }


@tool
def change_route_mode(mode: str) -> dict:
    """
    Change the transportation mode for existing routes.
    
    Args:
        mode: Transportation mode - one of: "all", "driving", "walking", "cycling", "motorcycle"
        
    Returns:
        dict with mode change action
    """
    valid_modes = ["all", "driving", "walking", "cycling", "motorcycle"]
    
    if mode.lower() not in valid_modes:
        return {
            "error": f"Invalid mode. Choose from: {', '.join(valid_modes)}",
            "requires_clarification": True,
            "available_options": valid_modes
        }
    
    return {
        "tool": "change_route_mode",
        "mode": mode.lower(),
        "action": "update_route_mode",
        "requires_frontend": True
    }


@tool
def change_route_sort(sort_by: str) -> dict:
    """
    Change how routes are sorted in the pathfinder.
    
    Args:
        sort_by: Sorting method - one of: "fastest", "safest", "best_balance"
        
    Returns:
        dict with sort change action
    """
    valid_sorts = ["fastest", "safest", "best_balance"]
    
    if sort_by.lower() not in valid_sorts:
        return {
            "error": f"Invalid sort method. Choose from: {', '.join(valid_sorts)}",
            "requires_clarification": True,
            "available_options": valid_sorts
        }
    
    return {
        "tool": "change_route_sort",
        "sort_by": sort_by.lower(),
        "action": "update_route_sort",
        "requires_frontend": True
    }


@tool
def open_pathfinder() -> dict:
    """
    Open the pathfinder panel to allow manual route planning.
    
    Returns:
        dict with pathfinder open action
    """
    return {
        "tool": "open_pathfinder",
        "action": "show_pathfinder",
        "requires_frontend": True
    }


@tool
def close_pathfinder() -> dict:
    """
    Close the pathfinder panel.
    
    Returns:
        dict with pathfinder close action
    """
    return {
        "tool": "close_pathfinder",
        "action": "hide_pathfinder",
        "requires_frontend": True
    }


# All available pathfinder tools
PATHFINDER_TOOLS = [
    find_route,
    change_route_mode,
    change_route_sort,
    open_pathfinder,
    close_pathfinder
]
