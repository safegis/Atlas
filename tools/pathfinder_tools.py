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
def set_pathfinder_tab(pathfinder_tab: str) -> dict:
    """
    Switch Pathfinder between point-to-point routing and shelter/evacuation (OSM) mode.

    Args:
        pathfinder_tab: "destination" for Set Destination (two addresses), or
            "evacuation" for Find Shelter/s. Synonyms: shelter, shelters, evac, point_to_point.

    Returns:
        dict instructing the frontend which tab to show
    """
    raw = (pathfinder_tab or "").strip().lower().replace(" ", "_")
    if raw in ("shelter", "shelters", "evacuation", "evac", "find_shelter", "find_shelters"):
        tab = "evacuation"
    elif raw in ("destination", "point_to_point", "point-to-point", "set_destination", "address"):
        tab = "destination"
    else:
        return {
            "error": 'Use pathfinder_tab "destination" or "evacuation".',
            "requires_clarification": True,
        }
    return {
        "tool": "set_pathfinder_tab",
        "pathfinder_tab": tab,
        "action": "set_pathfinder_tab",
        "requires_frontend": True,
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
def select_evacuation_destination(place_name: str, mode: str = "all") -> dict:
    """
    Select a shelter / school / facility already loaded in Pathfinder's evacuation (OSM) list
    by name, and recalculate routes from the current start to that place.

    Args:
        place_name: Name (or substring) matching an item in the evacuation dropdown, e.g. school name
        mode: Transportation mode — all, driving, walking, cycling, motorcycle

    Returns:
        dict instructing the frontend to select the row and refresh routes
    """
    valid_modes = ["all", "driving", "walking", "cycling", "motorcycle"]
    m = (mode or "all").lower()
    if m not in valid_modes:
        return {
            "error": f"Invalid mode. Choose from: {', '.join(valid_modes)}",
            "requires_clarification": True,
            "available_options": valid_modes,
        }
    name = (place_name or "").strip()
    if len(name) < 2:
        return {
            "error": "Need a place name to select (e.g. the school or shelter name from the list).",
            "requires_clarification": True,
        }
    return {
        "tool": "select_evacuation_destination",
        "place_name": name,
        "mode": m,
        "action": "select_evacuation_destination",
        "requires_frontend": True,
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


@tool
def clear_pathfinder_routes() -> dict:
    """
    Remove calculated routes from the map and reset Pathfinder route inputs (same as the
    in-panel "Clear Routes" button). Does NOT hide the pathfinder panel.

    Use when the user wants to clear / wipe / remove routes drawn on the map, not when
    they only want to collapse the pathfinder sidebar.

    Returns:
        dict instructing the frontend to clear routes and related markers
    """
    return {
        "tool": "clear_pathfinder_routes",
        "action": "clear_pathfinder_routes",
        "requires_frontend": True,
    }


# All available pathfinder tools
PATHFINDER_TOOLS = [
    find_route,
    change_route_mode,
    change_route_sort,
    select_evacuation_destination,
    set_pathfinder_tab,
    open_pathfinder,
    close_pathfinder,
    clear_pathfinder_routes,
]
