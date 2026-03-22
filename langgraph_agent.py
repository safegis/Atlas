"""
LangGraph Multi-Agent System for SafeGIS AI
============================================

This is the main entry point for the modular LangGraph system.
All components have been extracted into separate modules for better maintainability.

Module Structure:
-----------------
- state/         : AgentState definition
- tools/         : Tool definitions (search_location, change_map_style, etc.)
- agents/        : Agent classes (RouterAgent, MapAgent, HazardAgent, QAAgent, ClarificationAgent)
- graph/         : Workflow creation and message processing
- utils/         : Helper utilities (LlamaCppWrapper)

For new code, import directly from the specific modules:
    from agents import MapAgent, HazardAgent
    from graph import create_agent_graph, process_message
    from state import AgentState

This file maintains backward compatibility by re-exporting all components.
"""

# Re-export state
from state import AgentState

# Re-export tools
from tools import (
    search_location,
    change_map_style,
    switch_view_mode,
    control_earthquake_data,
    control_tsunami_data,
    control_weather_data,
    ask_clarification,
    TOOLS
)

# Re-export agents
from agents import (
    RouterAgent,
    MapAgent,
    HazardAgent,
    QAAgent,
    ClarificationAgent
)

# Re-export graph functions
from graph import create_agent_graph, process_message

# Re-export utilities
from utils import LlamaCppWrapper

# Define what's available when using "from langgraph_agent import *"
__all__ = [
    # State
    "AgentState",
    
    # Tools
    "search_location",
    "change_map_style",
    "switch_view_mode",
    "control_earthquake_data",
    "control_tsunami_data",
    "control_weather_data",
    "ask_clarification",
    "TOOLS",
    
    # Agents
    "RouterAgent",
    "MapAgent",
    "HazardAgent",
    "QAAgent",
    "ClarificationAgent",
    
    # Graph
    "create_agent_graph",
    "process_message",
    
    # Utils
    "LlamaCppWrapper"
]

# Version info
__version__ = "2.0.0"
__status__ = "Modular"
