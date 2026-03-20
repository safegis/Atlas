"""Agent state definition for LangGraph workflow"""
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """State shared across all agents"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    conversation_context: dict
    map_state: dict
    user_intent: str
    clarification_needed: bool
    pending_action: Optional[dict]
    web_search_enabled: bool
    uploaded_files: list  # List of uploaded file names for exposure assessment
    # User-connected spatial layers (API / PostGIS / file / MCP-style URL) from Simulation Studio
    spatial_context: list
