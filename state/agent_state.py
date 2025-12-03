"""Agent state definition for LangGraph workflow"""
from typing import TypedDict, Annotated, Sequence
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
    pending_action: dict | None
