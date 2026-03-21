"""LangGraph workflow creation and message processing"""
import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from state import AgentState
from agents import RouterAgent, MapAgent, HazardAgent, QAAgent, ClarificationAgent, WebSearchAgent, ExposureAssessmentAgent, PathfinderAgent, UIControlAgent, SpatialDataAgent
from utils import LlamaCppWrapper
import os


def create_agent_graph(llm=None, llm_wrapper=None, exa_api_key: str = None, rag_retriever=None):
    """Create the LangGraph multi-agent system.
    Pass either llm (for llama-cpp GGUF) or llm_wrapper (e.g. OllamaWrapper for Ollama).
    """
    if llm_wrapper is not None:
        wrapped_llm = llm_wrapper
    else:
        if llm is None:
            raise ValueError("Provide either llm (GGUF) or llm_wrapper (e.g. OllamaWrapper).")
        wrapped_llm = LlamaCppWrapper(llm)
    
    # Initialize agents
    router = RouterAgent(wrapped_llm)
    map_agent = MapAgent(wrapped_llm)
    hazard_agent = HazardAgent(wrapped_llm)
    qa_agent = QAAgent(wrapped_llm, rag_retriever=rag_retriever)
    clarification_agent = ClarificationAgent()
    exposure_agent = ExposureAssessmentAgent(wrapped_llm)
    pathfinder_agent = PathfinderAgent(wrapped_llm)
    ui_agent = UIControlAgent(wrapped_llm)
    spatial_data_agent = SpatialDataAgent(wrapped_llm)

    # Initialize web search agent if API key provided
    web_search_agent = None
    if exa_api_key:
        web_search_agent = WebSearchAgent(wrapped_llm, exa_api_key)
    
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    # Only return deltas — do not spread full state (would re-submit `messages` and duplicate with operator.add)
    workflow.add_node("router", lambda state: {"current_agent": router.route(state)})
    workflow.add_node("map_agent", map_agent.process)
    workflow.add_node("hazard_agent", hazard_agent.process)
    workflow.add_node("qa_agent", qa_agent.process)
    workflow.add_node("clarification_agent", clarification_agent.process)
    workflow.add_node("exposure_assessment_agent", exposure_agent.process)
    workflow.add_node("pathfinder_agent", pathfinder_agent.process)
    workflow.add_node("ui_agent", ui_agent.process)
    workflow.add_node("spatial_data_agent", spatial_data_agent.process)

    # Add web search node if available
    if web_search_agent:
        workflow.add_node("web_search_agent", web_search_agent.process)
    
    # Define routing logic
    def route_to_agent(state: AgentState) -> str:
        """Route to the appropriate agent based on router decision"""
        return state["current_agent"]
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router
    routing_map = {
        "map_agent": "map_agent",
        "hazard_agent": "hazard_agent",
        "qa_agent": "qa_agent",
        "clarification_agent": "clarification_agent",
        "exposure_assessment_agent": "exposure_assessment_agent",
        "pathfinder_agent": "pathfinder_agent",
        "ui_agent": "ui_agent",
        "spatial_data_agent": "spatial_data_agent",
    }
    
    # Add web search routing if available
    if web_search_agent:
        routing_map["web_search_agent"] = "web_search_agent"
    
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        routing_map
    )
    
    # All agents end after processing
    workflow.add_edge("map_agent", END)
    workflow.add_edge("hazard_agent", END)
    workflow.add_edge("qa_agent", END)
    workflow.add_edge("clarification_agent", END)
    workflow.add_edge("exposure_assessment_agent", END)
    workflow.add_edge("pathfinder_agent", END)
    workflow.add_edge("ui_agent", END)
    workflow.add_edge("spatial_data_agent", END)

    # Add web search edge if available
    if web_search_agent:
        workflow.add_edge("web_search_agent", END)
    
    return workflow.compile()


def process_message(graph, message: str, conversation_history: list = None, map_state: dict = None, web_search_enabled: bool = False, uploaded_files: list = None, spatial_context: list = None, conversation_id: Optional[str] = None) -> dict:
    """
    Process a user message through the agent graph
    
    Args:
        graph: Compiled LangGraph
        message: User message
        conversation_history: Previous messages
        map_state: Current map state
        conversation_id: Optional thread UUID; enables Qdrant memory scoped to this chat only.

    Returns:
        dict with response and actions
    """
    try:
        print(f"\n=== Processing Message: {message} ===")
        
        # Initialize state - LIMIT conversation history to prevent context overflow
        messages = conversation_history or []
        
        # Keep only last 8 messages to prevent context window overflow
        # With n_ctx=4096, system_prompt=~200, max_tokens=2048, we need ~1800 tokens for history+input
        if len(messages) > 8:
            print(f"Trimming conversation history from {len(messages)} to 8 messages")
            messages = messages[-8:]
        
        messages.append(HumanMessage(content=message))
        
        # Store web search flag in state
        web_search_flag = web_search_enabled
        
        # Check if there's a pending action from the last AI message
        pending_action = None
        clarification_needed = False
        
        # Only look for pending actions if the last message looks like a clarification response
        # This prevents old clarifications from interfering with new requests
        # Don't treat it as clarification if it has action keywords like "show", "enable", "disable", etc.
        # Short tokens like "now" / "also" match normal continuations ("Now for the exposure…", "Also use…")
        # and wrongly block restoring pending_action — keep multi-word phrases where needed.
        action_keywords = [
            "show",
            "enable",
            "disable",
            "turn on",
            "turn off",
            "activate",
            "deactivate",
            "hide",
            "remove",
            "display",
            "view",
            "see",
        ]
        
        potential_responses = ["yes", "yeah", "yep", "sure", "ok", "okay", "no", "nope", "nah", "cancel",
                              "philippines", "philippine", "phivolcs", "local", 
                              "global", "usgs", "worldwide", "world",
                              "both", "all",
                              "province", "provincial", "region",
                              "city", "municipality", "municipal",
                              "abra", "aklan", "agusan del norte", "agusan del sur",
                              "dark", "navigation", "navigation night",
                              "existing", "imported", "import", "upload", "file", "files", "system",
                              "run", "start", "go", "proceed", "confirm", "stop", "abort",
                              "option 1", "option 2", "option 3", "option 4", "option 5", "option 6",
                              "option one", "option two", "option three",
                              "go with", "use option", "choose option", "select option", "pick option",
                              "1", "2", "3", "4", "5", "6"]
        
        msg_lower = message.lower().strip()
        
        # Check if message contains action keywords - if so, it's a new request, not a clarification response
        has_action_keyword = any(keyword in msg_lower for keyword in action_keywords)
        # "Show both" / "Display global" answer a clarification; "show" must not block pending_action restore
        if has_action_keyword and re.match(
            r"^\s*(show|display)\s+(both|all|global|philippines?|phivolcs|usgs|worldwide|world)(\s+please)?\s*\.?\s*$",
            msg_lower,
        ):
            has_action_keyword = False
        # "Show available layers" / list-style layer requests (not a location search)
        if has_action_keyword and re.match(
            r"^\s*(show|display|list|open)\s+.*\blayers?\b",
            msg_lower,
        ):
            has_action_keyword = False
        
        # Check if message contains file references (for exposure assessment clarifications)
        has_file_reference = any(ext in msg_lower for ext in [".geojson", ".shp", ".kml", ".gpkg", ".json", ".csv"])
        
        # Only treat as clarification if it's a simple response without action keywords
        is_likely_clarification_response = ((msg_lower in potential_responses or 
                                            any(resp in msg_lower for resp in potential_responses) or
                                            has_file_reference) and not has_action_keyword)
        
        if is_likely_clarification_response:
            for msg in reversed(messages[:-1]):  # Check all messages except the new user message
                if hasattr(msg, 'content') or isinstance(msg, dict):
                    content = msg.content if hasattr(msg, 'content') else msg.get('content')
                    msg_type = getattr(msg, 'type', None) if hasattr(msg, 'type') else msg.get('role')
                    
                    if msg_type in ['ai', 'assistant'] and content:
                        try:
                            parsed = json.loads(content)
                            if parsed.get("type") == "clarification":
                                pending_action = parsed
                                clarification_needed = True
                                # Log what kind of pending action we found
                                if "pending_hazard_action" in parsed:
                                    print(f"Found pending action from previous message: {parsed.get('pending_hazard_action')}")
                                elif "pending_map_actions" in parsed:
                                    print(f"Found pending map actions from previous message: {parsed.get('pending_map_actions')}")
                                else:
                                    print(f"Found clarification from previous message: {parsed.get('suggested_action')}")
                                break
                        except:
                            pass
        
        initial_state = {
            "messages": messages,
            "current_agent": "",
            "conversation_context": {},
            "map_state": map_state or {},
            "user_intent": "",
            "clarification_needed": clarification_needed,
            "pending_action": pending_action,
            "web_search_enabled": web_search_flag,
            "uploaded_files": uploaded_files or [],
            "spatial_context": spatial_context or [],
            "conversation_id": (conversation_id or "").strip() or None,
        }
        
        print(f"Initial state created with {len(messages)} messages")
        print(f"Clarification needed: {clarification_needed}, Pending action: {pending_action is not None}")
        
        # Run the graph
        print("Running agent graph...")
        result = graph.invoke(initial_state)
        print(f"Graph completed. Result has {len(result['messages'])} messages")
        
        # Extract response
        last_message = result["messages"][-1]
        print(f"Last message type: {type(last_message)}, content: {last_message.content[:100] if hasattr(last_message, 'content') else last_message}")
        
        # Parse response
        if isinstance(last_message, AIMessage):
            try:
                content = json.loads(last_message.content)
                print(f"Parsed JSON response: {content}")
                return {
                    "response": content,
                    "requires_frontend": content.get("requires_frontend", False),
                    "requires_clarification": content.get("type") == "clarification",
                    "conversation_history": result["messages"]
                }
            except json.JSONDecodeError:
                print(f"Not JSON, returning as text")
                return {
                    "response": {"text": last_message.content},
                    "requires_frontend": False,
                    "conversation_history": result["messages"]
                }
        
        print("Fallback response")
        return {
            "response": {"text": "I'm not sure how to help with that."},
            "requires_frontend": False,
            "conversation_history": result["messages"]
        }
    except Exception as e:
        print(f"Error in process_message: {e}")
        import traceback
        traceback.print_exc()
        return {
            "response": {"text": f"Error processing message: {str(e)}"},
            "requires_frontend": False,
            "conversation_history": []
        }
