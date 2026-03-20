"""UI Control Agent for opening panels, dropdowns, and controls in Simulation-Studio."""
import json
from typing import Optional
from langchain_core.messages import AIMessage
from state import AgentState


class UIControlAgent:
    """Handles open-panel / show-UI requests: chat expand, dropdowns, pathfinder, boundary panel, etc."""

    def __init__(self, llm):
        self.llm = llm
        # Rule-based intent parsing only; no LLM prompts

    def _normalize_step_message(self, message: str) -> str:
        """Extract main intent from step-by-step prompts (e.g. 'then open pathfinder' -> 'open pathfinder')."""
        msg_lower = message.lower().strip()
        for prefix in ("then ", "now ", "next ", "after that ", "and then ", "and also ", "also "):
            if msg_lower.startswith(prefix):
                return msg_lower[len(prefix) :].strip()
        return msg_lower

    def _parse_open_panel_intent(self, message: str) -> Optional[dict]:
        """Parse 'open panel' / 'show UI' type commands. Returns open_panel tool with panel id."""
        msg = self._normalize_step_message(message)
        msg_lower = msg.lower()
        open_show = any(
            w in msg_lower
            for w in ["open", "show", "display", "expand", "maximize", "minimize", "collapse"]
        )
        if not open_show and not any(
            w in msg_lower for w in ["panel", "dropdown", "full screen", "fullscreen", "bar"]
        ):
            return None

        # Chat expand / fullscreen
        if any(
            phrase in msg_lower
            for phrase in [
                "expand atlas",
                "atlas full",
                "full screen",
                "fullscreen",
                "maximize chat",
                "expand chat",
                "chat bar to full",
                "chat to full",
                "atlas chat bar",
                "expand the chat",
                "open atlas full",
            ]
        ) or ("expand" in msg_lower and "chat" in msg_lower) or (
            "atlas" in msg_lower and "full" in msg_lower
        ):
            return {
                "tool": "open_panel",
                "panel": "chat_expand",
                "requires_frontend": True,
                "text": "Expanding Atlas chat to full screen.",
            }
        # Chat collapse / minimize
        if any(
            phrase in msg_lower
            for phrase in ["minimize chat", "collapse chat", "minimize atlas", "collapse the chat"]
        ):
            return {
                "tool": "open_panel",
                "panel": "chat_collapse",
                "requires_frontend": True,
                "text": "Minimizing the chat panel.",
            }

        # Map style dropdown
        if any(
            phrase in msg_lower
            for phrase in [
                "map style dropdown",
                "map style menu",
                "open map style",
                "show map style",
                "style dropdown",
                "change map style",
                "map style panel",
                "open the map style",
            ]
        ) or ("map style" in msg_lower and open_show):
            return {
                "tool": "open_panel",
                "panel": "map_style_dropdown",
                "requires_frontend": True,
                "text": "Opening the map style dropdown.",
            }
        # Time of day / lighting dropdown
        if any(
            phrase in msg_lower
            for phrase in [
                "time of day dropdown",
                "time of day menu",
                "lighting dropdown",
                "open time of day",
                "show time of day",
                "lighting menu",
                "time dropdown",
            ]
        ) or (("time of day" in msg_lower or "lighting" in msg_lower) and open_show):
            return {
                "tool": "open_panel",
                "panel": "time_of_day_dropdown",
                "requires_frontend": True,
                "text": "Opening the time of day / lighting dropdown.",
            }

        # Boundary panel
        if any(
            phrase in msg_lower
            for phrase in [
                "add boundary panel",
                "boundary panel",
                "boundaries panel",
                "show add boundary",
                "open boundary panel",
                "open the add boundary",
                "show the boundary panel",
            ]
        ) or ("boundary" in msg_lower and "panel" in msg_lower):
            return {
                "tool": "open_panel",
                "panel": "boundary_panel",
                "requires_frontend": True,
                "text": "Opening the Add Boundaries panel.",
            }

        # Pathfinder panel
        if any(
            phrase in msg_lower
            for phrase in [
                "pathfinder panel",
                "pathfinder",
                "open pathfinder",
                "show pathfinder",
                "route panel",
                "routing panel",
                "directions panel",
                "open the pathfinder",
                "route finder",
            ]
        ):
            return {
                "tool": "open_panel",
                "panel": "pathfinder_panel",
                "requires_frontend": True,
                "text": "Opening the Pathfinder panel.",
            }

        # Layers panel (hazard / critical facility)
        if any(
            phrase in msg_lower
            for phrase in [
                "layers panel",
                "layer panel",
                "hazard layers",
                "critical facility layers",
                "open layers",
                "show layers",
                "available layers",
                "list layers",
                "list my layers",
                "what layers",
                "which layers",
                "map layers",
                "geological hazards panel",
            ]
        ):
            return {
                "tool": "open_panel",
                "panel": "layers_panel",
                "requires_frontend": True,
                "text": "Opening the Layers panel.",
            }

        # Import / upload panel
        if any(
            phrase in msg_lower
            for phrase in [
                "import panel",
                "import files",
                "upload panel",
                "upload files",
                "import map files",
                "import file",
                "import files",
                "open import",
                "show import",
                "add files panel",
            ]
        ):
            return {
                "tool": "open_panel",
                "panel": "import_files_panel",
                "requires_frontend": True,
                "text": "Opening Import / connect spatial data.",
            }

        # Live hazard monitor — check close/hide *before* open (same phrases match both)
        mentions_live_hazard_ui = any(
            phrase in msg_lower
            for phrase in [
                "live hazard monitor",
                "hazard monitor",
                "live hazards",
                "earthquake and weather panel",
            ]
        )
        if mentions_live_hazard_ui:
            close_hazard_ui = any(
                w in msg_lower
                for w in [
                    "close",
                    "hide",
                    "dismiss",
                    "shut",
                    "exit",
                    "collapse",
                    "minimize",
                ]
            ) or ("turn" in msg_lower and "off" in msg_lower)
            if close_hazard_ui:
                return {
                    "tool": "close_panel",
                    "panel": "live_hazard_monitor",
                    "requires_frontend": True,
                    "text": "Closing the Live Hazard Monitor panel.",
                }
            if any(
                phrase in msg_lower
                for phrase in [
                    "open hazard monitor",
                    "show hazard monitor",
                    "live hazard monitor",
                    "hazard monitor",
                    "live hazards",
                    "earthquake and weather panel",
                ]
            ):
                return {
                    "tool": "open_panel",
                    "panel": "live_hazard_monitor",
                    "requires_frontend": True,
                    "text": "Opening the Live Hazard Monitor panel.",
                }


        # Select maps
        if any(
            phrase in msg_lower
            for phrase in [
                "select maps",
                "map selection",
                "maps panel",
                "open select maps",
                "show select maps",
            ]
        ):
            return {
                "tool": "open_panel",
                "panel": "select_maps",
                "requires_frontend": True,
                "text": "Opening the Select Maps panel.",
            }

        # Planning tools
        if any(
            phrase in msg_lower
            for phrase in [
                "planning tools",
                "planning panel",
                "planning suite",
                "open planning",
                "show planning",
            ]
        ):
            return {
                "tool": "open_panel",
                "panel": "planning_tools",
                "requires_frontend": True,
                "text": "Opening the Planning Tools panel.",
            }

        # Assessment tools (exposure, vulnerability)
        if any(
            phrase in msg_lower
            for phrase in [
                "assessment tools",
                "assessment panel",
                "exposure panel",
                "vulnerability panel",
                "open assessment",
                "show assessment",
                "exposure assessment panel",
            ]
        ):
            return {
                "tool": "open_panel",
                "panel": "assessment_tools",
                "requires_frontend": True,
                "text": "Opening the Assessment Tools panel.",
            }

        # Tool panel (left sidebar)
        if any(
            phrase in msg_lower
            for phrase in [
                "tool panel",
                "tools panel",
                "left panel",
                "open tools",
                "show tools",
                "sidebar",
            ]
        ):
            return {
                "tool": "open_panel",
                "panel": "tool_panel",
                "requires_frontend": True,
                "text": "Opening the tool panel.",
            }

        return None

    def process(self, state: AgentState) -> AgentState:
        """Process UI/panel open requests."""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""

        print(f"UI Control Agent processing: {last_message}")

        action = self._parse_open_panel_intent(last_message)
        print(f"UI Control Agent parsed action: {action}")

        if action:
            state["messages"].append(AIMessage(content=json.dumps(action)))
        else:
            state["messages"].append(
                AIMessage(
                    content=json.dumps(
                        {
                            "text": "I can open panels and controls like the map style dropdown, Add Boundaries panel, Pathfinder, Layers, Import files, Live Hazard Monitor, and more. Try: 'Open the map style dropdown' or 'Show the pathfinder panel'."
                        }
                    )
                )
            )
        return state
