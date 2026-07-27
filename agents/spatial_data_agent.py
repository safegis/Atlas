"""Agent for user-connected spatial data: imports, APIs, PostGIS, MCP-style URLs."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage

from state import AgentState


class SpatialDataAgent:
    """
    Handles requests about layers the user added via Import / connect spatial data
    (local files, API, PostGIS, HTTP GeoJSON for MCP-style endpoints).

    Returns JSON actions consumed by Simulation Studio (LangGraphAdapter).
    """

    def __init__(self, llm):
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        last_message = messages[-1].content
        msg_lower = last_message.lower().strip()

        spatial_ctx: list = state.get("spatial_context") or []
        uploaded_names: list = state.get("uploaded_files") or []

        # Chat-bar import already completed on the frontend this turn
        if "spatial files imported onto the map this turn" in msg_lower:
            text = (
                "Done — those spatial file(s) are on the map and listed under "
                "**Import / connect spatial data → On map**.\n\n"
                + self._describe_context(spatial_ctx, uploaded_names)
            )
            state["messages"].append(
                AIMessage(
                    content=json.dumps(
                        {
                            "text": text,
                            "requires_frontend": False,
                        }
                    )
                )
            )
            return state

        # --- Rule-based: inventory / analysis of what's already on the map ---
        inventory_triggers = (
            "what layer",
            "which layer",
            "list layer",
            "layers on the map",
            "layers on my map",
            "my layers",
            "imported data",
            "uploaded data",
            "spatial layers",
            "what did i import",
            "what's on the map",
            "whats on the map",
        )
        if any(t in msg_lower for t in inventory_triggers):
            text = self._describe_context(spatial_ctx, uploaded_names)
            state["messages"].append(
                AIMessage(
                    content=json.dumps(
                        {
                            "text": text,
                            "requires_frontend": False,
                        }
                    )
                )
            )
            return state

        # --- LLM: open panel, add layer from URL, or conversational reply ---
        system_prompt = f"""You are Atlas (SafeGIS) spatial data intent parser.

The user may have already added layers from local files, APIs, PostGIS, or HTTP GeoJSON.
They can also attach GeoJSON / KML / SHP / ZIP in the Atlas chat bar; Simulation Studio imports those onto the map before this agent runs (look for "[Spatial files imported onto the map this turn: ...]" in the message).

Current spatial context (JSON array of objects with name, layerName, sourceType):
{json.dumps(spatial_ctx, indent=0)}

Legacy uploaded file names only (strings): {json.dumps(uploaded_names)}

Return ONE JSON object (no markdown, no prose outside JSON):

1) Open import/connect panel (when they want the UI but did not attach files in chat):
{{"tool": "open_spatial_data_panel", "requires_frontend": true, "text": "short confirmation"}}

2) Add a layer by fetching GeoJSON from a URL (Simulation Studio backend will proxy the request):
{{"tool": "add_spatial_layer_from_url", "requires_frontend": true,
  "url": "https://...", "method": "GET"|"POST", "auth_type": "none"|"bearer"|"apikey_header"|"apikey_query"|"basic",
  "bearer_or_key_value": null or string, "api_key_header_name": "X-API-Key", "api_query_param_name": "api_key",
  "basic_user": null, "basic_password": null, "post_body_json": null or JSON string,
  "layer_display_name": "optional short label",
  "text": "short confirmation"}}

3) Only conversation / no UI action:
{{"text": "helpful answer about their layers or how to connect data", "requires_frontend": false}}

Rules:
- If the message says spatial files were already imported this turn, use (3): confirm they are on the map and appear under Import / connect → On map. Do NOT open the panel unless they ask.
- If they ask to upload/import spatial files but none were attached and no URL is given, use (1) to open the panel and tell them they can also attach .geojson/.kml/.shp/.zip in chat.
- If the user asks how to connect PostGIS or MCP, explain that PostGIS is supported from the SQL tab in the panel, and MCP full protocol is limited—HTTP endpoints returning GeoJSON work via API or this tool with a URL.
- If URLs are missing for case (2), use (3) and ask for the URL.
- Never invent secrets; use null for unknown tokens.
"""

        try:
            raw = self.llm.invoke(last_message, system_prompt=system_prompt)
            raw = (raw or "").strip()
            if raw.startswith("```json"):
                raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
            elif raw.startswith("```"):
                raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
            parsed: dict[str, Any] = json.loads(raw)
        except Exception as e:
            print(f"SpatialDataAgent LLM parse error: {e}")
            parsed = {
                "text": self._describe_context(spatial_ctx, uploaded_names)
                + "\n\nI couldn't parse a specific action. Open **Import / connect spatial data** from the toolbar to add API, database, or file layers.",
                "requires_frontend": False,
            }

        if parsed.get("tool") == "open_spatial_data_panel":
            parsed["requires_frontend"] = True
        elif parsed.get("tool") == "add_spatial_layer_from_url":
            parsed["requires_frontend"] = True
            if not parsed.get("url"):
                parsed = {
                    "text": "I need a full **https** URL that returns GeoJSON (FeatureCollection or Feature). Paste it in chat or use the API tab in Import / connect spatial data.",
                    "requires_frontend": False,
                }
        else:
            parsed.setdefault("requires_frontend", False)

        state["messages"].append(AIMessage(content=json.dumps(parsed)))
        return state

    def _describe_context(
        self, spatial_ctx: list, uploaded_names: list
    ) -> str:
        lines: list[str] = []
        if spatial_ctx:
            lines.append("**Layers currently tracked in this session:**")
            for i, row in enumerate(spatial_ctx, 1):
                if isinstance(row, dict):
                    nm = row.get("name") or "unnamed"
                    src = row.get("sourceType") or "unknown"
                    lines.append(f"{i}. **{nm}** (source: {src})")
                else:
                    lines.append(f"{i}. {row!s}")
        elif uploaded_names:
            lines.append("**Uploaded / map layers (names only):**")
            for i, n in enumerate(uploaded_names, 1):
                lines.append(f"{i}. {n}")
        else:
            lines.append(
                "No imported spatial layers are recorded in this chat session yet. "
                "Attach GeoJSON / KML / SHP / ZIP in the Atlas chat bar, or use the "
                "toolbar → **Import / connect spatial data** to add files, API endpoints, or PostGIS tables."
            )
        lines.append(
            "\nAsk me to **open the import panel**, **add a layer from a GeoJSON URL**, "
            "**upload attached spatial files** (via the chat paperclip), or **summarize layers** anytime."
        )
        return "\n".join(lines)
