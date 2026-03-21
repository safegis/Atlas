"""QA agent for handling general questions and explanations"""
import json
from types import SimpleNamespace
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from state import AgentState


def _unwrap_nested_json_text(s: str, max_depth: int = 10) -> str:
    """
    If s is JSON (possibly repeatedly) with a string 'text' field, return the innermost plain text.
    Fixes models echoing prior turns as JSON inside the answer text.
    """
    t = (s or "").strip()
    for _ in range(max_depth):
        if not t.startswith("{") or '"text"' not in t:
            break
        try:
            obj = json.loads(t)
        except json.JSONDecodeError:
            break
        if not isinstance(obj, dict):
            break
        inner = obj.get("text")
        if not isinstance(inner, str):
            break
        t = inner.strip()
    return t


class QAAgent:
    """Handles general questions and explanations; optional RAG from Qdrant + Ollama embeddings."""

    def __init__(self, llm, rag_retriever: Optional[Any] = None):
        self.llm = llm
        self.rag_retriever = rag_retriever

    @staticmethod
    def _role_and_content(m) -> tuple[str | None, str]:
        """Support LangChain messages and API JSON dicts ({role, content} from /chat)."""
        if isinstance(m, dict):
            raw_role = (m.get("role") or m.get("type") or "").strip().lower()
            content = (m.get("content") or "").strip()
            if raw_role in ("user", "human"):
                return "human", content
            if raw_role in ("assistant", "ai"):
                return "ai", content
            return None, content
        t = getattr(m, "type", None)
        if t is None:
            if isinstance(m, HumanMessage):
                t = "human"
            elif isinstance(m, AIMessage):
                t = "ai"
        content = (getattr(m, "content", None) or "").strip()
        if t == "human":
            return "human", content
        if t == "ai":
            return "ai", content
        return None, content

    @staticmethod
    def _last_human_content(messages) -> str:
        for m in reversed(messages):
            role, content = QAAgent._role_and_content(m)
            if role == "human" and content:
                return content
        return ""

    @staticmethod
    def _messages_for_llm_dialogue(messages) -> list:
        """
        OllamaWrapper flattens history into one user blob — strip JSON from past assistant
        turns so the model answers in plain language, not nested JSON strings.
        """
        out: list = []
        for m in messages:
            role, content = QAAgent._role_and_content(m)
            if not content:
                continue
            if role == "human":
                out.append(SimpleNamespace(content=f"User: {content}"))
            elif role == "ai":
                text = content
                if content.startswith("{"):
                    try:
                        p = json.loads(content)
                        if isinstance(p, dict) and isinstance(p.get("text"), str):
                            text = _unwrap_nested_json_text(p["text"])
                    except json.JSONDecodeError:
                        pass
                out.append(SimpleNamespace(content=f"Assistant: {text}"))
        return out

    def process(self, state: AgentState) -> AgentState:
        """Process Q&A requests"""
        messages = state["messages"]

        print(f"QA Agent processing {len(messages)} messages (RAG={'on' if self.rag_retriever else 'off'})")

        base_system = """You are SafeGIS AI, an expert assistant specialized in:
- Disaster management and emergency response
- GIS (Geographic Information Systems) and mapping
- Hazard assessment and spatial analysis
- Geospatial technology and remote sensing

Other agents control the live map (hazard monitor, layers, navigation). **You cannot enable layers or change the map yourself** — your replies use `requires_frontend: false` and do not send map commands.

When answering questions:
1. Provide concise, clear, and informative answers
2. If **retrieved knowledge base excerpts** are provided below, ground factual claims in them and cite like [1], [2] when using them
3. If NO excerpt block appears below, do **not** cite [1]/[2] or claim the "knowledge base" said something specific — answer from general expertise only
4. You may **offer** a map demo, but never claim you already enabled or displayed data on the map. If the user agrees, tell them to send a short command the system understands, e.g. **"Enable Philippine earthquake data"** or **"Show PHIVOLCS earthquakes"**, or to open the Live Hazard Monitor from the UI.
5. Be proactive in suggesting those follow-up commands when relevant

The conversation history uses plain "User:" / "Assistant:" lines only. Reply as **plain natural language** (no JSON, no code fences wrapping JSON).

Do not generate code or scripts.
If asked about unrelated topics, politely redirect to your expertise in GIS and disaster management."""

        last_query = self._last_human_content(messages)
        conv_id = state.get("conversation_id")
        rag_sources: list[dict] = []
        rag_context = ""
        if self.rag_retriever and last_query.strip():
            try:
                hits = self.rag_retriever.retrieve(
                    last_query.strip(), conversation_id=conv_id
                )
                print(f"QA RAG: {len(hits)} hit(s) for query preview={last_query.strip()[:80]!r}")
                rag_context = self.rag_retriever.format_context(hits)
                rag_sources = [
                    {
                        "source_id": h.get("source_id"),
                        "score": round(float(h.get("score") or 0), 4),
                    }
                    for h in hits
                    if h.get("text")
                ]
            except Exception as e:
                print(f"QA RAG retrieve error (continuing without context): {e}")

        if rag_context:
            system_prompt = (
                base_system
                + "\n\n---\n## Retrieved knowledge base excerpts (use when relevant; cite [n]):\n"
                + rag_context
                + "\n---"
            )
        else:
            system_prompt = base_system

        llm_messages = self._messages_for_llm_dialogue(messages)
        response = self.llm.invoke(llm_messages, system_prompt=system_prompt)
        if isinstance(response, str):
            response = _unwrap_nested_json_text(response)

        payload: dict = {"text": response, "requires_frontend": False}
        if rag_sources:
            payload["rag_used"] = True
            payload["rag_sources"] = rag_sources
        else:
            payload["rag_used"] = False

        json_response = json.dumps(payload)
        # Append-only for LangGraph operator.add on messages (do not return full state.messages)
        return {"messages": [AIMessage(content=json_response)]}
