import os
import json
from typing import TypedDict, Optional
from langchain_core.messages import SystemMessage
import dotenv

dotenv.load_dotenv()

# ── Reuse the robust fallback LLM from agent1 ─────────────────────────────────
try:
    from agents.agent1 import _call_llm_with_fallback
except ImportError:
    from langchain_groq import ChatGroq
    _llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY", ""))
    def _call_llm_with_fallback(prompt):
        try:
            r = _llm.invoke([SystemMessage(content=prompt)])
            return str(getattr(r, "content", "") or "").strip()
        except Exception:
            return ""

try:
    from langgraph.graph import StateGraph, END
    from langgraph_supervisor import create_supervisor
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


class SupervisorState(TypedDict):
    query: str
    action: str         # "route" | "route_alert" | "explain" | "full"
    needs_crime: bool
    needs_hospitals: bool
    needs_explanation: bool
    time_window: str    # e.g. "last 1 week", "last 24 hours"


def supervisor_node(state: SupervisorState):
    query = state.get("query", "")
    prompt = f"""You are a smart intent classifier for a night-safety navigation app.

Analyse the user query and output a JSON object (no markdown, no explanation):
{{
  "needs_route":        true/false,   // user wants a route
  "needs_crime":        true/false,   // user asks about crimes, incidents, news, or safety history
  "needs_hospitals":    true/false,   // user asks about hospitals, medical facilities, clinics
  "needs_explanation":  true/false,   // user asks WHY the route is safe, or for feature details
  "time_window":        "string",     // e.g. "last 1 week", "last 24 hours", "recent" — default "recent"
  "action":             "route|route_alert|explain|full"
                                       // "full" = needs_route + (crime or hospitals or explanation)
}}

Rules:
- action = "route_alert"   if needs_route AND needs_crime AND NOT needs_hospitals AND NOT needs_explanation
- action = "explain"       if NOT needs_route AND needs_explanation
- action = "full"          if needs more than one of: needs_crime, needs_hospitals, needs_explanation
- action = "route"         otherwise

User query: "{query}"
"""
    try:
        text = _call_llm_with_fallback(prompt)
        if not text:
            raise ValueError("empty")
        # Strip markdown fences
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else parts[0]
            if text.strip().startswith("json"):
                text = text.strip()[4:]
        data = json.loads(text.strip())
        return {
            "action":            str(data.get("action", "route")).strip(),
            "needs_crime":       bool(data.get("needs_crime", False)),
            "needs_hospitals":   bool(data.get("needs_hospitals", False)),
            "needs_explanation": bool(data.get("needs_explanation", False)),
            "time_window":       str(data.get("time_window", "recent")).strip(),
        }
    except Exception as e:
        print(f"[SUPERVISOR] Parse failed ({e}), falling back to keyword detection")
        return _keyword_fallback(query)


def _keyword_fallback(query: str) -> dict:
    q = query.lower()
    needs_crime      = any(w in q for w in ["crime", "incident", "robbery", "theft", "assault", "safe", "danger",
                                              "news", "recent", "week", "report", "attack"])
    needs_hospitals  = any(w in q for w in ["hospital", "clinic", "medical", "doctor", "emergency", "nearest"])
    needs_explanation = any(w in q for w in ["why", "explain", "reason", "feature", "factor", "how safe", "details"])
    needs_route      = any(w in q for w in ["route", "path", "navigate", "from", "to", "go", "safest", "travel", "tell me"])

    extras = needs_crime or needs_hospitals or needs_explanation
    if extras and needs_route:
        action = "full"
    elif needs_crime and not needs_hospitals and not needs_explanation:
        action = "route_alert"
    elif needs_explanation and not needs_route:
        action = "explain"
    else:
        action = "route"

    # Detect time window
    time_window = "recent"
    for phrase in ["last one week", "last 1 week", "past week", "7 days", "last week"]:
        if phrase in q:
            time_window = "last 1 week"
            break
    for phrase in ["last 24 hours", "today", "tonight"]:
        if phrase in q:
            time_window = "last 24 hours"
            break

    return {
        "action":            action,
        "needs_crime":       needs_crime,
        "needs_hospitals":   needs_hospitals,
        "needs_explanation": needs_explanation,
        "time_window":       time_window,
    }


if HAS_LANGGRAPH:
    from langgraph.graph import StateGraph, END
    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", END)
    supervisor_agent = workflow.compile()
else:
    # Minimal stub if langgraph isn't available
    class _FakeAgent:
        def invoke(self, state):
            return {**state, **supervisor_node(state)}
    supervisor_agent = _FakeAgent()
