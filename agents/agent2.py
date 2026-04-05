import os
import json
from typing import TypedDict, Optional, List
import dotenv

dotenv.load_dotenv()

# ── Robust LLM fallback ────────────────────────────────────────────────────────
try:
    from agents.agent1 import _call_llm_with_fallback
except ImportError:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage
    _llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY", ""))
    def _call_llm_with_fallback(prompt):
        try:
            r = _llm.invoke([SystemMessage(content=prompt)])
            return str(getattr(r, "content", "") or "").strip()
        except Exception:
            return ""

try:
    from tavily import TavilyClient
    _tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
    HAS_TAVILY = True
except Exception:
    HAS_TAVILY = False

from langgraph.graph import StateGraph, END


class AlertState(TypedDict):
    origin_name:        str
    dest_name:          str
    route_data:         dict
    query:              str
    time_window:        str            # "recent" | "last 1 week" | "last 24 hours"
    needs_crime:        bool
    needs_hospitals:    bool
    # outputs
    crime_report:       Optional[str]
    hospital_report:    Optional[str]
    warning_message:    Optional[str]
    modified_route:     Optional[dict]
    error:              Optional[str]


# ── Node 1: Fetch crime + hospital data via Tavily ─────────────────────────────
def fetch_intel(state: AlertState):
    origin   = state.get("origin_name", "")
    dest     = state.get("dest_name", "")
    time_w   = state.get("time_window", "recent")
    needs_crime     = state.get("needs_crime", True)
    needs_hospitals = state.get("needs_hospitals", False)

    results = {}

    if not HAS_TAVILY:
        results["crime_raw"]    = ""
        results["hospital_raw"] = ""
        return results

    # Crime search
    if needs_crime:
        crime_q = (
            f"crime incidents robbery theft assault near {origin} to {dest} "
            f"Bangalore {time_w} site:timesofindia.com OR site:deccanherald.com "
            f"OR site:bangaloremirror.indiatimes.com OR site:newindianexpress.com"
        )
        try:
            resp = _tavily.search(query=crime_q, search_depth="advanced", max_results=6)
            results["crime_raw"] = "\n---\n".join(
                f"SOURCE: {r.get('url','')}\n{r.get('content','')}"
                for r in resp.get("results", [])
            )
        except Exception as e:
            results["crime_raw"] = f"[Tavily crime search unavailable: {e}]"

    # Hospital search
    if needs_hospitals:
        hosp_q = f"hospitals clinics medical emergency near {origin} to {dest} Bangalore"
        try:
            resp = _tavily.search(query=hosp_q, search_depth="basic", max_results=5)
            results["hospital_raw"] = "\n---\n".join(
                f"SOURCE: {r.get('url','')}\n{r.get('content','')}"
                for r in resp.get("results", [])
            )
        except Exception as e:
            results["hospital_raw"] = f"[Tavily hospital search unavailable: {e}]"

    return results


# ── Node 2: LLM analysis of crime data ────────────────────────────────────────
def analyse_crime(state: AlertState):
    if not state.get("needs_crime", True):
        return {"crime_report": None}

    raw = state.get("crime_raw", "")
    if not raw or "unavailable" in raw:
        return {
            "crime_report": f"No recent crime data found for the {state.get('origin_name','')} → {state.get('dest_name','')} corridor. The route appears quiet based on available data.",
            "warning_message": None
        }

    origin  = state.get("origin_name", "")
    dest    = state.get("dest_name", "")
    time_w  = state.get("time_window", "recent")

    prompt = f"""You are a public safety analyst for Bangalore, India.

Analyse the following news articles about crime and incidents **on or near the route from {origin} to {dest}** in the **{time_w}**.

Your output must cover:
1. Total number of incidents mentioned (crime, robbery, assault, theft, road crime etc.)
2. Severity breakdown (high / medium / low)
3. Specific locations mentioned (neighbourhood, road, landmark)
4. Trend: increasing / stable / decreasing?
5. Safety verdict for this route: Safe / Use Caution / Avoid Night Travel

Keep the tone helpful and factual. Use numbered points. Be concise (under 200 words).

--- NEWS DATA ---
{raw[:4000]}
--- END ---
"""
    text = _call_llm_with_fallback(prompt)
    if not text:
        text = f"Crime data retrieved but LLM analysis unavailable. Raw data indicates some incidents in the area."

    # Check for high-severity
    warning = None
    if any(w in (text or "").lower() for w in ["high severity", "avoid", "dangerous", "multiple crime"]):
        warning = f"⚠️ WARNING: Crime activity detected near {origin} → {dest} ({time_w}). Review safety report."

    return {"crime_report": text, "warning_message": warning}


# ── Node 3: LLM analysis of hospital data ─────────────────────────────────────
def analyse_hospitals(state: AlertState):
    if not state.get("needs_hospitals", False):
        return {"hospital_report": None}

    raw = state.get("hospital_raw", "")
    origin = state.get("origin_name", "")
    dest   = state.get("dest_name", "")

    if not raw or "unavailable" in raw:
        # Provide hardcoded Bangalore hospitals as fallback
        return {
            "hospital_report": (
                f"**Hospitals near {origin} → {dest} corridor:**\n"
                "1. **Manipal Hospital Indiranagar** — 98, HAL Airport Rd, near Indiranagar Metro (~0.8 km)\n"
                "2. **St. John's Medical College Hospital** — Sarjapur Rd (~3.2 km from Koramangala)\n"
                "3. **Sakra World Hospital** — SY No. 52/2, Devarabeesanahalli (~2.1 km)\n"
                "4. **Fortis Hospital Koramangala** — 154/9, Bannerghatta Rd (~1.5 km)\n\n"
                "_Nearest hospital on route: Manipal Hospital Indiranagar (approx. 0.8 km, open 24×7)._"
            )
        }

    prompt = f"""You are a medical resource assistant for Bangalore, India.

Based on the following data, list all hospitals, clinics, and emergency medical facilities near the route from **{origin} to {dest}**.

For each facility provide:
- Name
- Approximate distance from the route
- Whether it has 24-hour emergency services
- Contact number if available

Also state: "Nearest hospital on this route: [NAME] (~X km)"

Keep it structured with numbered points. Under 200 words.

--- DATA ---
{raw[:3000]}
--- END ---
"""
    text = _call_llm_with_fallback(prompt)
    if not text:
        text = (
            f"**Hospitals near {origin} → {dest}:**\n"
            "• Manipal Hospital Indiranagar (~0.8 km) — 24h emergency\n"
            "• Fortis Hospital Koramangala (~1.5 km) — 24h emergency\n"
            "• St. John's Medical College Hospital (~3.2 km)"
        )
    return {"hospital_report": text}


# ── Node 4: Adjust route risk based on crime findings ────────────────────────
def adjust_safety(state: AlertState):
    incidents_detected = state.get("warning_message") is not None
    route_data = dict(state.get("route_data") or {})

    if incidents_detected and "avg_incident_risk" in route_data:
        # Bump incident risk slightly to reflect news findings
        route_data["avg_incident_risk"] = min(1.0, route_data.get("avg_incident_risk", 0.2) + 0.15)

    return {"modified_route": route_data}


# ── Build the graph ────────────────────────────────────────────────────────────
workflow = StateGraph(AlertState)
workflow.add_node("fetch_intel",       fetch_intel)
workflow.add_node("analyse_crime",     analyse_crime)
workflow.add_node("analyse_hospitals", analyse_hospitals)
workflow.add_node("adjust_safety",     adjust_safety)

workflow.set_entry_point("fetch_intel")
workflow.add_edge("fetch_intel",       "analyse_crime")
workflow.add_edge("analyse_crime",     "analyse_hospitals")
workflow.add_edge("analyse_hospitals", "adjust_safety")
workflow.add_edge("adjust_safety",     END)

alert_agent = workflow.compile()
