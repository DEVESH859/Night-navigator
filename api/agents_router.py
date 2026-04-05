from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Import the compiled agents
from agents.agent1 import route_agent, generate_summary
from agents.agent2 import alert_agent
from agents.agent3 import explain_agent
from agents.supervisor import supervisor_agent

agents_router = APIRouter()


class RouteQueryReq(BaseModel):
    query: str


class ExplainReq(BaseModel):
    origin: List[float]
    destination: List[float]
    metrics: Optional[dict] = {}


# ── Helper: run agent3 (explain) ──────────────────────────────────────────────
def _run_agent3(route_metrics: dict) -> Optional[str]:
    try:
        state_in  = {"route_metrics": route_metrics}
        exp_state = explain_agent.invoke(state_in)
        return exp_state.get("explanation")
    except Exception as e:
        print(f"[AGENT3] Failed: {e}")
        return None


# ── /agent/route  (agent1 only) ───────────────────────────────────────────────
@agents_router.post("/route")
async def process_route(req: RouteQueryReq):
    """
    Parses natural language → extracts coords → returns safest route + AI summary.
    """
    state_in = {"query": req.query}
    try:
        final_state = route_agent.invoke(state_in)
        if final_state.get("error"):
            raise HTTPException(status_code=400, detail=final_state["error"])
        return {
            "parsed_origin":      final_state.get("origin_name"),
            "parsed_destination": final_state.get("dest_name"),
            "summary":            final_state.get("summary"),
            "route_data":         final_state.get("api_response"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /agent/route-with-alerts  (agent1 + agent2) ───────────────────────────────
@agents_router.post("/route-with-alerts")
async def process_route_with_alerts(req: RouteQueryReq, time_window: str = "recent"):
    """
    Finds safest route, then searches Tavily for real-time crime incidents.
    """
    state_route = {"query": req.query}
    try:
        route_state = route_agent.invoke(state_route)
        if route_state.get("error"):
            raise HTTPException(status_code=400, detail=route_state["error"])

        r_data = route_state.get("api_response", {})

        state_alert = {
            "query":        req.query,
            "origin_name":  route_state.get("origin_name", ""),
            "dest_name":    route_state.get("dest_name", ""),
            "route_data":   r_data,
            "time_window":  time_window,
            "needs_crime":  True,
            "needs_hospitals": False,
        }
        alert_state = alert_agent.invoke(state_alert)

        return {
            "parsed_origin":       route_state.get("origin_name"),
            "parsed_destination":  route_state.get("dest_name"),
            "summary":             route_state.get("summary"),
            "incident_warnings":   alert_state.get("warning_message"),
            "crime_report":        alert_state.get("crime_report"),
            "extracted_incidents": alert_state.get("extracted_incidents", []),
            "route_data":          alert_state.get("modified_route", r_data),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /agent/explain  (agent3 only) ─────────────────────────────────────────────
@agents_router.post("/explain")
async def process_explain(req: ExplainReq):
    """
    Explains why a given route is safe based on metrics and feature importance.
    """
    state_in = {"route_metrics": req.metrics}
    try:
        final_state = explain_agent.invoke(state_in)
        if final_state.get("error"):
            raise HTTPException(status_code=400, detail=final_state["error"])
        return {"explanation": final_state.get("explanation")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /agent/orchestrate  (supervisor → agent1 + agent2 + agent3 as needed) ────
@agents_router.post("/orchestrate")
async def orchestrate_query(req: RouteQueryReq):
    """
    Smart orchestration:
      1. Supervisor classifies intent (needs_route, needs_crime, needs_hospitals, needs_explanation)
      2. Agent1 always runs to get the route
      3. Agent2 runs if needs_crime OR needs_hospitals
      4. Agent3 runs if needs_explanation
      5. Results are merged into a single rich response via generate_summary
    """
    try:
        # ── Step 1: Supervisor decides what's needed ──────────────────────────
        sup_state = supervisor_agent.invoke({"query": req.query})
        action          = sup_state.get("action", "route")
        needs_crime     = sup_state.get("needs_crime", False)
        needs_hospitals = sup_state.get("needs_hospitals", False)
        needs_explain   = sup_state.get("needs_explanation", False)
        time_window     = sup_state.get("time_window", "recent")

        print(f"[ORCHESTRATE] action={action} crime={needs_crime} hospitals={needs_hospitals} explain={needs_explain} window={time_window}")

        # ── Step 2: Agent1 — always get the route ────────────────────────────
        route_state = route_agent.invoke({"query": req.query})
        if route_state.get("error"):
            raise HTTPException(status_code=400, detail=route_state["error"])

        r_data      = route_state.get("api_response", {})
        origin_name = route_state.get("origin_name", "")
        dest_name   = route_state.get("dest_name", "")

        # ── Step 3: Agent2 — crime + hospital intel ───────────────────────────
        crime_report    = None
        hospital_report = None
        incident_warnings = None
        modified_route  = r_data

        if needs_crime or needs_hospitals or action in ("route_alert", "full"):
            state_alert = {
                "query":           req.query,
                "origin_name":     origin_name,
                "dest_name":       dest_name,
                "route_data":      r_data,
                "time_window":     time_window,
                "needs_crime":     needs_crime or action in ("route_alert", "full"),
                "needs_hospitals": needs_hospitals,
            }
            alert_state     = alert_agent.invoke(state_alert)
            crime_report    = alert_state.get("crime_report")
            hospital_report = alert_state.get("hospital_report")
            incident_warnings = alert_state.get("warning_message")
            modified_route  = alert_state.get("modified_route", r_data)

        # ── Step 4: Agent3 — safety explanation ───────────────────────────────
        explanation = None
        if needs_explain or action in ("explain", "full"):
            metrics = {
                "distance_m":        r_data.get("distance_m", 0),
                "avg_safety_score":  r_data.get("avg_safety_score", 0),
                "avg_incident_risk": r_data.get("avg_incident_risk", 0),
            }
            explanation = _run_agent3(metrics)

        # ── Step 5: Build rich summary via generate_summary ───────────────────
        # Inject enriched context into the route_state for generate_summary
        enriched_state = {
            **route_state,
            "crime_report":    crime_report,
            "hospital_report": hospital_report,
            "explanation":     explanation,
        }
        summary_result = generate_summary(enriched_state)
        final_summary  = summary_result.get("summary", route_state.get("summary", ""))

        return {
            "parsed_origin":      origin_name,
            "parsed_destination": dest_name,
            "summary":            final_summary,
            "crime_report":       crime_report,
            "hospital_report":    hospital_report,
            "explanation":        explanation,
            "incident_warnings":  incident_warnings,
            "route_data":         modified_route,
            "agents_used":        {
                "agent1": True,
                "agent2_crime":    needs_crime or action in ("route_alert", "full"),
                "agent2_hospitals": needs_hospitals,
                "agent3":          needs_explain or action in ("explain", "full"),
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
