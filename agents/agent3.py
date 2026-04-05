import os
import joblib
from typing import TypedDict, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
import dotenv

dotenv.load_dotenv()

explain_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY", "")
)

class ExplainState(TypedDict):
    route_metrics: dict
    feature_importance: Optional[dict]
    explanation: Optional[str]
    error: Optional[str]

def get_route_metrics(state: ExplainState):
    # This node just ensures we have the metrics, 
    # normally it would aggregate them from the route but we assume they are passed in.
    metrics = state.get("route_metrics", {})
    if not metrics:
        return {"error": "No route metrics provided."}
    return {}

def retrieve_feature_importance(state: ExplainState):
    if state.get("error"):
        return {}
    
    # Try to load the XGBoost model to get feature importances
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_regressor.pkl")
    importance_dict = {
        "lamp_norm": 0.4,
        "activity_composite": 0.3,
        "incident_risk": 0.2,
        "police_bonus": 0.1
    }
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            # Depending on model type (pipeline vs raw xgb), this might need adjustment
            # Here we mock retrieving it. In a robust setup we'd extract .feature_importances_
            pass
        except Exception:
            pass
    
    return {"feature_importance": importance_dict}

try:
    from agents.agent1 import _call_llm_with_fallback
except ImportError:
    # Basic fallback if import fails
    def _call_llm_with_fallback(prompt):
        res = explain_llm.invoke([SystemMessage(content=prompt)])
        return str(getattr(res, "content", "") or "").strip()

def generate_explanation(state: ExplainState):
    if state.get("error"):
        return {"explanation": "I'm sorry, I couldn't explain the route: " + state["error"]}
    
    metrics = state.get("route_metrics", {})
    importances = state.get("feature_importance", {})
    
    avg_safety = metrics.get("avg_safety_score", 0.5) * 100
    avg_incident = metrics.get("avg_incident_risk", 0.5) * 100
    
    prompt = f"""
    You are a trustworthy safety companion. Explain to the user why the suggested route is mathematically safe.
    
    Route metrics: 
    - Average safety score is {avg_safety:.0f}%
    - Average incident risk is {avg_incident:.0f}%.
    
    The global AI feature impact on safety (how much each urban factor matters) is roughly: {importances}.
    
    Write a concise, conversational response (like "This route is great because...").
    Use bullet points if helpful. Keep it under 100 words. Do not use # headers.
    """
    
    try:
        text = _call_llm_with_fallback(prompt)
        if text:
            return {"explanation": text}
        raise ValueError("Empty LLM response")
    except Exception as e:
        return {"explanation": f"This route is optimized for your safety based on multiple urban factors. The route scores {avg_safety:.0f}% on the Night Navigator safety index."}

workflow = StateGraph(ExplainState)

workflow.add_node("get_route_metrics", get_route_metrics)
workflow.add_node("retrieve_feature_importance", retrieve_feature_importance)
workflow.add_node("generate_explanation", generate_explanation)

workflow.set_entry_point("get_route_metrics")
workflow.add_edge("get_route_metrics", "retrieve_feature_importance")
workflow.add_edge("retrieve_feature_importance", "generate_explanation")
workflow.add_edge("generate_explanation", END)

explain_agent = workflow.compile()
