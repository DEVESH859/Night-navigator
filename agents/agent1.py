import os
import json
import time
import requests
from typing import TypedDict, Optional
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut, GeocoderServiceError
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
import dotenv

import threading

dotenv.load_dotenv()

import requests as _requests

# ── Per-model timeout wrapper ──────────────────────────────────────────────────
_GROQ_TIMEOUT_S = 12   # kill a stalled Groq call after 12 s

def _make_groq_llm(model: str):
    try:
        return ChatGroq(
            model=model,
            api_key=os.getenv("GROQ_API_KEY", ""),
            request_timeout=_GROQ_TIMEOUT_S,  # langchain-groq kwarg
        )
    except TypeError:
        # older langchain-groq doesn't accept request_timeout
        try:
            return ChatGroq(model=model, api_key=os.getenv("GROQ_API_KEY", ""))
        except Exception:
            return None
    except Exception:
        return None


def _invoke_with_timeout(llm_instance, prompt: str, timeout: int = _GROQ_TIMEOUT_S) -> str:
    """
    Run llm_instance.invoke() in a daemon thread; return content or ''
    on timeout / error — never raises, never returns None.
    """
    result_box: list = [None]
    exc_box:    list = [None]

    def _run():
        try:
            resp = llm_instance.invoke([SystemMessage(content=prompt)])
            result_box[0] = getattr(resp, "content", None) or ""
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        print(f"[LLM] invoke timed out after {timeout}s")
        return ""
    if exc_box[0]:
        print(f"[LLM] invoke raised: {exc_box[0]}")
        return ""
    return str(result_box[0] or "").strip()


def _call_openrouter(prompt: str, model: str = "mistralai/mistral-7b-instruct:free") -> str:
    """Fallback: OpenRouter via raw HTTP — tries multiple free models."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return ""
    try:
        resp = _requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://nightnavigator.app",
                "X-Title": "Night Navigator",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.1,
            },
            timeout=20,
        )
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return str(content or "").strip()
    except Exception as e:
        print(f"[OPENROUTER/{model}] Failed: {e}")
        return ""


# ── Ordered fallback chain ─────────────────────────────────────────────────────
_GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "llama-3.2-3b-preview",
    "llama-3.2-1b-preview",
]

_OPENROUTER_MODELS = [
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-2-9b-it:free",
    "qwen/qwen-2-7b-instruct:free",
]


def _call_llm_with_fallback(prompt: str) -> str:
    """
    Try Groq models → OpenRouter models in order.
    Each call is wrapped in a hard thread-level timeout.
    Returns '' only if ALL fail — never raises, never returns None.
    """
    # 1️⃣ Groq models
    for model in _GROQ_MODELS:
        llm_instance = _make_groq_llm(model)
        if not llm_instance:
            continue
        text = _invoke_with_timeout(llm_instance, prompt, timeout=_GROQ_TIMEOUT_S)
        if text:
            print(f"[LLM] ✓ Groq/{model}")
            return text
        print(f"[LLM] ✗ Groq/{model} — empty or timeout")

    # 2️⃣ OpenRouter models
    for or_model in _OPENROUTER_MODELS:
        print(f"[LLM] Trying OpenRouter/{or_model} …")
        text = _call_openrouter(prompt, model=or_model)
        if text:
            print(f"[LLM] ✓ OpenRouter/{or_model}")
            return text
        print(f"[LLM] ✗ OpenRouter/{or_model} — empty")

    print("[LLM] All providers failed — returning ''")
    return ""


# Keep a module-level llm for any legacy references
llm = _make_groq_llm("llama-3.1-8b-instant")

# FIX 1: Unique, compliant user_agent string per Nominatim ToS
geolocator = Nominatim(user_agent="night_navigator_app_v1", timeout=10)

# FIX 3: Hardcoded fallback coordinates for major Indian cities
CITY_FALLBACKS = {
    "bangalore":  (12.9716, 77.5946),
    "bengaluru":  (12.9716, 77.5946),
    "mysore":     (12.2958, 76.6394),
    "mumbai":     (19.0760, 72.8777),
    "delhi":      (28.7041, 77.1025),
    "new delhi":  (28.6139, 77.2090),
    "chennai":    (13.0827, 80.2707),
    "hyderabad":  (17.3850, 78.4867),
    "kolkata":    (22.5726, 88.3639),
    "pune":       (18.5204, 73.8567),
    "ahmedabad":  (23.0225, 72.5714),
    "jaipur":     (26.9124, 75.7873),
}

# FIX 4: Central landmark overrides for bare city names
CITY_LANDMARK_MAP = {
    "bangalore":  "MG Road, Bangalore, India",
    "bengaluru":  "MG Road, Bengaluru, India",
    "mysore":     "Mysore Palace, Mysore, India",
    "mumbai":     "Chhatrapati Shivaji Terminus, Mumbai, India",
    "delhi":      "Connaught Place, New Delhi, India",
    "new delhi":  "Connaught Place, New Delhi, India",
    "chennai":    "Marina Beach, Chennai, India",
    "hyderabad":  "Charminar, Hyderabad, India",
    "kolkata":    "Howrah Bridge, Kolkata, India",
    "pune":       "Pune Station, Pune, India",
}


def _build_geocode_query(location: str) -> str:
    """
    FIX 4: Convert bare city names into a specific, geocodeable landmark.
    If it's already a specific neighborhood or road, append Bangalore context.
    """
    normalized = location.strip().lower()
    if normalized in CITY_LANDMARK_MAP:
        return CITY_LANDMARK_MAP[normalized]
    if "india" not in normalized and "bangalore" not in normalized and "bengaluru" not in normalized:
        return f"{location}, Bangalore, India"
    return location


def _geocode_with_fallback(location: str) -> Optional[list]:
    """
    FIX 2 + 3: Robust geocoding with exception handling and hardcoded city fallbacks.
    Returns [lat, lon] or None.
    """
    query = _build_geocode_query(location)
    try:
        time.sleep(1.1)  # Nominatim requires >= 1s between requests
        loc = geolocator.geocode(query)
        if loc:
            return [loc.latitude, loc.longitude]
    except (GeocoderUnavailable, GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"[GEOCODE] Nominatim service error for '{query}': {e}")
    except Exception as e:
        print(f"[GEOCODE] Unexpected error for '{query}': {e}")

    # FIX 3: Fall back to hardcoded dict
    normalized = location.strip().lower()
    for key, coords in CITY_FALLBACKS.items():
        if key in normalized or normalized in key:
            print(f"[GEOCODE] Fallback coords for '{location}' -> {coords}")
            return list(coords)

    return None


class RouteState(TypedDict):
    query: str
    origin_name: Optional[str]
    dest_name: Optional[str]
    mode: Optional[str]
    origin_coords: Optional[list]
    dest_coords: Optional[list]
    api_response: Optional[dict]
    summary: Optional[str]
    error: Optional[str]
    crime_report: Optional[str]
    hospital_report: Optional[str]
    explanation: Optional[str]


# ── Bangalore neighborhood list for regex fallback ─────────────────────────
KNOWN_AREAS = [
    "indiranagar", "koramangala", "whitefield", "hebbal", "jayanagar",
    "mg road", "electronic city", "yeshwanthpur", "marathahalli", "btm layout",
    "hsr layout", "jp nagar", "rajajinagar", "malleshwaram", "yelahanka",
    "bellandur", "sarjapur", "kr puram", "silk board", "richmond road",
    "brigade road", "church street", "ulsoor", "shivajinagar", "majestic",
    "banashankari", "basavanagudi", "vijayanagar", "nagarbhavi", "kengeri",
]

def _regex_parse_fallback(query: str) -> dict:
    """
    Pure string matching fallback when all LLMs fail.
    Finds known Bangalore area names in the query text.
    """
    import re
    q_lower = query.lower()

    found = []
    for area in KNOWN_AREAS:
        if area in q_lower:
            found.append(area.title() + ", Bangalore")

    # Detect mode from keywords
    mode = "auto"
    if any(w in q_lower for w in ["night", "dark", "safe", "safety"]):
        mode = "night"
    elif any(w in q_lower for w in ["day", "morning", "afternoon"]):
        mode = "day"

    if len(found) >= 2:
        print(f"[REGEX FALLBACK] Found: {found[0]} → {found[1]}")
        return {"origin_name": found[0], "dest_name": found[1], "mode": mode}
    elif len(found) == 1:
        return {"error": f"Found only one location: '{found[0]}'. Please specify both origin and destination."}
    else:
        return {"error": "Could not find any Bangalore locations in your query. Try: 'Indiranagar to Koramangala'"}






def parse_intent(state: RouteState):
    query = state["query"]
    prompt = f"""You are a precise route intent extraction AI for Indian cities.

Your job:
1. Extract the ORIGIN and DESTINATION from the user text.
2. IMPORTANT: If the origin or destination is only a city name, convert it to a well-known central landmark:
   - "Bangalore" or "Bengaluru" -> "MG Road, Bangalore"
   - "Mysore" -> "Mysore Palace, Mysore"
   - "Mumbai" -> "Chhatrapati Shivaji Terminus, Mumbai"
   - "Delhi" or "New Delhi" -> "Connaught Place, New Delhi"
   - "Chennai" -> "Marina Beach, Chennai"
   - "Hyderabad" -> "Charminar, Hyderabad"
3. If the location is already a specific area (e.g. "Indiranagar", "Koramangala", "MG Road"),
   keep it but append ", Bangalore" if no city is given.
4. Detect mode: night/safety/dark -> "night", day/morning -> "day", else -> "auto".
5. Output ONLY valid JSON. No markdown, no explanation, no extra text.

Format exactly: {{"origin": "Landmark, City", "destination": "Landmark, City", "mode": "day|night|auto"}}

User text: {query}"""

    try:
        # ── Use multi-LLM fallback ─────────────────────────────────────────
        text = _call_llm_with_fallback(prompt)

        # ── Guard: empty response ──────────────────────────────────────────
        if not text:
            # Last resort: regex parse directly from query
            print("[PARSE] All LLMs failed — attempting regex fallback")
            return _regex_parse_fallback(query)

        # ── Strip markdown fences if present ──────────────────────────────
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else parts[0]
            if text.strip().startswith("json"):
                text = text.strip()[4:]
        text = text.strip()

        # ── Parse JSON ────────────────────────────────────────────────────
        data   = json.loads(text)
        origin = str(data.get("origin") or "").strip()
        dest   = str(data.get("destination") or "").strip()
        mode   = str(data.get("mode") or "auto").strip()

        if not origin or not dest:
            return _regex_parse_fallback(query)

        return {"origin_name": origin, "dest_name": dest, "mode": mode}

    except json.JSONDecodeError:
        print(f"[PARSE] JSON decode failed on: {text!r}")
        return _regex_parse_fallback(query)
    except Exception as e:
        return {"error": f"Failed to parse intent: {str(e)}"}


def validate_locations(state: RouteState):
    if state.get("error"):
        return {}
    origin_coords = _geocode_with_fallback(state.get("origin_name", ""))
    dest_coords   = _geocode_with_fallback(state.get("dest_name", ""))
    if not origin_coords:
        return {"error": f"Could not geocode origin: '{state['origin_name']}'. Try a more specific name."}
    if not dest_coords:
        return {"error": f"Could not geocode destination: '{state['dest_name']}'. Try a more specific name."}
    return {"origin_coords": origin_coords, "dest_coords": dest_coords}


def call_route_api(state: RouteState):
    if state.get("error"):
        return {}
    payload = {
        "origin":      state["origin_coords"],
        "destination": state["dest_coords"],
        "mode":        state.get("mode") or "auto"
    }

    # Try multiple ports in order — whichever uvicorn is alive wins
    CANDIDATE_PORTS = [8001, 8000, 8002]
    last_error = ""

    for port in CANDIDATE_PORTS:
        url = f"http://127.0.0.1:{port}/route"
        try:
            print(f"[ROUTE] Trying {url} …")
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            print(f"[ROUTE] ✓ Success on port {port}")
            return {"api_response": response.json()}
        except requests.exceptions.ConnectionError:
            print(f"[ROUTE] ✗ Port {port} — connection refused")
            last_error = f"Port {port} offline"
            continue
        except requests.exceptions.Timeout:
            print(f"[ROUTE] ✗ Port {port} — timed out (60s)")
            last_error = f"Port {port} timed out"
            continue
        except requests.exceptions.HTTPError as e:
            # Server responded but with an error — propagate immediately
            try:
                detail = response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            return {"error": f"Route API error: {detail}"}
        except Exception as e:
            last_error = str(e)
            continue

    return {"error": f"Route API unreachable on all ports ({', '.join(str(p) for p in CANDIDATE_PORTS)}). Last error: {last_error}"}


def generate_summary(state: RouteState):
    """
    Produces a comprehensive answer covering route + crime + hospitals + explanation.
    crime_report, hospital_report, explanation are injected externally by the orchestrator.

    """
    if state.get("error"):
        return {"summary": f"Could not complete request: {state['error']}"}

    data        = state.get("api_response") or {}
    dist        = data.get("distance_m", 0) / 1000
    comp        = data.get("comparison", {})
    safety_gain = comp.get("safety_gain_pct", 0)
    mode        = data.get("mode_used", "auto").upper()
    avg_safety  = data.get("avg_safety_score", 0) * 100
    avg_risk    = data.get("avg_incident_risk", 0) * 100

    # Enriched sections injected by orchestrator after running agent2/agent3
    crime_report    = state.get("crime_report")
    hospital_report = state.get("hospital_report")
    explanation     = state.get("explanation")

    extra_sections = ""
    if crime_report:
        extra_sections += f"\n\nCRIME INTELLIGENCE (include as a clearly labelled section):\n{crime_report}"
    if hospital_report:
        extra_sections += f"\n\nHOSPITAL / MEDICAL FACILITIES (include as a clearly labelled section):\n{hospital_report}"
    if explanation:
        extra_sections += f"\n\nROUTE SAFETY EXPLANATION (include as a clearly labelled section):\n{explanation}"

    query = state.get("query", "")
    prompt = f"""You are Night Navigator's expert safety assistant for Bangalore.

The user asked: "{query}"
(Parsed as route from **{state.get('origin_name','?')}** to **{state.get('dest_name','?')}**)

Route facts:
- Distance: {dist:.1f} km | Mode: {mode}
- Guardian safety score: {avg_safety:.0f}% | Incident risk: {avg_risk:.0f}%
- Safety gain vs shortest path: {safety_gain:+.1f}%
{extra_sections}

Your task:
1. Start with a 2-sentence route summary (distance, mode, safety score).
2. Answer any extra questions or specific constraints the user asked in their query (e.g. if they asked about restaurants, hospitals, specific POIs, or context, use your general knowledge and the route facts to help them).
3. If crime data is provided, add section "🔍 Crime Report" with key findings.
4. If hospital data is provided, add section "🏥 Medical Facilities" with list + nearest hospital.
5. If safety explanation is provided, add section "🛡️ Why This Route Is Safe".
6. End with a short reassuring closing line.

Use emoji + bold text as section headers. Do NOT use # markdown headers.
Keep total response under 350 words. Be warm, factual, and helpful."""

    try:
        summary_text = _call_llm_with_fallback(prompt)
        if summary_text:
            return {"summary": summary_text}
        raise ValueError("empty")
    except Exception:
        parts = [
            f"🗺️ Your Guardian route from {state.get('origin_name','?')} to "
            f"{state.get('dest_name','?')} is {dist:.1f} km "
            f"({avg_safety:.0f}% safety score, {safety_gain:+.1f}% safer than shortest). Stay safe! 🛡️"
        ]
        if crime_report:
            parts.append(f"\n🔍 **Crime Report:**\n{crime_report}")
        if hospital_report:
            parts.append(f"\n🏥 **Medical Facilities:**\n{hospital_report}")
        if explanation:
            parts.append(f"\n🛡️ **Why This Route Is Safe:**\n{explanation}")
        return {"summary": "\n".join(parts)}


workflow = StateGraph(RouteState)
workflow.add_node("parse_intent",       parse_intent)
workflow.add_node("validate_locations", validate_locations)
workflow.add_node("call_route_api",     call_route_api)
workflow.add_node("generate_summary",   generate_summary)
workflow.set_entry_point("parse_intent")
workflow.add_edge("parse_intent",       "validate_locations")
workflow.add_edge("validate_locations", "call_route_api")
workflow.add_edge("call_route_api",     "generate_summary")
workflow.add_edge("generate_summary",   END)

route_agent = workflow.compile()
