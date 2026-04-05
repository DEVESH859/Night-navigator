"""
api/main.py
===========
Night Navigator — FastAPI Routing & Safety API

Endpoints:
  GET  /health          – liveness check + graph stats
  POST /route           – dual-path routing (shortest vs safest)
  GET  /safety-map      – full edge GeoJSON with safety attributes
  GET  /traffic-scores  – area-level traffic safety scores
  GET  /diagnostics     – safety score distribution stats

Run from project root:
  python -m uvicorn api.main:app --host 0.0.0.0 --port 8001
"""

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — IMPORTS AND PATHS
# ──────────────────────────────────────────────────────────────────────────────

import os
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from scipy import stats as scipy_stats

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.agents_router import agents_router

warnings.filterwarnings("ignore")

_BASE        = Path(__file__).resolve().parent.parent
DATA_DIR     = _BASE / "data"
FRONTEND_DIR = _BASE / "frontend"
GRAPH_FILE   = DATA_DIR / "bangalore_graph.graphml"
EDGES_FILE   = DATA_DIR / "edges_with_safety.geojson"
TRAFFIC_CSV  = DATA_DIR / "traffic_area_scores.csv"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — APP SETUP AND CORS
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Night Navigator API",
    description="Safe routing API for Bangalore using OSM + ML safety scores",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agents_router, prefix="/agent", tags=["Agents"])

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — GLOBAL STATE
# ──────────────────────────────────────────────────────────────────────────────

G          = None   # OSMnx graph
edges_df   = None   # GeoDataFrame of enriched edges
traffic_df = None   # traffic area scores DataFrame
diag_cache = None   # precomputed diagnostics dict

# Routing weights
ALPHA      = 0.50   # distance weight
BETA       = 0.50   # safety weight
MAX_LENGTH = 1.0    # set during startup

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — STARTUP EVENT
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global G, edges_df, traffic_df, diag_cache, MAX_LENGTH

    print("[STARTUP] Loading graph …")
    G = ox.load_graphml(str(GRAPH_FILE))
    print(f"[OK] Nodes={G.number_of_nodes():,}  Edges={G.number_of_edges():,}")

    print("[STARTUP] Loading safety edges …")
    edges_df = gpd.read_file(str(EDGES_FILE))
    print(f"[OK] Safety edges: {len(edges_df):,}")

    # ── Assign safety scores to graph edges ──────────────────────────────────
    print("[STARTUP] Building safety lookup …")
    safety_lookup: dict = {}
    for _, row in edges_df.iterrows():
        key = (int(row["u"]), int(row["v"]))
        safety_lookup[key] = {
            "safety_score":       float(row.get("safety_score",       0.30)),
            "safety_score_night": float(row.get("safety_score_night", 0.25)),
            "activity_composite": float(row.get("activity_composite", 0.30)),
            "incident_risk":      float(row.get("incident_risk",      0.50)),
            "police_bonus":       float(row.get("police_bonus",       0.00)),
            "edge_name":          str(row.get("name", "Unnamed Road")),
        }

    for u, v, key, data in G.edges(keys=True, data=True):
        info = safety_lookup.get((u, v)) or safety_lookup.get((v, u)) or {}
        data["safety_score"]       = info.get("safety_score",       0.30)
        data["safety_score_night"] = info.get("safety_score_night", 0.25)
        data["activity_composite"] = info.get("activity_composite", 0.30)
        data["incident_risk"]      = info.get("incident_risk",      0.50)
        data["police_bonus"]       = info.get("police_bonus",       0.00)
        data["edge_name"]          = info.get("edge_name",          "Unnamed Road")

    # ── Compute cost function ─────────────────────────────────────────────────
    print("[STARTUP] Computing edge costs …")
    all_lengths = [d.get("length", 1) for _, _, d in G.edges(data=True)]
    MAX_LENGTH  = max(all_lengths) if all_lengths else 1.0

    for u, v, key, data in G.edges(keys=True, data=True):
        length = data.get("length", 1)
        pb     = data.get("police_bonus",       0.00)
        act    = data.get("activity_composite", 0.30)
        ir     = data.get("incident_risk",      0.50)

        for mode_label, score_col in [
            ("day",   "safety_score"),
            ("night", "safety_score_night"),
        ]:
            base_safety = data.get(score_col, 0.30)
            cost = ALPHA * (length / MAX_LENGTH) + BETA * (1.0 - base_safety)
            cost = max(cost - pb, 0.01)
            data[f"safe_cost_{mode_label}"] = cost

    print("[STARTUP] Graph ready.")

    # ── Load traffic scores ───────────────────────────────────────────────────
    if TRAFFIC_CSV.exists():
        traffic_df = pd.read_csv(TRAFFIC_CSV)
        print(f"[OK] Traffic scores: {len(traffic_df)} areas")
    else:
        traffic_df = pd.DataFrame()
        print("[WARN] Traffic CSV not found.")

    # ── Precompute diagnostics ────────────────────────────────────────────────
    diag_cache = _compute_diagnostics(edges_df)
    print("[STARTUP] Diagnostics cached.")
    print("[STARTUP] API ready.\n")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — INTERNAL HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _auto_mode() -> str:
    """Return 'night' between 20:00–05:59, else 'day'."""
    hour = datetime.now().hour
    return "night" if (hour < 6 or hour >= 20) else "day"


def _get_route_stats(path_nodes: list, mode: str) -> dict:
    """Aggregate distance, avg safety, and avg incident risk along a path."""
    total_dist    = 0.0
    safety_vals   = []
    incident_vals = []
    score_col = "safety_score_night" if mode == "night" else "safety_score"

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if v not in G[u]:
            continue
        edge_data = min(G[u][v].values(), key=lambda d: d.get("length", 1))
        total_dist    += edge_data.get("length", 0)
        safety_vals.append(edge_data.get(score_col, 0.30))
        incident_vals.append(edge_data.get("incident_risk", 0.50))

    return {
        "distance_m":   round(total_dist, 1),
        "avg_safety":   round(float(np.mean(safety_vals))   if safety_vals   else 0.0, 4),
        "avg_incident": round(float(np.mean(incident_vals)) if incident_vals else 0.0, 4),
    }


def _get_path_coords(path_nodes: list) -> list:
    """Convert node IDs to [[lat, lon], …] for map display."""
    return [
        [G.nodes[n]["y"], G.nodes[n]["x"]]
        for n in path_nodes
        if "y" in G.nodes[n] and "x" in G.nodes[n]
    ]


def _shortest_path(origin_latlon: tuple, dest_latlon: tuple) -> list:
    """Dijkstra shortest path by road distance."""
    orig = ox.nearest_nodes(G, origin_latlon[1], origin_latlon[0])
    dest = ox.nearest_nodes(G, dest_latlon[1],   dest_latlon[0])
    return nx.shortest_path(G, orig, dest, weight="length")


def _safest_path(origin_latlon: tuple, dest_latlon: tuple, mode: str) -> list:
    """
    A* safest path using precomputed safe_cost_day / safe_cost_night.
    Geographic Euclidean heuristic keeps search directional.
    Falls back to shortest path if A* fails.
    """
    orig      = ox.nearest_nodes(G, origin_latlon[1], origin_latlon[0])
    dest_node = ox.nearest_nodes(G, dest_latlon[1],   dest_latlon[0])
    cost_col  = f"safe_cost_{mode}"

    dest_lat = G.nodes[dest_node]["y"]
    dest_lon = G.nodes[dest_node]["x"]

    def heuristic(u, v):
        u_lat    = G.nodes[u]["y"]
        u_lon    = G.nodes[u]["x"]
        dist_deg = ((u_lat - dest_lat) ** 2 + (u_lon - dest_lon) ** 2) ** 0.5
        return dist_deg * 111.0   # 1 degree ≈ 111 km → converts to metre scale

    try:
        return nx.astar_path(G, orig, dest_node,
                             heuristic=heuristic,
                             weight=cost_col)
    except Exception:
        return nx.shortest_path(G, orig, dest_node, weight="length")


def _compute_diagnostics(edf: gpd.GeoDataFrame) -> dict:
    """
    Compute per-column statistics for safety scores:
    mean, std, skewness, and 5-bucket histogram.
    """
    result = {}
    for col in ["safety_score", "safety_score_night"]:
        if col not in edf.columns:
            continue
        s          = edf[col].dropna()
        skew       = float(scipy_stats.skew(s))
        bins       = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        counts, _  = np.histogram(s, bins=bins)
        result[col] = {
            "mean":     round(float(s.mean()), 4),
            "std":      round(float(s.std()),  4),
            "skewness": round(skew, 4),
            "buckets": {
                "0.0-0.2": int(counts[0]),
                "0.2-0.4": int(counts[1]),
                "0.4-0.6": int(counts[2]),
                "0.6-0.8": int(counts[3]),
                "0.8-1.0": int(counts[4]),
            },
        }
    return result

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — REQUEST MODELS (Pydantic)
# ──────────────────────────────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    origin:           list[float]              # [lat, lon]
    destination:      list[float]              # [lat, lon]
    mode:             Optional[str] = "auto"   # "day" | "night" | "auto"
    avoid_congestion: Optional[bool] = False
    waypoints:        Optional[list[list[float]]] = []  # [[lat,lon], ...] max 3

# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check: returns graph stats and current auto-detected mode."""
    mode = _auto_mode()
    return {
        "status":       "ok",
        "nodes":        G.number_of_nodes() if G else 0,
        "edges":        G.number_of_edges() if G else 0,
        "current_mode": mode,
        "current_hour": datetime.now().hour,
    }


# ── POST /route ───────────────────────────────────────────────────────────────

def build_route_response(req: RouteRequest):
    """
    Compute both the shortest-distance path and the safest A* path.
    Supports up to 3 intermediate waypoints — stitches per-leg paths.
    Returns coordinates for both paths plus a side-by-side comparison.
    """
    if G is None:
        raise HTTPException(503, "Graph not loaded yet")

    # Resolve mode
    mode = _auto_mode() if req.mode == "auto" else req.mode
    if mode not in ("day", "night"):
        mode = _auto_mode()

    origin_latlon = tuple(req.origin)       # (lat, lon)
    dest_latlon   = tuple(req.destination)  # (lat, lon)

    # Build ordered stop list: origin → [waypoints...] → destination
    waypoints = [list(w) for w in (req.waypoints or [])[:3]]  # cap at 3
    stops = [list(req.origin)] + waypoints + [list(req.destination)]

    def stitch_path(fn):
        """Compute a path through all stops using fn(origin, dest, *args)."""
        full_path = []
        for i in range(len(stops) - 1):
            a, b = tuple(stops[i]), tuple(stops[i + 1])
            leg = fn(a, b) if fn == _shortest_path else fn(a, b, mode)
            if full_path:
                full_path.extend(leg[1:])   # avoid duplicating junction node
            else:
                full_path.extend(leg)
        return full_path

    try:
        path_short = stitch_path(_shortest_path)
        path_safe  = stitch_path(_safest_path)

        stats_short = _get_route_stats(path_short, mode)
        stats_safe  = _get_route_stats(path_safe,  mode)

        path_coords  = _get_path_coords(path_safe)
        short_coords = _get_path_coords(path_short)

        safety_gain = (
            (stats_safe["avg_safety"] - stats_short["avg_safety"])
            / max(stats_short["avg_safety"], 0.01) * 100
        )
        dist_overhead = (
            (stats_safe["distance_m"] - stats_short["distance_m"])
            / max(stats_short["distance_m"], 1) * 100
        )

        return {
            "mode_used":         mode,
            "path_coords":       path_coords,
            "short_coords":      short_coords,
            "distance_m":        stats_safe["distance_m"],
            "avg_safety_score":  stats_safe["avg_safety"],
            "avg_incident_risk": stats_safe["avg_incident"],
            "waypoint_count":    len(waypoints),
            "comparison": {
                "shortest": {
                    "distance_m":   stats_short["distance_m"],
                    "avg_safety":   stats_short["avg_safety"],
                    "avg_incident": stats_short["avg_incident"],
                },
                "safest": {
                    "distance_m":   stats_safe["distance_m"],
                    "avg_safety":   stats_safe["avg_safety"],
                    "avg_incident": stats_safe["avg_incident"],
                },
                "safety_gain_pct":       round(safety_gain,   2),
                "distance_overhead_pct": round(dist_overhead, 2),
            },
        }

    except Exception as e:
        raise HTTPException(500, f"Routing failed: {str(e)}")


@app.post("/route")
def route(req: RouteRequest):
    return build_route_response(req)


# ── GET /safety-map ───────────────────────────────────────────────────────────

@app.get("/safety-map")
def safety_map():
    """
    Stream all edges as GeoJSON with safety properties.
    Slims the payload to essential columns and rounds floats to 4 dp.
    """
    if edges_df is None:
        raise HTTPException(503, "Edges not loaded yet")

    keep_cols = [
        "geometry", "safety_score", "safety_score_night",
        "incident_risk", "activity_composite",
        "police_bonus", "road_importance", "name",
    ]
    available = [c for c in keep_cols if c in edges_df.columns]
    slim = edges_df[available].copy()

    if "name" in slim.columns:
        slim["name"] = slim["name"].fillna("Unnamed Road")

    for col in ["safety_score", "safety_score_night", "incident_risk",
                "activity_composite", "police_bonus", "road_importance"]:
        if col in slim.columns:
            slim[col] = slim[col].round(4)

    geojson_str = slim.to_json()

    return StreamingResponse(
        iter([geojson_str]),
        media_type="application/geo+json",
    )


# ── GET /traffic-scores ───────────────────────────────────────────────────────

@app.get("/traffic-scores")
def traffic_scores():
    """Return area-level traffic safety scores keyed by area name."""
    if traffic_df is None or traffic_df.empty:
        raise HTTPException(503, "Traffic data not loaded")

    result = {}
    for _, row in traffic_df.iterrows():
        area = str(row.get("Area Name", "Unknown"))
        result[area] = {
            "traffic_safety_score": round(float(row.get("traffic_safety_score", 0.5)), 4),
            "activity_score":       round(float(row.get("activity_score",       0.5)), 4),
            "incident_risk":        round(float(row.get("incident_risk",        0.5)), 4),
        }
    return result


# ── GET /diagnostics ──────────────────────────────────────────────────────────

@app.get("/diagnostics")
def diagnostics():
    """Return precomputed safety score distribution stats (mean, std, skewness, buckets)."""
    if diag_cache is None:
        raise HTTPException(503, "Diagnostics not ready")
    return diag_cache


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 — RUN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8001")),
        reload=False,   # use reload=True only for dev — slow with large graph
    )
