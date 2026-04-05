"""
safe_router.py
==============
Night Navigator — Safe Routing Engine

Loads a pre-built OSMnx road graph and safety-enriched edges,
assigns per-edge cost values (day & night), then compares the
shortest-distance path against the safest-path (A*) for two
representative Bangalore origin–destination pairs.

Usage:
    python routing/safe_router.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — IMPORTS AND PATHS
# ──────────────────────────────────────────────────────────────────────────────

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from datetime import datetime

warnings.filterwarnings("ignore")

_BASE      = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR   = os.path.abspath(os.path.join(_BASE, "data"))
GRAPH_FILE = os.path.join(DATA_DIR, "bangalore_graph.graphml")
EDGES_FILE = os.path.join(DATA_DIR, "edges_with_safety.geojson")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD GRAPH AND SAFETY EDGES
# ──────────────────────────────────────────────────────────────────────────────

print("[STEP 2] Loading graph …")
G = ox.load_graphml(GRAPH_FILE)
print(f"[OK]   Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")

print("[STEP 3] Loading safety edges …")
edges_safe = gpd.read_file(EDGES_FILE)
print(f"[OK]   Safety edges loaded: {len(edges_safe):,}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — BUILD SAFETY LOOKUP AND ASSIGN TO GRAPH
# ──────────────────────────────────────────────────────────────────────────────

print("[STEP 4] Building safety lookup and assigning to graph edges …")

safety_lookup: dict = {}
for _, row in edges_safe.iterrows():
    key = (int(row["u"]), int(row["v"]))
    safety_lookup[key] = {
        "safety_score":       float(row.get("safety_score",       0.30)),
        "safety_score_night": float(row.get("safety_score_night", 0.25)),
        "activity_composite": float(row.get("activity_composite", 0.30)),
        "incident_risk":      float(row.get("incident_risk",      0.50)),
        "police_bonus":       float(row.get("police_bonus",       0.00)),
        "edge_name":          str(row.get("name", "Unnamed Road")),
    }

matched = 0
for u, v, key, data in G.edges(keys=True, data=True):
    info = safety_lookup.get((u, v)) or safety_lookup.get((v, u)) or {}
    data["safety_score"]       = info.get("safety_score",       0.30)
    data["safety_score_night"] = info.get("safety_score_night", 0.25)
    data["activity_composite"] = info.get("activity_composite", 0.30)
    data["incident_risk"]      = info.get("incident_risk",      0.50)
    data["police_bonus"]       = info.get("police_bonus",       0.00)
    data["edge_name"]          = info.get("edge_name",          "Unnamed Road")
    if info:
        matched += 1

match_pct = 100 * matched / max(G.number_of_edges(), 1)
print(f"[OK]   Edges matched: {matched:,} / {G.number_of_edges():,} ({match_pct:.1f}%)")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — COMPUTE COST FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

print("[STEP 5] Computing edge cost function …")

all_lengths = [d.get("length", 1) for u, v, d in G.edges(data=True)]
max_length  = max(all_lengths) if all_lengths else 1.0

# Weights
alpha = 0.50   # distance contribution
beta  = 0.50   # safety contribution (primary goal)

for u, v, key, data in G.edges(keys=True, data=True):
    length = data.get("length", 1)
    act    = data.get("activity_composite", 0.30)
    ir     = data.get("incident_risk",      0.50)
    pb     = data.get("police_bonus",       0.00)

    for mode_label, score_col in [("day", "safety_score"), ("night", "safety_score_night")]:
        base_safety = data.get(score_col, 0.30 if mode_label == "day" else 0.25)

        cost = alpha * (length / max_length) + beta * (1.0 - base_safety)

       

        cost = max(cost, 0.01)
        data[f"safe_cost_{mode_label}"] = cost

cost_day_vals   = [d["safe_cost_day"]   for _, _, d in G.edges(data=True)]
cost_night_vals = [d["safe_cost_night"] for _, _, d in G.edges(data=True)]

print("[OK]   Cost function applied to all graph edges.")
print(f"       Mean safe_cost_day  : {np.mean(cost_day_vals):.4f}")
print(f"       Mean safe_cost_night: {np.mean(cost_night_vals):.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def auto_detect_mode() -> str:
    """Return 'night' between 20:00–05:59, else 'day'."""
    hour = datetime.now().hour
    return "night" if (hour < 6 or hour >= 20) else "day"


def get_route_stats(G, path_nodes: list, mode: str = "night") -> dict:
    """
    Walk path edges and collect distance, safety, and incident statistics.

    Parameters
    ----------
    G          : OSMnx MultiDiGraph with precomputed cost attributes
    path_nodes : ordered list of node IDs along the path
    mode       : 'day' | 'night'

    Returns
    -------
    dict with keys: distance_m, avg_safety, avg_incident, unique_roads
    """
    total_dist    = 0.0
    safety_vals   = []
    incident_vals = []
    names         = []
    score_col = "safety_score_night" if mode == "night" else "safety_score"

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        # Pick the parallel edge with minimum length
        edge_data = min(G[u][v].values(), key=lambda d: d.get("length", 1))
        total_dist    += edge_data.get("length", 0)
        safety_vals.append(edge_data.get(score_col, 0.30))
        incident_vals.append(edge_data.get("incident_risk", 0.50))
        names.append(edge_data.get("edge_name", "Unnamed"))

    return {
        "distance_m":   round(total_dist, 1),
        "avg_safety":   round(float(np.mean(safety_vals))   if safety_vals   else 0.0, 4),
        "avg_incident": round(float(np.mean(incident_vals)) if incident_vals else 0.0, 4),
        "unique_roads": len(set(n for n in names if n != "Unnamed Road")),
    }


def get_path_coords(G, path_nodes: list) -> list:
    """Convert a list of node IDs to [(lat, lon), …] for API / frontend use."""
    return [
        (G.nodes[n]["y"], G.nodes[n]["x"])
        for n in path_nodes
        if "y" in G.nodes[n] and "x" in G.nodes[n]
    ]


def get_shortest_path(G, origin_latlon: tuple, dest_latlon: tuple) -> list:
    """Return the shortest path (by road distance) between two lat/lon points."""
    orig = ox.nearest_nodes(G, origin_latlon[1], origin_latlon[0])
    dest = ox.nearest_nodes(G, dest_latlon[1],   dest_latlon[0])
    return nx.shortest_path(G, orig, dest, weight="length")


def get_safest_path(G, origin_latlon: tuple, dest_latlon: tuple,
                    mode: str = "night") -> list:
    """
    Return the safest path using precomputed safe_cost_day / safe_cost_night.
    Uses a geographic heuristic to keep A* on track toward the destination.
    Falls back to shortest path by distance if A* fails.
    """
    orig     = ox.nearest_nodes(G, origin_latlon[1], origin_latlon[0])
    dest     = ox.nearest_nodes(G, dest_latlon[1],   dest_latlon[0])
    cost_col = f"safe_cost_{mode}"

    # Heuristic: Euclidean distance in degrees (fast, keeps A* directional)
    dest_lat = G.nodes[dest]["y"]
    dest_lon = G.nodes[dest]["x"]

    def heuristic(u, v):
        u_lat = G.nodes[u]["y"]
        u_lon = G.nodes[u]["x"]
        dist_deg = ((u_lat - dest_lat)**2 + (u_lon - dest_lon)**2) ** 0.5
        return dist_deg * 111.0   # 1 degree ≈ 111 km, converts to metres scale  # tuned scaling factor

    try:
        return nx.astar_path(G, orig, dest,
                             heuristic=heuristic,
                             weight=cost_col)
    except Exception as e:
        print(f"[WARN] A* failed ({e}), falling back to shortest path.")
        return nx.shortest_path(G, orig, dest, weight="length")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — TEST WITH HARDCODED BANGALORE OD PAIRS
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL: Allow user to enter custom start/destination
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  ROUTING OPTIONS")
print("=" * 72)
print("Enter custom coordinates (lat,lon) or press Enter to use default pairs.")
custom_origin = input("Origin (lat,lon): ").strip()
custom_dest   = input("Destination (lat,lon): ").strip()

if custom_origin and custom_dest:
    try:
        lat1, lon1 = map(float, custom_origin.split(','))
        lat2, lon2 = map(float, custom_dest.split(','))
        OD_PAIRS = [{
            "name": f"Custom ({lat1:.4f},{lon1:.4f}) → ({lat2:.4f},{lon2:.4f})",
            "origin": (lat1, lon1),
            "dest":   (lat2, lon2),
        }]
        print(f"[INFO] Using custom route: {OD_PAIRS[0]['name']}")
    except ValueError:
        print("[WARN] Invalid coordinates. Falling back to default pairs.")
        OD_PAIRS = [
            {"name": "Indiranagar → Koramangala", "origin": (12.9784, 77.6408), "dest": (12.9352, 77.6245)},
            {"name": "Hebbal → Electronic City",   "origin": (13.0354, 77.5970), "dest": (12.8458, 77.6603)},
        ]
else:
    print("[INFO] Using default route pairs.")
    OD_PAIRS = [
        {"name": "Indiranagar → Koramangala", "origin": (12.9784, 77.6408), "dest": (12.9352, 77.6245)},
        {"name": "Hebbal → Electronic City",   "origin": (13.0354, 77.5970), "dest": (12.8458, 77.6603)},
    ]


    

mode = auto_detect_mode()
print(f"\n[INFO] Auto-detected mode: {mode.upper()}")
print("=" * 72)

results_pairs = []

for pair in OD_PAIRS:
    name   = pair["name"]
    origin = pair["origin"]
    dest   = pair["dest"]

    print(f"\n{'─' * 72}")
    print(f"  PAIR: {name}")
    print(f"{'─' * 72}")

    # ── Shortest path ─────────────────────────────────────────────────────────
    t0          = time.time()
    path_short  = get_shortest_path(G, origin, dest)
    t_short     = (time.time() - t0) * 1000

    # ── Safest path ───────────────────────────────────────────────────────────
    t0         = time.time()
    path_safe  = get_safest_path(G, origin, dest, mode=mode)
    t_safe     = (time.time() - t0) * 1000

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats_short = get_route_stats(G, path_short, mode=mode)
    stats_safe  = get_route_stats(G, path_safe,  mode=mode)

    safety_gain   = (
        (stats_safe["avg_safety"] - stats_short["avg_safety"])
        / max(stats_short["avg_safety"], 0.01) * 100
    )
    dist_overhead = (
        (stats_safe["distance_m"] - stats_short["distance_m"])
        / max(stats_short["distance_m"], 1) * 100
    )

    # ── Pretty table ──────────────────────────────────────────────────────────
    # Column widths
    w_metric = 26
    w_val    = 18

    header  = (f"  ╔{'═'*w_metric}╦{'═'*w_val}╦{'═'*w_val}╗")
    divider = (f"  ╠{'═'*w_metric}╬{'═'*w_val}╬{'═'*w_val}╣")
    footer  = (f"  ╚{'═'*w_metric}╩{'═'*w_val}╩{'═'*w_val}╝")

    def row(label, v_short, v_safe):
        return (
            f"  ║ {label:<{w_metric-2}} ║ {v_short:^{w_val-2}} ║ {v_safe:^{w_val-2}} ║"
        )

    print(header)
    print(row("Metric", "Shortest Path", "Safest Path"))
    print(divider)
    print(row("Distance (m)",
              f"{stats_short['distance_m']:,.1f}",
              f"{stats_safe['distance_m']:,.1f}"))
    print(row("Avg Safety Score",
              f"{stats_short['avg_safety']:.4f}",
              f"{stats_safe['avg_safety']:.4f}"))
    print(row("Avg Incident Risk",
              f"{stats_short['avg_incident']:.4f}",
              f"{stats_safe['avg_incident']:.4f}"))
    print(row("Unique Named Roads",
              str(stats_short["unique_roads"]),
              str(stats_safe["unique_roads"])))
    print(row("Compute Time (ms)",
              f"{t_short:.1f}",
              f"{t_safe:.1f}"))
    print(row("Safety Gain (%)",
              "—",
              f"{safety_gain:+.2f}%"))
    print(row("Distance Overhead (%)",
              "—",
              f"{dist_overhead:+.2f}%"))
    print(footer)

    print(f"  Shortest path: {len(path_short)} nodes")
    print(f"  Safest path:   {len(path_safe)} nodes")

    results_pairs.append({
        "name":    name,
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
    })

# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — EXPORT ROUTE COMPARISON AS JSON
# ──────────────────────────────────────────────────────────────────────────────

output_payload = {
    "mode":  mode,
    "pairs": results_pairs,
}

output_path = os.path.join(DATA_DIR, "routing_test_results.json")
with open(output_path, "w", encoding="utf-8") as fh:
    json.dump(output_payload, fh, indent=2, ensure_ascii=False)

print("\n[DONE] routing/safe_router.py complete.")
print(f"       Results saved to data/routing_test_results.json")
