"""
evaluation/evaluate.py
======================
Night Navigator — Routing & Safety Evaluation Pipeline

Steps:
  1. Load graph + assign safety scores
  2. Routing helpers (shortest / safest)
  3. 20 random OD-pair benchmark
  4. Aggregate metrics summary table
  5. Safety score distribution plots
  6. Feature ablation study
  7. Safety-gain vs distance-overhead scatter plot
  8. Save evaluation/results.json
  9. Final summary print

Run from project root:
  python evaluation/evaluate.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# STEP 0 — IMPORTS AND PATHS
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

_BASE      = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR   = os.path.abspath(os.path.join(_BASE, "data"))
EVAL_DIR   = os.path.abspath(os.path.join(_BASE, "evaluation"))
GRAPH_FILE = os.path.join(DATA_DIR, "bangalore_graph.graphml")
EDGES_FILE = os.path.join(DATA_DIR, "edges_with_safety.geojson")

os.makedirs(EVAL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD GRAPH AND ASSIGN SAFETY SCORES
# ──────────────────────────────────────────────────────────────────────────────

print("[STEP 1] Loading graph and safety edges …")
G          = ox.load_graphml(GRAPH_FILE)
edges_safe = gpd.read_file(EDGES_FILE)
print(f"[OK]   Nodes={G.number_of_nodes():,}  Edges={G.number_of_edges():,}")

# Build safety lookup
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

# Stamp graph edges
for u, v, key, data in G.edges(keys=True, data=True):
    info = safety_lookup.get((u, v)) or safety_lookup.get((v, u)) or {}
    data["safety_score"]       = info.get("safety_score",       0.30)
    data["safety_score_night"] = info.get("safety_score_night", 0.25)
    data["activity_composite"] = info.get("activity_composite", 0.30)
    data["incident_risk"]      = info.get("incident_risk",      0.50)
    data["police_bonus"]       = info.get("police_bonus",       0.00)
    data["edge_name"]          = info.get("edge_name",          "Unnamed Road")

# Cost function — same weights as api/main.py
ALPHA      = 0.50
BETA       = 0.50
all_lens   = [d.get("length", 1) for _, _, d in G.edges(data=True)]
max_length = max(all_lens) if all_lens else 1.0

for u, v, key, data in G.edges(keys=True, data=True):
    length = data.get("length", 1)
    pb     = data.get("police_bonus", 0.00)
    for mode_label, score_col in [
        ("day",   "safety_score"),
        ("night", "safety_score_night"),
    ]:
        base_safety = data.get(score_col, 0.30)
        cost = ALPHA * (length / max_length) + BETA * (1.0 - base_safety)
        cost = max(cost - pb, 0.01)
        data[f"safe_cost_{mode_label}"] = cost

print("[OK]   Graph ready with safety costs.")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — ROUTING HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def get_route_stats(G, path_nodes: list, mode: str = "night") -> dict:
    total_dist, safety_vals, incident_vals = 0.0, [], []
    score_col = "safety_score_night" if mode == "night" else "safety_score"
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        if v not in G[u]:
            continue
        ed = min(G[u][v].values(), key=lambda d: d.get("length", 1))
        total_dist    += ed.get("length", 0)
        safety_vals.append(ed.get(score_col,      0.30))
        incident_vals.append(ed.get("incident_risk", 0.50))
    return {
        "distance_m":   round(total_dist, 1),
        "avg_safety":   round(float(np.mean(safety_vals))   if safety_vals   else 0.0, 4),
        "avg_incident": round(float(np.mean(incident_vals)) if incident_vals else 0.0, 4),
    }


def get_shortest_path(G, origin_latlon: tuple, dest_latlon: tuple) -> list:
    orig = ox.nearest_nodes(G, origin_latlon[1], origin_latlon[0])
    dest = ox.nearest_nodes(G, dest_latlon[1],   dest_latlon[0])
    return nx.shortest_path(G, orig, dest, weight="length")


def get_safest_path(G, origin_latlon: tuple, dest_latlon: tuple,
                    mode: str = "night") -> list:
    orig      = ox.nearest_nodes(G, origin_latlon[1], origin_latlon[0])
    dest_node = ox.nearest_nodes(G, dest_latlon[1],   dest_latlon[0])
    cost_col  = f"safe_cost_{mode}"
    dest_lat  = G.nodes[dest_node]["y"]
    dest_lon  = G.nodes[dest_node]["x"]

    def heuristic(u, v):
        u_lat = G.nodes[u]["y"]
        u_lon = G.nodes[u]["x"]
        return ((u_lat - dest_lat) ** 2 + (u_lon - dest_lon) ** 2) ** 0.5 * 111.0

    try:
        return nx.astar_path(G, orig, dest_node,
                             heuristic=heuristic, weight=cost_col)
    except Exception:
        return nx.shortest_path(G, orig, dest_node, weight="length")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — 20 RANDOM OD PAIRS
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 3] Running 20 random OD pair evaluations …")

all_nodes = list(G.nodes())
rng       = np.random.default_rng(42)
sampled   = rng.choice(len(all_nodes), size=40, replace=False)
node_pool = [all_nodes[i] for i in sampled]

od_results = []

for i in range(20):
    orig_node = node_pool[i]
    dest_node = node_pool[i + 20]

    origin_latlon = (G.nodes[orig_node]["y"], G.nodes[orig_node]["x"])
    dest_latlon   = (G.nodes[dest_node]["y"], G.nodes[dest_node]["x"])

    try:
        t0         = time.time()
        path_short = get_shortest_path(G, origin_latlon, dest_latlon)
        t_short    = (time.time() - t0) * 1000

        t0        = time.time()
        path_safe = get_safest_path(G, origin_latlon, dest_latlon, mode="night")
        t_safe    = (time.time() - t0) * 1000

        stats_short = get_route_stats(G, path_short, mode="night")
        stats_safe  = get_route_stats(G, path_safe,  mode="night")

        safety_gain = (
            (stats_safe["avg_safety"] - stats_short["avg_safety"])
            / max(stats_short["avg_safety"], 0.01) * 100
        )
        dist_overhead = (
            (stats_safe["distance_m"] - stats_short["distance_m"])
            / max(stats_short["distance_m"], 1) * 100
        )

        record = {
            "pair_id":           i,
            "shortest_distance": stats_short["distance_m"],
            "safest_distance":   stats_safe["distance_m"],
            "shortest_safety":   stats_short["avg_safety"],
            "safest_safety":     stats_safe["avg_safety"],
            "shortest_incident": stats_short["avg_incident"],
            "safest_incident":   stats_safe["avg_incident"],
            "safety_gain_pct":   round(safety_gain,   2),
            "dist_overhead_pct": round(dist_overhead, 2),
            "t_short_ms":        round(t_short, 1),
            "t_safe_ms":         round(t_safe,  1),
            "status":            "ok",
        }

    except Exception as e:
        record = {"pair_id": i, "status": f"failed: {str(e)}"}

    od_results.append(record)
    print(f"  [{i+1:02d}/20] {record['status']}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — AGGREGATE METRICS
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 4] Aggregate metrics …")

valid_records = [r for r in od_results if r.get("status") == "ok"]
valid_pairs   = len(valid_records)

if valid_records:
    safety_gains    = [r["safety_gain_pct"]   for r in valid_records]
    dist_overheads  = [r["dist_overhead_pct"] for r in valid_records]
    t_shorts        = [r["t_short_ms"]        for r in valid_records]
    t_safes         = [r["t_safe_ms"]         for r in valid_records]

    avg_safety_gain           = float(np.mean(safety_gains))
    avg_dist_overhead         = float(np.mean(dist_overheads))
    route_quality_score       = avg_safety_gain / max(avg_dist_overhead, 0.01)
    pct_safer_also_shorter    = float(np.mean(
        [r["safest_distance"] <= r["shortest_distance"] for r in valid_records]
    )) * 100
    avg_t_short_ms            = float(np.mean(t_shorts))
    avg_t_safe_ms             = float(np.mean(t_safes))
else:
    safety_gains = dist_overheads = t_shorts = t_safes = []
    avg_safety_gain = avg_dist_overhead = route_quality_score = 0.0
    pct_safer_also_shorter = avg_t_short_ms = avg_t_safe_ms = 0.0

# Pretty summary table
w_m, w_v = 40, 14
print(f"\n  ╔{'═'*w_m}╦{'═'*w_v}╗")
print(f"  ║ {'Metric':<{w_m-2}} ║ {'Value':^{w_v-2}} ║")
print(f"  ╠{'═'*w_m}╬{'═'*w_v}╣")

def trow(label, value):
    return f"  ║ {label:<{w_m-2}} ║ {value:^{w_v-2}} ║"

print(trow("Valid pairs evaluated",             str(valid_pairs)))
print(trow("Avg safety gain (%)",               f"{avg_safety_gain:.2f}"))
print(trow("Avg distance overhead (%)",         f"{avg_dist_overhead:.2f}"))
print(trow("Route quality score",               f"{route_quality_score:.4f}"))
print(trow("Pairs where safest = shorter (%)",  f"{pct_safer_also_shorter:.1f}"))
print(trow("Avg compute time — shortest (ms)",  f"{avg_t_short_ms:.1f}"))
print(trow("Avg compute time — safest (ms)",    f"{avg_t_safe_ms:.1f}"))
print(f"  ╚{'═'*w_m}╩{'═'*w_v}╝")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — DISTRIBUTION VALIDATION PLOTS
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 5] Distribution validation plots …")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
dist_stats: dict = {}

for ax, col, label in [
    (axes[0], "safety_score",       "DAY Safety Score"),
    (axes[1], "safety_score_night", "NIGHT Safety Score"),
]:
    if col not in edges_safe.columns:
        ax.text(0.5, 0.5, f"{col}\nnot found", ha="center", va="center",
                transform=ax.transAxes)
        dist_stats[col] = {}
        continue

    s    = edges_safe[col].dropna()
    skew = float(scipy_stats.skew(s))
    kurt = float(scipy_stats.kurtosis(s))

    color = "#2980b9" if "night" not in col else "#8e44ad"
    ax.hist(s, bins=50, color=color, edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(s.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean={s.mean():.3f}")
    skew_flag = "  ⚠ HIGH" if abs(skew) > 1.5 else ""
    ax.set_title(f"{label}\nSkewness={skew:.3f}{skew_flag}")
    ax.set_xlabel("Safety Score")
    ax.set_ylabel("Edge Count")
    ax.legend()
    ax.grid(alpha=0.3)

    # Bucket stats
    bins      = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    counts, _ = np.histogram(s, bins=bins)
    total     = len(s)
    bucket_pct = {
        "0.0-0.2": round(counts[0] / total * 100, 2),
        "0.2-0.4": round(counts[1] / total * 100, 2),
        "0.4-0.6": round(counts[2] / total * 100, 2),
        "0.6-0.8": round(counts[3] / total * 100, 2),
        "0.8-1.0": round(counts[4] / total * 100, 2),
    }

    skew_note = " ⚠ HIGH SKEW" if abs(skew) > 1.5 else ""
    print(f"\n  {label}")
    print(f"    Mean:     {s.mean():.4f}   Std: {s.std():.4f}")
    print(f"    Min:      {s.min():.4f}   Max: {s.max():.4f}")
    print(f"    Skewness: {skew:.4f}{skew_note}   Kurtosis: {kurt:.4f}")
    for band, pct in bucket_pct.items():
        bar = "█" * int(pct / 2)
        print(f"    {band}: {pct:6.2f}%  {bar}")

    dist_stats[col] = {
        "mean":       round(float(s.mean()),  4),
        "std":        round(float(s.std()),   4),
        "min":        round(float(s.min()),   4),
        "max":        round(float(s.max()),   4),
        "skewness":   round(skew, 4),
        "kurtosis":   round(kurt, 4),
        "bucket_pct": bucket_pct,
    }

plt.tight_layout()
dist_plot_path = os.path.join(EVAL_DIR, "safety_distribution.png")
plt.savefig(dist_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print("\n[OK]   safety_distribution.png saved")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — ABLATION STUDY
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 6] Ablation study …")

FEATURES_TO_ABLATE = [
    "pedestrian_norm",
    "incident_risk",
    "activity_composite",
    "police_bonus",
    "lamp_norm",
    "poi_norm",
    "road_importance",
]

def _recompute_safety(df: gpd.GeoDataFrame) -> pd.Series:
    """Recompute safety_score using the same formula as safety_scorer.py."""
    # Default traffic_score if column missing
    traffic_score = df["traffic_score"].fillna(0.35) \
        if "traffic_score" in df.columns \
        else pd.Series(0.35, index=df.index)

    poi_norm        = df["poi_norm"].fillna(0.0)        if "poi_norm"        in df.columns else pd.Series(0.0, index=df.index)
    pedestrian_norm = df["pedestrian_norm"].fillna(0.0) if "pedestrian_norm" in df.columns else pd.Series(0.0, index=df.index)
    road_importance = df["road_importance"].fillna(0.0) if "road_importance" in df.columns else pd.Series(0.0, index=df.index)
    activity_comp   = df["activity_composite"].fillna(0.0) if "activity_composite" in df.columns else pd.Series(0.0, index=df.index)
    lamp_norm       = df["lamp_norm"].fillna(0.0)       if "lamp_norm"       in df.columns else pd.Series(0.0, index=df.index)
    incident_risk   = df["incident_risk"].fillna(0.0)   if "incident_risk"   in df.columns else pd.Series(0.0, index=df.index)
    police_bonus    = df["police_bonus"].fillna(0.0)    if "police_bonus"    in df.columns else pd.Series(0.0, index=df.index)

    footfall  = (0.40 * poi_norm + 0.30 * pedestrian_norm
                 + 0.20 * road_importance + 0.10 * activity_comp)
    lighting  = 0.55 * lamp_norm + 0.45 * road_importance
    crime     = (0.60 * incident_risk + 0.25 * (1 - lighting)
                 + 0.15 * (1 - footfall)).clip(0, 1)
    act_comp  = (0.50 * activity_comp + 0.30 * pedestrian_norm
                 + 0.20 * traffic_score)
    raw       = 0.30 * lighting + 0.30 * footfall + 0.20 * act_comp - 0.10 * crime
    raw      += police_bonus
    score     = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return score.clip(0.01, 0.99)


# Full model baseline
full_mean = float(edges_safe["safety_score"].mean()) \
    if "safety_score" in edges_safe.columns else 0.0
full_skew = float(scipy_stats.skew(edges_safe["safety_score"].dropna())) \
    if "safety_score" in edges_safe.columns else 0.0

ablation_table = []

print(f"\n  {'Feature Removed':<22} | {'Avg Safety':>10} | {'Skewness':>8} | {'Delta vs Full':>13}")
print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*8}-+-{'-'*13}")
print(f"  {'None (full model)':<22} | {full_mean:>10.4f} | {full_skew:>8.4f} | {'—':>13}")

for feat in FEATURES_TO_ABLATE:
    ablated = edges_safe.copy()
    if feat in ablated.columns:
        ablated[feat] = 0.0

    try:
        score          = _recompute_safety(ablated)
        ablated_mean   = float(score.mean())
        ablated_skew   = float(scipy_stats.skew(score))
        delta_mean     = ablated_mean - full_mean
    except Exception as ex:
        ablated_mean = ablated_skew = delta_mean = float("nan")
        print(f"  [WARN] Ablation failed for {feat}: {ex}")

    ablation_table.append({
        "feature_removed": feat,
        "avg_safety":      round(ablated_mean, 4),
        "skewness":        round(ablated_skew,  4),
        "delta_vs_full":   round(delta_mean,   4),
    })
    print(f"  {feat:<22} | {ablated_mean:>10.4f} | {ablated_skew:>8.4f} | {delta_mean:>+13.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — ROUTING SCATTER PLOT
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 7] Routing evaluation scatter plot …")

if valid_records:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        dist_overheads, safety_gains,
        alpha=0.75, s=60, color="#27ae60",
        edgecolors="white", linewidth=0.5,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Distance Overhead (%)")
    ax.set_ylabel("Safety Gain (%)")
    ax.set_title(
        "Safety Gain vs Distance Overhead\n"
        "(each point = one OD pair, night mode)"
    )
    ax.annotate(
        f"Avg Safety Gain: {avg_safety_gain:.1f}%\n"
        f"Avg Dist Overhead: {avg_dist_overhead:.1f}%\n"
        f"Quality Score: {route_quality_score:.2f}",
        xy=(0.05, 0.88), xycoords="axes fraction",
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow"),
    )
    plt.tight_layout()
    scatter_path = os.path.join(EVAL_DIR, "routing_scatter.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK]   routing_scatter.png saved")
else:
    print("[WARN] No valid OD pairs — scatter plot skipped.")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 — SAVE FULL RESULTS JSON
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 8] Saving evaluation/results.json …")

output = {
    "aggregate_metrics": {
        "valid_pairs":            valid_pairs,
        "avg_safety_gain_pct":    round(avg_safety_gain,        4),
        "avg_dist_overhead_pct":  round(avg_dist_overhead,      4),
        "route_quality_score":    round(route_quality_score,    4),
        "pct_safer_also_shorter": round(pct_safer_also_shorter, 4),
        "avg_t_short_ms":         round(avg_t_short_ms,         2),
        "avg_t_safe_ms":          round(avg_t_safe_ms,          2),
    },
    "od_pair_results":    od_results,
    "ablation_table":     ablation_table,
    "distribution_stats": dist_stats,
}

results_path = os.path.join(EVAL_DIR, "results.json")
with open(results_path, "w", encoding="utf-8") as fh:
    json.dump(output, fh, indent=2, ensure_ascii=False)

print(f"[OK]   results.json saved ({os.path.getsize(results_path):,} bytes)")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 9 — FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print("\n[DONE] Evaluation complete.")
print("  • evaluation/results.json")
print("  • evaluation/safety_distribution.png")
print("  • evaluation/routing_scatter.png")
print("  (feature_importance.png and shap_summary.png")
print("   already saved by Agent 3)")
