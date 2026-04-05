"""
Safety Scorer v3 — Night Navigator
Enriches road edges with OSM + traffic features and computes
day/night safety scores with fixed formulas, dynamic road importance,
and gentle normalization.
"""

import os
import sys
import warnings
import ast

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats as scipy_stats
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR          = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR           = os.path.abspath(os.path.join(_BASE_DIR, "data"))

EDGES_FILE         = os.path.join(DATA_DIR, "edges.geojson")
POIS_FILE          = os.path.join(DATA_DIR, "pois.geojson")
LAMPS_FILE         = os.path.join(DATA_DIR, "lamps.geojson")          # kept for fallback
WARDS_FILE         = os.path.join(DATA_DIR, "wards_369.geojson")
WARD_LIGHTS_CSV    = os.path.join(DATA_DIR, "ward_streetlights.csv")
TRAFFIC_RAW_CSV    = os.path.join(DATA_DIR, "Banglore_traffic_Dataset.csv")
OUTPUT_FILE        = os.path.join(DATA_DIR, "edges_with_safety.geojson")
POLICE_FILE        = os.path.join(DATA_DIR, "police_stations.geojson")

# UTM Zone 44N — accurate for Bangalore in metres
METRIC_CRS   = "EPSG:32644"
POI_BUFFER_M = 300   # increased from 150 m to capture more POIs

# ── Highway base scores ───────────────────────────────────────────────────────
HIGHWAY_BASE: dict[str, float] = {
    "motorway":         1.0,
    "motorway_link":    0.9,
    "trunk":            0.9,
    "trunk_link":       0.85,
    "primary":          0.8,
    "primary_link":     0.75,
    "secondary":        0.7,
    "secondary_link":   0.65,
    "tertiary":         0.6,
    "tertiary_link":    0.55,
    "living_street":    0.5,
    "residential":      0.4,
    "unclassified":     0.3,
    "service":          0.2,
    "path":             0.1,
    "track":            0.1,
    "footway":          0.1,
    "cycleway":         0.1,
}
DEFAULT_HIGHWAY_BASE = 0.35

# ── Hardcoded area centroids (lat, lon) ───────────────────────────────────────
AREA_COORDS: dict[str, tuple[float, float]] = {
    "Indiranagar":     (12.9784, 77.6408),
    "Whitefield":      (12.9698, 77.7499),
    "Koramangala":     (12.9352, 77.6245),
    "M.G. Road":       (12.9766, 77.6101),
    "Jayanagar":       (12.9308, 77.5838),
    "Hebbal":          (13.0354, 77.5970),
    "Yeshwanthpur":    (13.0289, 77.5510),
    "Electronic City": (12.8458, 77.6603),
}
AREA_NAMES = list(AREA_COORDS.keys())
AREA_LATS  = np.array([v[0] for v in AREA_COORDS.values()], dtype=np.float64)
AREA_LONS  = np.array([v[1] for v in AREA_COORDS.values()], dtype=np.float64)


# ── Utility functions ─────────────────────────────────────────────────────────

def minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]; returns 0.5 everywhere if all values equal."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def robust_norm(series: pd.Series) -> pd.Series:
    """
    Robust normalisation: clip to 2nd–98th percentile, then min-max to [0, 1].
    Reduces the effect of extreme outliers on the normalised range.
    """
    lo = series.quantile(0.02)
    hi = series.quantile(0.98)
    clipped = series.clip(lo, hi)
    return minmax_norm(clipped)


def robust_norm_clip(
    series: pd.Series,
    lower: float = 0.02,
    upper: float = 0.98,
    clip_min: float = 0.01,
    clip_max: float = 0.99,
) -> pd.Series:
    """
    Robust normalisation with percentile clipping.
    1. Clip series to [lower_pct, upper_pct] percentiles (removes outlier influence).
    2. Apply min-max normalisation to spread the result across [0, 1].
    3. Clip final output to [clip_min, clip_max].
    """
    lo_val = series.quantile(lower)
    hi_val = series.quantile(upper)
    print(f"[INFO] Robust norm: lower={lo_val:.4f} (p{lower*100:.0f}), "
          f"upper={hi_val:.4f} (p{upper*100:.0f})")
    clipped = series.clip(lo_val, hi_val)
    return minmax_norm(clipped).clip(clip_min, clip_max)


def load_gdf(filepath: str, label: str) -> gpd.GeoDataFrame | None:
    if not os.path.exists(filepath):
        print(f"[WARN] {label} not found: {filepath}")
        return None
    try:
        gdf = gpd.read_file(filepath)
        print(f"[OK]   {label}: {len(gdf):,} records")
        return gdf
    except Exception as exc:
        print(f"[ERROR] Failed to load {label}: {exc}")
        return None


def load_csv(filepath: str, label: str) -> pd.DataFrame | None:
    if not os.path.exists(filepath):
        print(f"[WARN] {label} not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        print(f"[OK]   {label}: {len(df):,} rows")
        return df
    except Exception as exc:
        print(f"[ERROR] Failed to load {label}: {exc}")
        return None


def highway_base_score(val) -> float:
    """
    Map any OSM highway tag representation to a base importance score.
    Handles: plain str, list, numpy ndarray, and str repr of a list.
    """
    import numpy as np  # local import to avoid circular deps

    # numpy ndarray (what geopandas returns after reading GeoJSON)
    if isinstance(val, np.ndarray):
        val = val.tolist()

    # String that looks like a list repr: "['residential']"
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass

    if isinstance(val, list):
        if not val:
            return DEFAULT_HIGHWAY_BASE
        return max(HIGHWAY_BASE.get(str(h).strip(), DEFAULT_HIGHWAY_BASE) for h in val)

    if isinstance(val, str):
        return HIGHWAY_BASE.get(val.strip(), DEFAULT_HIGHWAY_BASE)

    return DEFAULT_HIGHWAY_BASE


def count_within_buffer(edges_m: gpd.GeoDataFrame,
                         points: gpd.GeoDataFrame,
                         buffer_m: float) -> pd.Series:
    """Count points inside a buffer (metres) around each edge (spatial join)."""
    buffers = edges_m.geometry.buffer(buffer_m)
    buf_gdf = gpd.GeoDataFrame(
        {"edge_idx": edges_m.index},
        geometry=buffers,
        crs=edges_m.crs,
    )
    pts = points.copy()
    if not all(pts.geometry.geom_type == "Point"):
        pts["geometry"] = pts.geometry.centroid
    pts = pts[["geometry"]].reset_index(drop=True)

    joined = gpd.sjoin(buf_gdf, pts, how="left", predicate="contains")
    counts = joined.groupby("edge_idx").size()
    return counts.reindex(edges_m.index, fill_value=0)


def haversine_distances(lat: float, lon: float,
                         lats: np.ndarray,
                         lons: np.ndarray) -> np.ndarray:
    R = 6371.0
    dlat = np.radians(lats - lat)
    dlon = np.radians(lons - lon)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat)) * np.cos(np.radians(lats))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def serialize_lists(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert list-typed OSM columns to comma-joined strings for GeoJSON."""
    for col in gdf.columns:
        if gdf[col].dtype == object:
            sample = gdf[col].dropna().head(10)
            if sample.apply(lambda x: isinstance(x, list)).any():
                gdf[col] = gdf[col].apply(
                    lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x
                )
    return gdf


# ── Ward streetlight helpers ──────────────────────────────────────────────────

def load_wards_with_lights() -> tuple[gpd.GeoDataFrame | None, dict[str, float]]:
    """
    Load ward polygons from GeoJSON and join streetlight counts from CSV.
    Returns (wards_gdf_with_light_count, ward_name->light_count dict).
    Missing wards are imputed with the median of known counts.
    """
    # ── Load ward polygons ──────────────────────────────────────────────────
    if not os.path.exists(WARDS_FILE):
        print(f"[WARN] Wards GeoJSON not found: {WARDS_FILE}")
        return None, {}
    try:
        wards = gpd.read_file(WARDS_FILE)

        # GeoJSON uses 'ward_name'; rename to 'Name' for internal consistency
        name_col = next(
            (c for c in wards.columns
             if c.strip().lower() in ("name", "ward_name", "wardname", "ward name")),
            None
        )
        if name_col is None:
            print(f"[WARN] No ward name column found. Columns: {wards.columns.to_list()}")
            return None, {}
        if name_col != "Name":
            wards = wards.rename(columns={name_col: "Name"})

        wards = wards[["Name", "geometry"]].copy()
        wards = wards[wards.geometry.notna()]
        wards = wards[wards["Name"].notna() & (wards["Name"].str.strip() != "")]
        wards["Name"] = wards["Name"].str.strip()
        print(f"[OK]   Wards GeoJSON: {len(wards):,} wards")
    except Exception as exc:
        print(f"[ERROR] Failed to load wards GeoJSON: {exc}")
        return None, {}

    # ── Load streetlight CSV ────────────────────────────────────────────────
    light_dict: dict[str, float] = {}
    if not os.path.exists(WARD_LIGHTS_CSV):
        print(f"[WARN] Streetlights CSV not found: {WARD_LIGHTS_CSV}")
    else:
        try:
            sl = pd.read_csv(WARD_LIGHTS_CSV)
            # Strip whitespace AND carriage returns from column names (CSV may have \r)
            sl.columns = sl.columns.str.strip().str.replace("\r", "", regex=False)

            # Try exact known column names first, fall back to fuzzy match
            ward_col  = "Ward Name"       if "Ward Name"       in sl.columns else None
            light_col = "Street lights #" if "Street lights #" in sl.columns else None

            if ward_col is None:
                ward_col = next(
                    (c for c in sl.columns if "ward" in c.lower() and "name" in c.lower()), None
                )
            if light_col is None:
                light_col = next(
                    (c for c in sl.columns if "light" in c.lower()), None
                )

            if ward_col and light_col:
                sl[light_col] = pd.to_numeric(sl[light_col], errors="coerce")
                light_dict = {
                    str(name).strip(): val
                    for name, val in zip(sl[ward_col], sl[light_col])
                    if pd.notna(val)
                }
                print(f"[OK]   Streetlights CSV: {len(light_dict):,} wards with data")
            else:
                print(f"[WARN] Could not identify ward/light columns. Found: {sl.columns.to_list()}")
        except Exception as exc:
            print(f"[ERROR] Failed to load streetlights CSV: {exc}")

    # ── Median imputation for missing wards ──────────────────────────────────
    known_counts = list(light_dict.values())
    median_count = float(np.median(known_counts)) if known_counts else 500.0
    n_with    = len(light_dict)
    n_missing = len(wards) - n_with
    print(f"[INFO] Wards with light data: {n_with} | without: {n_missing} | imputed median: {median_count:.0f}")

    wards["light_count"] = wards["Name"].map(
        lambda name: light_dict.get(name, np.nan)
    )
    wards["light_count"] = wards["light_count"].fillna(median_count)

    return wards, light_dict


def compute_lamp_density_fast(
    edges_m: gpd.GeoDataFrame,
    wards_m: gpd.GeoDataFrame,
) -> pd.Series:
    """
    Assign each edge to the nearest ward by its centroid using KDTree (fast).
    lamp_density = light_count / total_road_km  per ward, mapped back to edges.
    Returns a Series of lamp_density aligned to edges_m.index.
    """
    # Edge centroid coordinates (metric CRS)
    edge_centroids = edges_m.geometry.centroid
    edge_coords    = np.array([(p.x, p.y) for p in edge_centroids])

    # Ward centroid coordinates
    ward_centroids = wards_m.geometry.centroid
    ward_coords    = np.array([(p.x, p.y) for p in ward_centroids])

    # Nearest-ward lookup via KDTree (fully vectorized, no Python loops)
    tree = cKDTree(ward_coords)
    _, ward_indices = tree.query(edge_coords, k=1)

    # Assign ward name + light_count to each edge
    edges_work = edges_m[["geometry"]].copy()
    edges_work["_ward_name"]   = wards_m.iloc[ward_indices]["Name"].values
    edges_work["_light_count"] = wards_m.iloc[ward_indices]["light_count"].values
    edges_work["_length_km"]   = edges_m.geometry.length / 1000.0

    # Per-ward total road km
    ward_road_km = (
        edges_work.groupby("_ward_name")["_length_km"]
        .sum()
        .rename("total_road_km")
    )
    wards_work = wards_m.copy()
    wards_work = wards_work.merge(
        ward_road_km, left_on="Name", right_index=True, how="left"
    )
    wards_work["total_road_km"] = wards_work["total_road_km"].fillna(0.1)

    # Lamp density per ward
    EPSILON = 1e-3
    wards_work["lamp_density"] = (
        wards_work["light_count"] / (wards_work["total_road_km"] + EPSILON)
    )

    # Debug stats
    ld = wards_work["lamp_density"]
    print(f"[DEBUG] lamp_density — min={ld.min():.2f}, max={ld.max():.2f}, "
          f"mean={ld.mean():.2f}, zeros={(ld == 0).sum()}")
    print("[DEBUG] Top 5 wards by lamp_density:")
    top5 = wards_work.nlargest(5, "lamp_density")[
        ["Name", "light_count", "total_road_km", "lamp_density"]
    ]
    print(top5.to_string(index=False))

    # Map ward-level density back to each edge
    ward_density_map = dict(zip(wards_work["Name"], wards_work["lamp_density"]))
    lamp_density = edges_work["_ward_name"].map(ward_density_map).fillna(0.0)

    return lamp_density



def compute_police_bonus(edges_m: gpd.GeoDataFrame) -> pd.Series:
    """
    Compute police station proximity bonus for each edge.
    Returns police_bonus (0 to 0.05) as a Series aligned to edges_m.index.
    Uses .centroid to handle Point, MultiPoint, and Polygon geometries safely.
    """
    if not os.path.exists(POLICE_FILE):
        print(f"[WARN] Police stations file not found: {POLICE_FILE}")
        return pd.Series(0.0, index=edges_m.index)

    try:
        ps = gpd.read_file(POLICE_FILE)
        print(f"[OK]   Police stations loaded: {len(ps)}")
    except Exception as e:
        print(f"[WARN] Failed to load police stations: {e}")
        return pd.Series(0.0, index=edges_m.index)

    # Reproject to metric CRS (same as edges)
    ps = ps.to_crs(edges_m.crs)

    # Use .centroid to safely handle Point, MultiPoint, and Polygon
    ps_coords = np.array([(g.centroid.x, g.centroid.y) for g in ps.geometry])

    # Build KDTree for fast nearest-neighbour search
    tree = cKDTree(ps_coords)

    # Edge centroids in metric CRS
    edge_centroids = edges_m.geometry.centroid
    edge_coords    = np.array([(p.x, p.y) for p in edge_centroids])

    # Query nearest distance (metres)
    distances, _ = tree.query(edge_coords, k=1)

    # Convert distance to bonus (max influence 1000 m, weight 0.05)
    MAX_DIST = 1000.0
    WEIGHT   = 0.05

    raw_bonus    = 1 - np.clip(distances / MAX_DIST, 0, 1)
    police_bonus = WEIGHT * raw_bonus

    print(f"[INFO] Police proximity: mean distance = {distances.mean():.1f} m, "
          f"min = {distances.min():.1f} m, max = {distances.max():.1f} m")
    print(f"[INFO] police_bonus mean = {police_bonus.mean():.4f}, "
          f"min = {police_bonus.min():.4f}, max = {police_bonus.max():.4f}")

    return pd.Series(police_bonus, index=edges_m.index)


def print_diagnostics(series: pd.Series, label: str) -> None:
    """Print distribution stats, histogram, and skewness warning."""
    clean = series.dropna()
    skew  = scipy_stats.skew(clean)
    bins  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["[0.0-0.2]", "[0.2-0.4]", "[0.4-0.6]", "[0.6-0.8]", "[0.8-1.0]"]

    print(f"\n{'=' * 65}")
    print(f"  DIAGNOSTICS — {label}")
    print(f"{'=' * 65}")
    print(f"  Mean     : {clean.mean():.4f}")
    print(f"  Std      : {clean.std():.4f}")
    print(f"  Min      : {clean.min():.4f}")
    print(f"  Max      : {clean.max():.4f}")
    print(f"  Skewness : {skew:.4f}")
    if abs(skew) > 1.5:
        print(f"  ⚠ WARNING: distribution still skewed (|skew| > 1.5)")

    counts, _ = np.histogram(clean, bins=bins)
    total  = max(counts.sum(), 1)
    print(f"\n  Histogram:")
    for lbl, cnt in zip(labels, counts):
        bar = "█" * int(cnt / total * 30)
        print(f"    {lbl}  {bar} ({cnt:,})")


def build_area_traffic_from_raw(raw_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Aggregate the raw Bangalore traffic dataset by 'Area Name' to derive
    realistic per-area traffic/pedestrian/incident scores.

    Returns { area_name: { traffic_score, activity_score, incident_risk,
                           avg_pedestrian, avg_traffic_volume } }
    """
    needed = {
        "Traffic Volume":              "avg_traffic_volume",
        "Pedestrian and Cyclist Count": "avg_pedestrian",
        "Incident Reports":             "avg_incidents",
        "Congestion Level":             "avg_congestion",
    }
    # Keep only columns that exist in the dataset
    present = {k: v for k, v in needed.items() if k in raw_df.columns}
    missing = set(needed) - set(present)
    if missing:
        print(f"[WARN] Missing columns in raw traffic data: {missing}")

    agg = raw_df.groupby("Area Name")[[*present.keys()]].mean().rename(
        columns=present
    )

    # ── Debug: top areas ─────────────────────────────────────────────────────
    print("\n[DEBUG] Top 5 areas by pedestrian count:")
    if "avg_pedestrian" in agg.columns:
        print(agg.nlargest(5, "avg_pedestrian")[["avg_pedestrian"]].to_string())
    print("\n[DEBUG] Top 5 areas by incident reports:")
    if "avg_incidents" in agg.columns:
        print(agg.nlargest(5, "avg_incidents")[["avg_incidents"]].to_string())

    # ── Normalise across areas ────────────────────────────────────────────────
    def _norm(col: str) -> pd.Series:
        s = agg[col]
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)

    norm_traffic    = _norm("avg_traffic_volume")  if "avg_traffic_volume" in agg.columns else pd.Series(0.5, index=agg.index)
    norm_pedestrian = _norm("avg_pedestrian")       if "avg_pedestrian"     in agg.columns else pd.Series(0.5, index=agg.index)
    norm_congestion = _norm("avg_congestion")       if "avg_congestion"     in agg.columns else pd.Series(0.5, index=agg.index)
    norm_incidents  = _norm("avg_incidents")        if "avg_incidents"      in agg.columns else pd.Series(0.5, index=agg.index)

    # traffic_score  = 0.5 * traffic + 0.5 * pedestrian
    # activity_score = 0.5 * congestion + 0.3 * pedestrian + 0.2 * traffic
    # incident_risk  = norm_incidents
    traffic_score  = 0.5 * norm_traffic + 0.5 * norm_pedestrian
    activity_score = 0.5 * norm_congestion + 0.3 * norm_pedestrian + 0.2 * norm_traffic
    incident_risk  = norm_incidents

    lookup: dict[str, dict[str, float]] = {}
    for area in agg.index:
        lookup[area] = {
            "traffic_score":  float(traffic_score.get(area, 0.5)),
            "activity_score": float(activity_score.get(area, 0.5)),
            "incident_risk":  float(incident_risk.get(area, 0.5)),
            "avg_pedestrian": float(agg.loc[area, "avg_pedestrian"]) if "avg_pedestrian" in agg.columns else 0.5,
        }

    print(f"[OK]   Built traffic lookup for {len(lookup)} areas from raw dataset")
    return lookup


def assign_area_features_from_raw(
    centroids_wgs84: gpd.GeoSeries,
    lookup: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    For each edge centroid, find nearest Bangalore area by Haversine distance
    and retrieve traffic features from the raw-derived lookup.
    Returns a DataFrame aligned to centroids_wgs84.index.
    """
    # Global defaults = mean across areas (fallback for unmapped areas)
    all_vals: dict[str, list[float]] = {}
    for area_data in lookup.values():
        for k, v in area_data.items():
            all_vals.setdefault(k, []).append(v)
    defaults = {k: float(np.mean(v)) if v else 0.5 for k, v in all_vals.items()}

    FEATURE_COLS = ["traffic_score", "activity_score", "incident_risk", "avg_pedestrian"]

    # Build area centroid arrays for vectorised nearest-area lookup
    area_names_list = list(lookup.keys())
    # For areas in lookup that are also in AREA_COORDS use known coords;
    # otherwise fall back to AREA_COORDS for the 8 canonical areas.
    lookup_lats = np.array([
        AREA_COORDS.get(a, list(AREA_COORDS.values())[0])[0] for a in area_names_list
    ])
    lookup_lons = np.array([
        AREA_COORDS.get(a, list(AREA_COORDS.values())[0])[1] for a in area_names_list
    ])

    # If the lookup only covers the canonical 8 areas, use the pre-built arrays
    use_canonical = set(area_names_list) <= set(AREA_NAMES)
    if use_canonical:
        area_names_list = AREA_NAMES
        lookup_lats     = AREA_LATS
        lookup_lons     = AREA_LONS

    records = []
    for geom in centroids_wgs84:
        lat, lon = geom.y, geom.x
        dists    = haversine_distances(lat, lon, lookup_lats, lookup_lons)
        area     = area_names_list[int(np.argmin(dists))]
        area_data = lookup.get(area, {})
        rec = {col: area_data.get(col, defaults.get(col, 0.5)) for col in FEATURE_COLS}
        records.append(rec)

    return pd.DataFrame(records, index=centroids_wgs84.index)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Night Navigator — Safety Scorer v3")
    print("=" * 65)

    # ── 1. Load all inputs ────────────────────────────────────────────────────
    print("\n[STEP 1] Loading input files …")
    edges       = load_gdf(EDGES_FILE, "Edges")
    pois        = load_gdf(POIS_FILE,  "POIs")
    traffic_raw = load_csv(TRAFFIC_RAW_CSV, "Raw traffic dataset")
    wards_raw, _ = load_wards_with_lights()

    if edges is None or edges.empty:
        print("[FATAL] Edge data required. Exiting.")
        sys.exit(1)

    # ── 2. Reproject to metric CRS ────────────────────────────────────────────
    print(f"\n[STEP 2] Reprojecting to {METRIC_CRS} …")
    edges_m  = edges.to_crs(METRIC_CRS)
    pois_m   = pois.to_crs(METRIC_CRS) if (pois is not None and not pois.empty) else None
    wards_m  = wards_raw.to_crs(METRIC_CRS) if wards_raw is not None else None

    # ── 3a. Highway base score ────────────────────────────────────────────────
    print("\n[STEP 3a] Highway base scores …")
    if "highway" in edges_m.columns:
        # Show a sample of the raw types so we can catch future breakage early
        sample_types = edges_m["highway"].dropna().head(3).apply(
            lambda v: f"{type(v).__name__}: {repr(v)[:50]}"
        ).to_list()
        print(f"[DEBUG] highway raw sample: {sample_types}")

        edges_m["highway_score"] = edges_m["highway"].apply(highway_base_score)

        print("[DEBUG] highway_score value_counts (top 10):")
        print(edges_m["highway_score"].value_counts().head(10).to_string())
    else:
        print("[WARN] No 'highway' column — defaulting to 0.35")
        edges_m["highway_score"] = DEFAULT_HIGHWAY_BASE

    # ── 4. Spatial buffer counts ──────────────────────────────────────────────
    # ── 4a. POI counts (buffer 150 m) ─────────────────────────────────────────
    print(f"\n[STEP 4a] POI counts (buffer {POI_BUFFER_M} m) …")
    if pois_m is not None:
        try:
            edges_m["poi_count"] = count_within_buffer(edges_m, pois_m, POI_BUFFER_M)
        except Exception as e:
            print(f"[WARN] POI join failed: {e} — using 0")
            edges_m["poi_count"] = 0
    else:
        edges_m["poi_count"] = 0

    edges_m["poi_norm"] = minmax_norm(edges_m["poi_count"].astype(float))

    # ── 4b. Ward-based lamp density (replaces OSM lamp buffer) ───────────────
    print("\n[STEP 4b] Ward-based lamp density …")
    if wards_m is not None:
        lamp_density = compute_lamp_density_fast(edges_m, wards_m)
        edges_m["lamp_density"] = lamp_density
        edges_m["lamp_norm"]    = robust_norm(edges_m["lamp_density"])
    else:
        print("[WARN] No ward data — lamp_norm set to 0")
        edges_m["lamp_density"] = 0.0
        edges_m["lamp_norm"]    = 0.0

    # ── 5. Traffic join from raw dataset ──────────────────────────────────────
    print("\n[STEP 5] Building traffic features from raw dataset …")
    if traffic_raw is not None and not traffic_raw.empty:
        area_lookup = build_area_traffic_from_raw(traffic_raw)
    else:
        print("[WARN] Raw traffic data unavailable — using uniform defaults")
        area_lookup = {name: {"traffic_score": 0.5, "activity_score": 0.5,
                               "incident_risk": 0.5, "avg_pedestrian": 0.5}
                       for name in AREA_NAMES}

    centroids_wgs84 = edges_m.geometry.centroid.to_crs("EPSG:4326")
    area_feats      = assign_area_features_from_raw(centroids_wgs84, area_lookup)

    for col in area_feats.columns:
        edges_m[col] = area_feats[col].values

    # Normalise avg_pedestrian across all edges → pedestrian_norm
    edges_m["pedestrian_norm"] = minmax_norm(edges_m["avg_pedestrian"].astype(float))

    print(f"[INFO] traffic_score   mean={edges_m['traffic_score'].mean():.4f}")
    print(f"[INFO] activity_score  mean={edges_m['activity_score'].mean():.4f}")
    print(f"[INFO] incident_risk   mean={edges_m['incident_risk'].mean():.4f}")
    print(f"[INFO] avg_pedestrian  mean={edges_m['avg_pedestrian'].mean():.2f}")

    # ── 3b. Dynamic road_importance (after traffic join) ─────────────────────
    print("\n[STEP 3b] Dynamic road_importance (highway + traffic + activity) …")
    edges_m["road_importance"] = (
        0.5 * edges_m["highway_score"]
        + 0.3 * edges_m["traffic_score"]
        + 0.2 * edges_m["activity_score"]
    )
    edges_m["road_importance"] = minmax_norm(edges_m["road_importance"])

    # Debug: raw feature check
    print("\n[DEBUG] Raw feature distributions:")
    debug_cols = ["poi_count", "lamp_density", "lamp_norm", "highway_score",
                  "road_importance", "traffic_score", "activity_score"]
    print(edges_m[[c for c in debug_cols if c in edges_m.columns]].describe().round(4).to_string())

    # ── 6. Composite sub-scores ───────────────────────────────────────────────
    print("\n[STEP 6] Computing composite scores …")

    ri  = edges_m["road_importance"]
    poi = edges_m["poi_norm"]
    lmp = edges_m["lamp_norm"]
    ped = edges_m["pedestrian_norm"]
    ts  = edges_m["traffic_score"]
    act = edges_m["activity_score"]
    ir  = edges_m["incident_risk"]


    # footfall_score
    edges_m["footfall_score"] = (
        0.40 * poi
        + 0.30 * ped
        + 0.20 * ri
        + 0.10 * act
    )

    # lighting_score (signal_bonus removed — weight redistributed to road_importance)
    edges_m["lighting_score"] = (
        0.55 * lmp
        + 0.45 * ri
    )

    fs = edges_m["footfall_score"]
    ls = edges_m["lighting_score"]

    # crime_score — incident-anchored, clipped
    edges_m["crime_score"] = (
        0.60 * ir
        + 0.25 * (1 - ls)
        + 0.15 * (1 - fs)
    ).clip(0, 1)

    cs = edges_m["crime_score"]

    # activity_composite
    edges_m["activity_composite"] = (
        0.50 * act
        + 0.30 * ped
        + 0.20 * ts
    )

    ac = edges_m["activity_composite"]

    # ── 7. Final safety scores ────────────────────────────────────────────────
    print("\n[STEP 7] Computing day & night safety scores …")

    raw_day = (
        0.30 * ls
        + 0.30 * fs
        + 0.20 * ac
        - 0.10 * cs
    )

    raw_night = (
        0.40 * ls
        + 0.25 * fs
        + 0.15 * ac
        - 0.10 * cs
    )

    print("[INFO] signal_bonus removed from formulas.")

    # ── 7b. Police station proximity bonus ───────────────────────────────────
    print("\n[STEP 7b] Police station proximity bonus …")
    police_bonus = compute_police_bonus(edges_m)
    edges_m["police_bonus"] = police_bonus.values

    # Add bonus to raw scores BEFORE normalisation so it affects the scale
    raw_day_pb   = raw_day   + police_bonus
    raw_night_pb = raw_night + police_bonus

    # ── 8. Robust percentile normalisation then clip ──────────────────────────
    print("\n[STEP 8] Applying robust_norm_clip to safety scores …")
    print("  [DAY]")
    edges_m["safety_score"]       = robust_norm_clip(raw_day_pb)
    print("  [NIGHT]")
    edges_m["safety_score_night"] = robust_norm_clip(raw_night_pb)

    # ── 9. Save ───────────────────────────────────────────────────────────────
    print(f"\n[STEP 9] Saving → {OUTPUT_FILE} …")
    try:
        edges_out = edges_m.to_crs("EPSG:4326")
        if isinstance(edges_out.index, pd.MultiIndex):
            edges_out = edges_out.reset_index()
        edges_out = serialize_lists(edges_out)
        edges_out.to_file(OUTPUT_FILE, driver="GeoJSON")
        print(f"[OK]   Saved {len(edges_out):,} enriched edges.")
    except Exception as exc:
        print(f"[ERROR] Save failed: {exc}")

    # ── 10. Distribution diagnostics ─────────────────────────────────────────
    print_diagnostics(edges_m["safety_score"],       "safety_score (DAY)")
    print_diagnostics(edges_m["safety_score_night"], "safety_score (NIGHT)")

    # ── 11. Top-5 safest & most dangerous ────────────────────────────────────
    score_cols = [
        "highway_score", "road_importance", "footfall_score",
        "lighting_score", "crime_score", "activity_composite",
        "safety_score", "safety_score_night",
    ]
    if "name" in edges_m.columns:
        display = ["name"] + score_cols
    else:
        display = score_cols

    for col, label in [("safety_score", "DAY"), ("safety_score_night", "NIGHT")]:
        print(f"\n{'=' * 65}")
        print(f"  TOP 5 SAFEST — {label}")
        print(f"{'=' * 65}")
        print(edges_m.nlargest(5, col)[display].to_string())

        print(f"\n  TOP 5 MOST DANGEROUS — {label}")
        print(f"{'=' * 65}")
        print(edges_m.nsmallest(5, col)[display].to_string())

    print(f"\n{'=' * 65}")
    print("[DONE] edges_with_safety.geojson written to data/")


if __name__ == "__main__":
    main()
