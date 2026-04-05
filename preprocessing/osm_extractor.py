"""
OSM Extractor for Night Navigator
Extracts road network, POIs, and street lamps for Bangalore, India
using OSMnx and saves them as GeoJSON / GraphML files.
"""

import subprocess
import sys
import os


# ── 1. Ensure osmnx is installed ─────────────────────────────────────────────
def install_package(package: str) -> None:
    """Install a Python package using pip if it is not already available."""
    try:
        __import__(package.replace("-", "_"))
        print(f"[INFO] '{package}' is already installed.")
    except ImportError:
        print(f"[INFO] '{package}' not found. Installing …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"[INFO] '{package}' installed successfully.")


install_package("osmnx")
install_package("geopandas")

import osmnx as ox          # noqa: E402  (imported after conditional install)
import geopandas as gpd     # noqa: E402


# ── Configuration ─────────────────────────────────────────────────────────────
PLACE_NAME   = "Bangalore, India"
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")

GRAPH_FILE   = os.path.join(OUTPUT_DIR, "bangalore_graph.graphml")
EDGES_FILE   = os.path.join(OUTPUT_DIR, "edges.geojson")
NODES_FILE   = os.path.join(OUTPUT_DIR, "nodes.geojson")
POIS_FILE    = os.path.join(OUTPUT_DIR, "pois.geojson")
LAMPS_FILE   = os.path.join(OUTPUT_DIR, "lamps.geojson")

POI_TAGS     = {"amenity": True, "shop": True}
LAMP_TAGS    = {"highway": "street_lamp"}


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    print(f"[INFO] Output directory ready: {os.path.abspath(path)}")


def save_geodataframe(gdf: gpd.GeoDataFrame, filepath: str, label: str) -> int:
    """Save a GeoDataFrame to GeoJSON, returning the row count."""
    if gdf is None or gdf.empty:
        print(f"[WARN] No {label} data to save.")
        return 0
    # GeoJSON requires EPSG:4326
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(filepath, driver="GeoJSON")
    count = len(gdf)
    print(f"[OK]   {label} saved → {filepath}  ({count} records)")
    return count


# ── 2. Road network ───────────────────────────────────────────────────────────
def extract_road_network():
    print(f"\n[STEP 1] Downloading road network for '{PLACE_NAME}' …")
    graph = ox.graph_from_place(PLACE_NAME, network_type="drive")

    # ── 3. Save graph ─────────────────────────────────────────────────────────
    try:
        ox.save_graphml(graph, filepath=GRAPH_FILE)
        print(f"[OK]   Graph saved → {GRAPH_FILE}")
    except Exception as exc:
        print(f"[ERROR] Failed to save GraphML: {exc}")

    # ── 4. Extract edges & nodes as GeoDataFrames ─────────────────────────────
    try:
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)
    except Exception as exc:
        print(f"[ERROR] graph_to_gdfs failed: {exc}")
        return 0, 0, graph

    # ── 5. Save edges & nodes ─────────────────────────────────────────────────
    n_nodes = save_geodataframe(nodes_gdf, NODES_FILE, "Nodes")
    n_edges = save_geodataframe(edges_gdf, EDGES_FILE, "Edges")

    return n_nodes, n_edges, graph


# ── 6 & 7. Points of Interest ─────────────────────────────────────────────────
def extract_pois():
    print(f"\n[STEP 2] Downloading POIs for '{PLACE_NAME}' …")
    try:
        pois = ox.features_from_place(PLACE_NAME, tags=POI_TAGS)
        return save_geodataframe(pois, POIS_FILE, "POIs")
    except Exception as exc:
        print(f"[ERROR] POI extraction failed: {exc}")
        return 0


# ── 8 & 9. Street lamps ───────────────────────────────────────────────────────
def extract_lamps():
    print(f"\n[STEP 3] Downloading street lamps for '{PLACE_NAME}' …")
    try:
        lamps = ox.features_from_place(PLACE_NAME, tags=LAMP_TAGS)
        return save_geodataframe(lamps, LAMPS_FILE, "Lamps")
    except Exception as exc:
        print(f"[ERROR] Street lamp extraction failed: {exc}")
        return 0


# ── 10. Main entry point ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Night Navigator — OSM Data Extractor")
    print("=" * 60)

    ensure_output_dir(OUTPUT_DIR)

    n_nodes, n_edges, _ = extract_road_network()
    n_pois              = extract_pois()
    n_lamps             = extract_lamps()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Nodes   : {n_nodes:>8,}")
    print(f"  Edges   : {n_edges:>8,}")
    print(f"  POIs    : {n_pois:>8,}")
    print(f"  Lamps   : {n_lamps:>8,}")
    print("=" * 60)
    print("[DONE] All data saved to the 'data/' directory.")


if __name__ == "__main__":
    main()
