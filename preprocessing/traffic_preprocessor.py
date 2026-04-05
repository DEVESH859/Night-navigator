"""
Traffic Preprocessor v2 — Night Navigator
Loads Bangalore traffic data, computes per-area safety-relevant scores
(overall and night-time) using rebalanced formulas and saves CSVs to data/.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR          = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR           = os.path.abspath(os.path.join(_BASE_DIR, "data"))

INPUT_CSV          = os.path.join(DATA_DIR, "Banglore_traffic_Dataset.csv")
OUTPUT_AREA_CSV    = os.path.join(DATA_DIR, "traffic_area_scores.csv")
OUTPUT_NIGHT_CSV   = os.path.join(DATA_DIR, "traffic_night_scores.csv")

USE_COLS = [
    "Area Name",
    "Traffic Volume",
    "Average Speed",
    "Travel Time Index",
    "Congestion Level",
    "Pedestrian and Cyclist Count",
    "Incident Reports",
    "Traffic Signal Compliance",
    "Date",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]; returns 0 everywhere if all values equal."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


def map_hour_to_period(hour: int) -> str:
    if 0 <= hour < 6:
        return "night"
    elif 6 <= hour < 10:
        return "morning"
    elif 10 <= hour < 18:
        return "day"
    else:
        return "evening"


def compute_scores(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Given aggregated DataFrame with columns:
        avg_traffic_volume, avg_pedestrian, avg_congestion,
        avg_incidents, avg_signal_compliance
    Appends all derived score columns and returns the enriched DataFrame.
    """
    df = agg.copy()

    # ── Step 4: Normalise all numeric aggregates ──────────────────────────────
    df["norm_traffic"]    = minmax_norm(df["avg_traffic_volume"])
    df["norm_pedestrian"] = minmax_norm(df["avg_pedestrian"])
    df["norm_congestion"] = minmax_norm(df["avg_congestion"])
    df["norm_incidents"]  = minmax_norm(df["avg_incidents"])
    df["norm_signal"]     = minmax_norm(df["avg_signal_compliance"])

    # ── Step 5a: traffic_score ────────────────────────────────────────────────
    df["traffic_score"] = (
        0.5 * df["norm_traffic"]
        + 0.5 * df["norm_pedestrian"]
    )

    # ── Step 5b: activity_score (NEW) ─────────────────────────────────────────
    df["activity_score"] = (
        0.5 * df["norm_congestion"]
        + 0.3 * df["norm_pedestrian"]
        + 0.2 * df["norm_traffic"]
    )

    # ── Step 5c: incident_risk (FIXED — includes signal compliance) ───────────
    df["incident_risk"] = (
        0.7 * df["norm_incidents"]
        + 0.3 * (1 - df["norm_signal"])
    )

    # ── Step 5d: signal_bonus (NEW) ───────────────────────────────────────────
    df["signal_bonus"] = df["norm_signal"]

    # ── Step 5e: traffic_safety_score (REBALANCED) ────────────────────────────
    df["traffic_safety_score"] = (
        0.35 * df["traffic_score"]
        + 0.25 * df["activity_score"]
        + 0.25 * (1 - df["incident_risk"])
        + 0.15 * df["signal_bonus"]
    ).clip(0.01, 1.0)

    return df


# ── Step 1: Load ───────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print(f"[STEP 1] Loading dataset: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Dataset not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, usecols=lambda c: c in USE_COLS)

    missing = [c for c in USE_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns (will use defaults): {missing}")
        for col in missing:
            if col != "Date":
                df[col] = 0.0

    print(f"[OK]   Loaded {len(df):,} rows × {df.shape[1]} columns.")
    return df


# ── Step 2: Parse dates & assign synthetic time_of_day ────────────────────────
def add_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 2] Parsing 'Date' and assigning synthetic time_of_day …")
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")

    # Always use synthetic hours (seed=42) as per spec
    rng = np.random.default_rng(seed=42)
    df["hour"] = rng.integers(0, 24, size=len(df))

    df["time_of_day"] = df["hour"].apply(map_hour_to_period)

    period_counts = df["time_of_day"].value_counts().to_dict()
    print(f"[INFO] Period distribution: {period_counts}")
    return df


# ── Steps 3-6: Overall aggregate ─────────────────────────────────────────────
def compute_area_scores(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 3-6] Per-area aggregate stats and safety scores …")

    agg = (
        df.groupby("Area Name", as_index=False)
        .agg(
            avg_traffic_volume    =("Traffic Volume",               "mean"),
            avg_speed             =("Average Speed",                "mean"),
            avg_congestion        =("Congestion Level",             "mean"),
            avg_pedestrian        =("Pedestrian and Cyclist Count", "mean"),
            avg_incidents         =("Incident Reports",             "mean"),
            avg_signal_compliance =("Traffic Signal Compliance",    "mean"),
        )
    )

    agg = compute_scores(agg)

    out_cols = [
        "Area Name",
        "avg_traffic_volume", "avg_speed", "avg_congestion",
        "avg_pedestrian", "avg_incidents", "avg_signal_compliance",
        "traffic_score", "activity_score", "incident_risk",
        "signal_bonus", "traffic_safety_score",
    ]
    agg = agg[[c for c in out_cols if c in agg.columns]]

    agg.to_csv(OUTPUT_AREA_CSV, index=False)
    print(f"[OK]   Area scores saved → {OUTPUT_AREA_CSV}")
    return agg


# ── Step 7: Night-time specific stats ─────────────────────────────────────────
def compute_night_scores(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 7] Night-time stats (time_of_day == 'night') …")

    night_df = df[df["time_of_day"] == "night"]
    print(f"[INFO] Night rows: {len(night_df):,} / {len(df):,}")

    night_agg = (
        night_df.groupby("Area Name", as_index=False)
        .agg(
            avg_traffic_volume    =("Traffic Volume",               "mean"),
            avg_speed             =("Average Speed",                "mean"),
            avg_congestion        =("Congestion Level",             "mean"),
            avg_pedestrian        =("Pedestrian and Cyclist Count", "mean"),
            avg_incidents         =("Incident Reports",             "mean"),
            avg_signal_compliance =("Traffic Signal Compliance",    "mean"),
        )
    )

    night_agg = compute_scores(night_agg)

    # Rename final score column for night output
    night_agg = night_agg.rename(columns={
        "avg_traffic_volume":   "night_traffic_volume",
        "avg_pedestrian":       "night_pedestrian",
        "avg_incidents":        "night_incidents",
        "traffic_safety_score": "night_traffic_safety_score",
    })

    out_cols = [
        "Area Name",
        "night_traffic_volume", "night_pedestrian", "night_incidents",
        "activity_score", "incident_risk", "signal_bonus",
        "night_traffic_safety_score",
    ]
    night_agg = night_agg[[c for c in out_cols if c in night_agg.columns]]

    night_agg.to_csv(OUTPUT_NIGHT_CSV, index=False)
    print(f"[OK]   Night scores saved → {OUTPUT_NIGHT_CSV}")
    return night_agg


# ── Step 8: Ranked summary table ──────────────────────────────────────────────
def print_ranked_summary(area_scores: pd.DataFrame) -> None:
    ranked = (
        area_scores[["Area Name", "traffic_safety_score", "activity_score", "incident_risk"]]
        .sort_values("traffic_safety_score", ascending=False)
        .reset_index(drop=True)
    )
    ranked.index += 1     # 1-based rank

    col_w = [6, 26, 22, 16, 16]
    header = (
        f"  {'Rank':<{col_w[0]}}"
        f"{'Area Name':<{col_w[1]}}"
        f"{'traffic_safety_score':>{col_w[2]}}"
        f"{'activity_score':>{col_w[3]}}"
        f"{'incident_risk':>{col_w[4]}}"
    )
    sep = "=" * (sum(col_w) + 4)

    print(f"\n{sep}")
    print("  AREA SAFETY SCORE RANKING")
    print(sep)
    print(header)
    print("-" * (sum(col_w) + 4))

    for rank, row in ranked.iterrows():
        ts  = row["traffic_safety_score"]
        act = row["activity_score"]
        inc = row["incident_risk"]
        bar = "█" * int(ts * 18)
        print(
            f"  {rank:<{col_w[0]}}"
            f"{row['Area Name']:<{col_w[1]}}"
            f"{ts:>{col_w[2]}.4f}"
            f"{act:>{col_w[3]}.4f}"
            f"{inc:>{col_w[4]}.4f}"
            f"  {bar}"
        )

    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 72)
    print("  Night Navigator — Traffic Preprocessor v2")
    print("=" * 72)

    df          = load_data()
    df          = add_time_of_day(df)
    area_scores = compute_area_scores(df)
    _           = compute_night_scores(df)

    print_ranked_summary(area_scores)

    print("\n[DONE] Outputs:")
    print(f"  • {OUTPUT_AREA_CSV}")
    print(f"  • {OUTPUT_NIGHT_CSV}")


if __name__ == "__main__":
    main()
