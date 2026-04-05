"""
Night Navigator — Safety Model Trainer
Trains Random Forest and XGBoost models on real + synthetic edge data.
Outputs trained models, feature importance chart, and SHAP summary.
"""

# ── STEP 0: Install dependencies ──────────────────────────────────────────────
import subprocess, sys

def _pip_install(*pkgs):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *pkgs]
    )

try:
    import sklearn; import xgboost; import shap; import joblib
except ImportError:
    print("[SETUP] Installing dependencies …")
    _pip_install("scikit-learn", "xgboost", "shap", "joblib", "matplotlib")
    print("[SETUP] Dependencies installed.\n")

# ── Standard imports ──────────────────────────────────────────────────────────
import os, json, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor
import shap
import joblib

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE       = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.abspath(os.path.join(_BASE, "data"))
MODELS_DIR  = os.path.abspath(os.path.join(_BASE, "models"))
EVAL_DIR    = os.path.abspath(os.path.join(_BASE, "evaluation"))
EDGES_FILE  = os.path.join(DATA_DIR, "edges_with_safety.geojson")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,   exist_ok=True)

# ── Feature / target columns ──────────────────────────────────────────────────
X_COLS = [
    "poi_norm",            # OSM footfall proxy
    "lamp_norm",          # ward-based lighting density (normalised)
    "road_importance",    # dynamic: highway + traffic + activity
    "pedestrian_norm",    # real pedestrian signal from traffic dataset
    "activity_composite", # liveliness: congestion + pedestrian + traffic
    "incident_risk",      # crime anchor from real Bangalore dataset
    "police_bonus",       # proximity to police stations (0–0.05)
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — LOAD REAL EDGE DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 65)
print("  Night Navigator — Safety Model Trainer")
print("=" * 65)
print("\n[STEP 1] Loading real edge data …")

if not os.path.exists(EDGES_FILE):
    print(f"[FATAL] edges_with_safety.geojson not found at {EDGES_FILE}")
    print("        Run feature_engineering/safety_scorer.py first.")
    sys.exit(1)

gdf = gpd.read_file(EDGES_FILE)
gdf = gdf.drop(columns=["geometry"], errors="ignore")

# Keep only rows where ALL feature columns are present and non-NaN
missing_cols = [c for c in X_COLS if c not in gdf.columns]
if missing_cols:
    print(f"[WARN] Missing feature columns in edge file: {missing_cols}")
    print("       These will be filled with 0.")
    for c in missing_cols:
        gdf[c] = 0.0

df_real = gdf.dropna(subset=X_COLS + ["safety_score"]).copy()
df_real = df_real[X_COLS + ["safety_score"]].reset_index(drop=True)

print(f"[OK]   Total real edges loaded : {len(df_real):,}")
print("\n[INFO] Feature means (real data):")
for col in X_COLS:
    print(f"         {col:<22}: {df_real[col].mean():.4f}")

y_real_reg = df_real["safety_score"]
y_real_clf = (y_real_reg > 0.5).astype(int)

print(f"\n[INFO] safety_score — mean={y_real_reg.mean():.4f}, std={y_real_reg.std():.4f}")
safe_n   = int(y_real_clf.sum())
unsafe_n = len(y_real_clf) - safe_n
print(f"[INFO] Class balance — safe(1): {safe_n:,}  unsafe(0): {unsafe_n:,}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — SYNTHETIC TRAINING DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 2] Generating synthetic training data …")

rng      = np.random.default_rng(42)
N        = 5000

poi      = rng.uniform(0, 1, N)
lamp     = rng.uniform(0, 1, N)
road_imp = rng.uniform(0, 1, N)
ped      = rng.uniform(0, 1, N)
act      = rng.uniform(0, 1, N)
incident = rng.uniform(0, 1, N)
police   = rng.uniform(0, 0.05, N)

# Exact same formula as safety_scorer.py
footfall  = 0.40 * poi + 0.30 * ped + 0.20 * road_imp + 0.10 * act
lighting  = 0.55 * lamp + 0.45 * road_imp
crime     = (0.60 * incident + 0.25 * (1 - lighting) + 0.15 * (1 - footfall)).clip(0, 1)
act_comp  = 0.50 * act + 0.30 * ped + 0.20 * poi   # poi as traffic proxy
raw_score = 0.30 * lighting + 0.30 * footfall + 0.20 * act_comp - 0.10 * crime
raw_score += police

# Soft normalise → add noise → clip (mirrors Step 8 of safety_scorer.py)
raw_score  = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-9)
raw_score += rng.normal(0, 0.03, N)
raw_score  = raw_score.clip(0.01, 0.99)

binary_label = (raw_score > 0.5).astype(int)

df_synth = pd.DataFrame({
    "poi_norm":            poi,
    "lamp_norm":           lamp,
    "road_importance":     road_imp,
    "pedestrian_norm":     ped,
    "activity_composite":  act_comp,
    "incident_risk":       incident,
    "police_bonus":        police,
    "safety_score":        raw_score,
})

# Tag sources and combine
df_real["source"]  = "real"
df_synth["source"] = "synthetic"
df_all = pd.concat([df_real, df_synth], ignore_index=True)

synth_safe   = int(binary_label.sum())
synth_unsafe = N - synth_safe
combined_clf = (df_all["safety_score"] > 0.5).astype(int)
comb_safe    = int(combined_clf.sum())
comb_unsafe  = len(combined_clf) - comb_safe

print(f"[OK]   Total training rows : {len(df_all):,} ({len(df_real):,} real + {N:,} synthetic)")
print(f"[INFO] Synthetic class balance — safe(1): {synth_safe:,}  unsafe(0): {synth_unsafe:,}")
print(f"[INFO] Combined class balance — safe(1): {comb_safe:,}  unsafe(0): {comb_unsafe:,}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 — TRAIN / TEST SPLIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 3] Splitting train / test (80/20) …")

X_all   = df_all[X_COLS].values
y_reg   = df_all["safety_score"].values
y_clf   = (y_reg > 0.5).astype(int)

X_train, X_test, \
y_train_reg, y_test_reg, \
y_train_clf, y_test_clf = train_test_split(
    X_all, y_reg, y_clf,
    test_size=0.20, random_state=42,
    stratify=y_clf,
)

print(f"[OK]   Train size : {len(X_train):,}")
print(f"[OK]   Test  size : {len(X_test):,}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4a — RANDOM FOREST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n━━━ RANDOM FOREST RESULTS ━━━")

# Classifier
print("\n[4a] Training RF Classifier …")
rf_classifier = RandomForestClassifier(
    n_estimators=150, max_depth=12,
    class_weight="balanced",
    random_state=42, n_jobs=-1,
)
rf_classifier.fit(X_train, y_train_clf)
rf_clf_preds = rf_classifier.predict(X_test)
rf_acc = accuracy_score(y_test_clf, rf_clf_preds)
rf_f1  = f1_score(y_test_clf, rf_clf_preds, average="weighted")
print(f"  Accuracy : {rf_acc:.4f}")
print(f"  F1 (wt)  : {rf_f1:.4f}")
print(classification_report(y_test_clf, rf_clf_preds,
                             target_names=["Unsafe(0)", "Safe(1)"],
                             zero_division=0))

# Regressor
print("[4a] Training RF Regressor …")
rf_regressor = RandomForestRegressor(
    n_estimators=150, max_depth=12,
    random_state=42, n_jobs=-1,
)
rf_regressor.fit(X_train, y_train_reg)
rf_reg_preds = rf_regressor.predict(X_test)
rf_rmse = float(np.sqrt(mean_squared_error(y_test_reg, rf_reg_preds)))
rf_mae  = float(mean_absolute_error(y_test_reg, rf_reg_preds))
rf_r2   = float(r2_score(y_test_reg, rf_reg_preds))
print(f"  RMSE : {rf_rmse:.4f}")
print(f"  MAE  : {rf_mae:.4f}")
print(f"  R²   : {rf_r2:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4b — XGBOOST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n━━━ XGBOOST RESULTS ━━━")

# Classifier
print("\n[4b] Training XGBoost Classifier …")
xgb_classifier = XGBClassifier(
    n_estimators=200, max_depth=6,
    learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42, n_jobs=-1,
    verbosity=0,
)
xgb_classifier.fit(X_train, y_train_clf)
xgb_clf_preds = xgb_classifier.predict(X_test)
xgb_acc = accuracy_score(y_test_clf, xgb_clf_preds)
xgb_f1  = f1_score(y_test_clf, xgb_clf_preds, average="weighted")
print(f"  Accuracy : {xgb_acc:.4f}")
print(f"  F1 (wt)  : {xgb_f1:.4f}")
print(classification_report(y_test_clf, xgb_clf_preds,
                             target_names=["Unsafe(0)", "Safe(1)"],
                             zero_division=0))

# Regressor
print("[4b] Training XGBoost Regressor …")
xgb_regressor = XGBRegressor(
    n_estimators=200, max_depth=6,
    learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="rmse",
    random_state=42, n_jobs=-1,
    verbosity=0,
)
xgb_regressor.fit(X_train, y_train_reg)
xgb_reg_preds = xgb_regressor.predict(X_test)
xgb_rmse = float(np.sqrt(mean_squared_error(y_test_reg, xgb_reg_preds)))
xgb_mae  = float(mean_absolute_error(y_test_reg, xgb_reg_preds))
xgb_r2   = float(r2_score(y_test_reg, xgb_reg_preds))
print(f"  RMSE : {xgb_rmse:.4f}")
print(f"  MAE  : {xgb_mae:.4f}")
print(f"  R²   : {xgb_r2:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5 — MODEL COMPARISON TABLE + BEST MODEL SELECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 5] Model comparison:")
print("┌─────────────────┬──────────────────┬──────────────────┐")
print("│ Metric          │  Random Forest   │    XGBoost       │")
print("├─────────────────┼──────────────────┼──────────────────┤")
print(f"│ Regressor RMSE  │ {rf_rmse:^16.4f} │ {xgb_rmse:^16.4f} │")
print(f"│ Regressor MAE   │ {rf_mae:^16.4f} │ {xgb_mae:^16.4f} │")
print(f"│ Regressor R²    │ {rf_r2:^16.4f} │ {xgb_r2:^16.4f} │")
print(f"│ Classifier Acc  │ {rf_acc:^16.4f} │ {xgb_acc:^16.4f} │")
print(f"│ Classifier F1   │ {rf_f1:^16.4f} │ {xgb_f1:^16.4f} │")
print("└─────────────────┴──────────────────┴──────────────────┘")

if rf_rmse <= xgb_rmse:
    best_regressor = rf_regressor
    best_name      = "RF"
else:
    best_regressor = xgb_regressor
    best_name      = "XGBoost"

print(f"\n[INFO] Best model (lowest RMSE): {best_name}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6 — PREDICTION DISTRIBUTION CHECK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 6] Prediction distribution check …")
preds = best_regressor.predict(X_test)
print(f"  Predicted mean={preds.mean():.3f}, std={preds.std():.3f}")
if preds.mean() < 0.35 or preds.mean() > 0.65:
    print("  ⚠ WARNING: predicted mean outside [0.35, 0.65] — model may be skewed")
else:
    print("  ✅ Predicted mean in healthy range")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 7 — FEATURE IMPORTANCE TABLE + CHART
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 7] Feature importance …")

rf_imp  = rf_regressor.feature_importances_
xgb_imp = xgb_regressor.feature_importances_

# Sort by RF importance for display
rf_order = np.argsort(rf_imp)[::-1]

print(f"\n  {'Feature':<22} | {'RF Importance':>13} | {'XGB Importance':>14} | Rank (RF)")
print("  " + "-" * 70)
for rank, idx in enumerate(rf_order, 1):
    feat = X_COLS[idx]
    print(f"  {feat:<22} | {rf_imp[idx]:>13.4f} | {xgb_imp[idx]:>14.4f} | {rank}")

# Grouped bar chart
fig, ax = plt.subplots(figsize=(10, 5))
x      = np.arange(len(X_COLS))
width  = 0.35
bars1  = ax.bar(x - width / 2, rf_imp[rf_order],  width, label="Random Forest",
                color="#4C72B0", alpha=0.88)
bars2  = ax.bar(x + width / 2, xgb_imp[rf_order], width, label="XGBoost",
                color="#DD8452", alpha=0.88)

ax.set_xticks(x)
ax.set_xticklabels([X_COLS[i] for i in rf_order], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Feature Importance")
ax.set_title("Feature Importance: RF vs XGBoost (Regressor)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

importance_path = os.path.join(EVAL_DIR, "feature_importance.png")
plt.savefig(importance_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[OK]   Feature importance chart saved → {importance_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 8 — SHAP ANALYSIS (best model)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n[STEP 8] SHAP analysis on best model ({best_name}) …")

# Sample up to 2000 rows from test set for speed
sample_size  = min(2000, len(X_test))
rng_shap     = np.random.default_rng(0)
sample_idx   = rng_shap.choice(len(X_test), size=sample_size, replace=False)
X_test_sample = X_test[sample_idx]

X_test_df = pd.DataFrame(X_test_sample, columns=X_COLS)

explainer   = shap.TreeExplainer(best_regressor)
shap_values = explainer.shap_values(X_test_df)

shap_path = os.path.join(EVAL_DIR, "shap_summary.png")
plt.figure()
shap.summary_plot(shap_values, X_test_df,
                  feature_names=X_COLS, show=False)
plt.tight_layout()
plt.savefig(shap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK]   SHAP summary plot saved  → {shap_path}")

shap_means = np.abs(shap_values).mean(axis=0)
top3 = sorted(zip(X_COLS, shap_means), key=lambda x: -x[1])[:3]
print("\n  Top 3 features by mean |SHAP value|:")
for rank, (feat, val) in enumerate(top3, 1):
    print(f"    #{rank} {feat:<22}: mean |SHAP| = {val:.4f}")


# ── STEP 8b — Residuals Plot ──────────────────────────────────────────────────
print("\n[STEP 8b] Residuals vs Predicted plot …")

import matplotlib.pyplot as plt
import os

preds_full      = best_regressor.predict(X_test)
residuals       = y_test_reg - preds_full
preds_full_rmse = float(np.sqrt(mean_squared_error(y_test_reg, preds_full)))
r2              = float(r2_score(y_test_reg, preds_full))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Plot 1: Residuals vs Predicted ───────────────────────────────────────────
axes[0].scatter(
    preds_full, residuals,
    alpha=0.15, s=3, color="#2980b9", rasterized=True
)
axes[0].axhline(0, color="red", linewidth=1.2, linestyle="--")
axes[0].set_xlabel("Predicted Safety Score")
axes[0].set_ylabel("Residual (Actual − Predicted)")
axes[0].set_title("Residuals vs Predicted Values")
axes[0].annotate(
    f"RMSE={preds_full_rmse:.4f}\nR²={r2:.4f}",
    xy=(0.05, 0.92), xycoords="axes fraction",
    fontsize=9, color="black",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow")
)

# ── Plot 2: Residual Distribution ────────────────────────────────────────────
axes[1].hist(residuals, bins=80, color="#27ae60", edgecolor="white", linewidth=0.3)
axes[1].axvline(0, color="red", linewidth=1.2, linestyle="--")
axes[1].set_xlabel("Residual Value")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution (should be ~N(0, σ))")
axes[1].annotate(
    f"Mean={residuals.mean():.4f}\nStd={residuals.std():.4f}",
    xy=(0.65, 0.88), xycoords="axes fraction",
    fontsize=9, color="black",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow")
)

plt.tight_layout()
os.makedirs("evaluation", exist_ok=True)
plt.savefig("evaluation/residuals_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK]   Residuals plot saved → evaluation/residuals_plot.png")




# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 9 — SAVE ALL MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[STEP 9] Saving models …")

joblib.dump(rf_classifier,  os.path.join(MODELS_DIR, "rf_classifier.pkl"))
joblib.dump(rf_regressor,   os.path.join(MODELS_DIR, "rf_regressor.pkl"))
joblib.dump(xgb_classifier, os.path.join(   MODELS_DIR, "xgb_classifier.pkl"))
joblib.dump(xgb_regressor,  os.path.join(MODELS_DIR, "xgb_regressor.pkl"))

with open(os.path.join(MODELS_DIR, "feature_cols.json"), "w") as f:
    json.dump(X_COLS, f, indent=2)

print("[DONE] Models saved to models/")
print("  • rf_classifier.pkl")
print("  • rf_regressor.pkl")
print("  • xgb_classifier.pkl")
print("  • xgb_regressor.pkl")
print("  • feature_cols.json")
print(f"\n  Charts saved to evaluation/")
print("  • feature_importance.png")
print("  • shap_summary.png")
