"""
Nova Scotia Wildfire Spread Prediction Model — Training Script
Hackathon demo build

Inputs (at inference time):
    wind speed, wind direction, fuel type, slope, humidity (rh),
    current location (lat/lon), fire border coordinates*

Outputs:
    fire_direction  — potential spread direction in degrees (0–360)
    fire_severity   — normalized fire intensity (0–1)

Labels are derived via physics-based formulas from the Canadian FBP system
rather than direct ground-truth observations (see derive_labels() below).

* Fire border coordinates are a planned input feature but are absent from the
  current training dataset. When available, engineer them as: border centroid
  lat/lon, distance from current position to nearest border point, and bearing
  from current position to fire centroid — then add those columns to FEATURES.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_PATH   = "training_features_ns_2023.csv"
MODEL_OUT   = "wildfire_model.pkl"
ENCODER_OUT = "fuel_label_encoder.pkl"
BOUNDS_OUT  = "hfi_norm_bounds.npy"

FEATURES = [
    "lat", "lon",
    "ws", "wd_sin", "wd_cos",       # wind speed + cyclically-encoded direction
    "fuel_enc",                      # label-encoded fuel type
    "slope",
    "aspect_sin", "aspect_cos",      # cyclically-encoded terrain aspect
    "rh",                            # relative humidity (dryness proxy)
    "temp",
    "ffmc", "dmc", "dc", "bui", "isi",  # FWI system weather indices
    "pcp",
    "elev",
]

# ── 1. Load ────────────────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  {len(df):,} rows, {df.shape[1]} columns")


# ── 2. Derive labels (physics-based) ──────────────────────────────────────────

def derive_fire_direction(wd, aspect, slope, ws):
    """
    Potential fire spread direction from vector combination of wind and upslope
    forces, following Canadian FBP system principles.

    Wind contributes a force scaled by wind speed. Slope contributes an upslope
    force (in the direction of `aspect`) scaled via tanh — this saturates around
    ~15° slope so that very steep terrain doesn't completely override wind.
    The combined vector is converted back to a compass bearing (0–360°).

    Params:
        wd     - wind direction in degrees (meteorological: direction wind comes FROM)
        aspect - terrain aspect in degrees (direction the slope faces / upslope direction)
        slope  - slope magnitude (degrees or radians-ish; treated as raw FBP output)
        ws     - wind speed (m/s or km/h; units cancel in the weighting)

    Returns:
        fire direction in degrees (0–360)
    """
    wd_rad     = np.radians(wd)
    aspect_rad = np.radians(aspect)

    # Slope influence saturates — steep terrain dominates, gentle terrain defers to wind
    slope_factor = np.tanh(slope / 10.0)

    # Wind vector — direction wind blows TO is wd + 180, but fire spreads
    # in the direction the wind is pushing, i.e. downwind = wd as-is
    wind_x = ws * np.sin(wd_rad)
    wind_y = ws * np.cos(wd_rad)

    # Upslope vector weighted by slope factor
    slope_x = slope_factor * np.sin(aspect_rad)
    slope_y = slope_factor * np.cos(aspect_rad)

    # Resultant direction
    combined_x = wind_x + slope_x
    combined_y = wind_y + slope_y
    direction = np.degrees(np.arctan2(combined_x, combined_y)) % 360
    return direction


def derive_fire_severity(hfi):
    """
    Normalize Head Fire Intensity (kW/m) to a 0–1 severity score.

    HFI is heavily right-skewed (a few extreme crown fires dominate), so a
    log1p transform is applied first to compress the tail, then min-max scaled.
    log1p handles zero-HFI rows (non-burning fuel types) cleanly.

    Params:
        hfi - Head Fire Intensity in kW/m

    Returns:
        severity in [0, 1]
    """
    log_hfi = np.log1p(hfi)
    lo, hi  = log_hfi.min(), log_hfi.max()
    return (log_hfi - lo) / (hi - lo), lo, hi


print("Deriving physics-based labels...")
df["fire_direction"] = derive_fire_direction(
    df["wd"], df["aspect"], df["slope"], df["ws"]
)
df["fire_severity"], hfi_log_min, hfi_log_max = derive_fire_severity(df["hfi"])

print(f"  fire_direction : {df['fire_direction'].min():.1f}° – {df['fire_direction'].max():.1f}°")
print(f"  fire_severity  : {df['fire_severity'].min():.3f} – {df['fire_severity'].max():.3f}")


# ── 3. Feature engineering ─────────────────────────────────────────────────────

print("Engineering features...")

# Cyclical encoding — prevents the model treating 359° and 1° as far apart
df["wd_sin"]     = np.sin(np.radians(df["wd"]))
df["wd_cos"]     = np.cos(np.radians(df["wd"]))
df["aspect_sin"] = np.sin(np.radians(df["aspect"]))
df["aspect_cos"] = np.cos(np.radians(df["aspect"]))

# Fuel type — label encode; preserve encoder for inference
le = LabelEncoder()
df["fuel_enc"] = le.fit_transform(df["fuel"].fillna("unknown").astype(str))
print(f"  Fuel classes : {list(le.classes_)}")

TARGETS = ["fire_direction", "fire_severity"]

df_model = df[FEATURES + TARGETS].dropna()
print(f"  {len(df_model):,} usable rows after dropping nulls (dropped {len(df) - len(df_model)})")

X = df_model[FEATURES].values
y = df_model[TARGETS].values


# ── 4. Train / test split ──────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nSplit — train: {len(X_train):,}  test: {len(X_test):,}")


# ── 5. Train ───────────────────────────────────────────────────────────────────

print("Training RandomForest (multi-output)...")

# RandomForest handles multi-output natively and is robust on small datasets.
# With ~2K rows, GradientBoosting would need careful tuning to avoid overfitting;
# RF's bagging gives reasonable variance reduction out of the box.
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, y_train)
print("  Done.")


# ── 6. Evaluate ────────────────────────────────────────────────────────────────

print("\nEvaluation on held-out test set:")
y_pred = model.predict(X_test)

for i, col in enumerate(TARGETS):
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    r2  = r2_score(y_test[:, i], y_pred[:, i])
    print(f"  {col:<20}  MAE={mae:.4f}  R²={r2:.4f}")

# Circular MAE for direction — wraps correctly at the 0/360 boundary
dir_err = np.abs(y_test[:, 0] - y_pred[:, 0])
dir_err = np.minimum(dir_err, 360.0 - dir_err)
print(f"  {'fire_direction':<20}  Circular MAE = {dir_err.mean():.2f}°")


# ── 7. Feature importance ──────────────────────────────────────────────────────

importances = pd.Series(model.feature_importances_, index=FEATURES)
print("\nFeature importances (top 10):")
for feat, imp in importances.sort_values(ascending=False).head(10).items():
    print(f"  {feat:<20}  {imp:.4f}")


# ── 8. Save artifacts ──────────────────────────────────────────────────────────

print("\nSaving artifacts...")
joblib.dump(model, MODEL_OUT)
joblib.dump(le, ENCODER_OUT)
# Save HFI log bounds so inference can apply the exact same normalization
np.save(BOUNDS_OUT, np.array([hfi_log_min, hfi_log_max]))

print(f"  {MODEL_OUT}")
print(f"  {ENCODER_OUT}")
print(f"  {BOUNDS_OUT}")
print("\nDone.")


# ── 9. Inference helper ────────────────────────────────────────────────────────

def predict(
    lat, lon,
    wind_speed, wind_direction,
    fuel, slope, aspect,
    humidity, temp,
    ffmc, dmc, dc, bui, isi,
    pcp, elev,
    model=model, encoder=le,
):
    """
    Run inference for a single observation.

    Params:
        lat, lon          - current position (decimal degrees)
        wind_speed        - m/s or km/h (match training units)
        wind_direction    - degrees, meteorological (direction wind comes FROM)
        fuel              - fuel type string, e.g. "C2", "M1_50", "D1"
        slope             - slope magnitude (FBP units)
        aspect            - terrain aspect in degrees
        humidity          - relative humidity 0–100
        temp              - temperature °C
        ffmc,dmc,dc,bui,isi - FWI system indices
        pcp               - precipitation mm
        elev              - elevation metres

    Returns:
        dict with fire_direction (degrees) and fire_severity (0–1)
    """
    fuel_enc = encoder.transform([fuel])[0] if fuel in encoder.classes_ else encoder.transform(["unknown"])[0]

    row = np.array([[
        lat, lon,
        wind_speed,
        np.sin(np.radians(wind_direction)),
        np.cos(np.radians(wind_direction)),
        fuel_enc,
        slope,
        np.sin(np.radians(aspect)),
        np.cos(np.radians(aspect)),
        humidity, temp,
        ffmc, dmc, dc, bui, isi,
        pcp, elev,
    ]])

    direction, severity = model.predict(row)[0]
    direction = direction % 360  # ensure 0–360

    return {
        "fire_direction": round(float(direction), 1),
        "fire_severity":  round(float(np.clip(severity, 0, 1)), 4),
    }


# Example call (comment out if importing as a module)
if __name__ == "__main__":
    example = predict(
        lat=45.57, lon=-64.27,
        wind_speed=14.5, wind_direction=220,
        fuel="C2", slope=1.87, aspect=286,
        humidity=39, temp=13.1,
        ffmc=87.9, dmc=8.7, dc=18.7, bui=8.6, isi=6.6,
        pcp=0.0, elev=50,
    )
    print(f"\nExample prediction: {example}")
