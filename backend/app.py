"""
FastAPI backend for the Turbofan RUL Prediction dashboard.

Reads the pre-computed predictions, uncertainty estimates, and raw sensor
data produced by the ML training pipeline and serves them to the React
frontend via a simple REST API.

All per-engine data is pre-computed at startup and cached in memory so
that API responses are instantaneous.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ML_DIR       = os.path.join(PROJECT_ROOT, "ML")
DATA_DIR     = os.path.join(ML_DIR, "data", "Dataset")
MODEL_DIR    = os.path.join(ML_DIR, "models", "Advanced_CNN_LSTM")
BASELINE_DIR = os.path.join(ML_DIR, "models", "Baseline_LSTM")
RNN_DIR      = os.path.join(ML_DIR, "models", "Baseline_RNN")

# Ensure scripts/ is importable
sys.path.insert(0, os.path.join(ML_DIR, "scripts"))
from data_prep import (
    load_data, SETTING_NAMES, SENSOR_NAMES, INDEX_NAMES, RUL_CAP,
    extract_operating_conditions,
)

# ── CMAPSS Sensor Name Mapping (official documentation) ───────────────
SENSOR_META = {
    "s_1":  {"name": "Fan inlet temperature",       "unit": "°R"},
    "s_2":  {"name": "LPC outlet temperature",      "unit": "°R"},
    "s_3":  {"name": "HPC outlet temperature",      "unit": "°R"},
    "s_4":  {"name": "LPT outlet temperature",      "unit": "°R"},
    "s_5":  {"name": "Fan inlet pressure",           "unit": "psia"},
    "s_6":  {"name": "Bypass-duct pressure",         "unit": "psia"},
    "s_7":  {"name": "HPC outlet pressure",          "unit": "psia"},
    "s_8":  {"name": "Physical fan speed",           "unit": "rpm"},
    "s_9":  {"name": "Physical core speed",          "unit": "rpm"},
    "s_10": {"name": "Engine pressure ratio",        "unit": "—"},
    "s_11": {"name": "HPC outlet static pressure",   "unit": "psia"},
    "s_12": {"name": "Fuel-air ratio",               "unit": "—"},
    "s_13": {"name": "Corrected fan speed",          "unit": "rpm"},
    "s_14": {"name": "Corrected core speed",         "unit": "rpm"},
    "s_15": {"name": "Bypass ratio",                 "unit": "—"},
    "s_16": {"name": "Burner fuel-air ratio",        "unit": "—"},
    "s_17": {"name": "Bleed enthalpy",               "unit": "—"},
    "s_18": {"name": "Demanded fan speed",           "unit": "rpm"},
    "s_19": {"name": "Demanded corrected fan speed", "unit": "rpm"},
    "s_20": {"name": "HPT coolant bleed",            "unit": "lbm/s"},
    "s_21": {"name": "LPT coolant bleed",            "unit": "lbm/s"},
}

# Sensors that INCREASE as the engine degrades (need to be inverted for HI)
# Determined empirically from CMAPSS FD004 degradation trends:
# - Temperatures generally rise as components degrade
# - Corrected speeds generally decrease
SENSORS_INVERT_FOR_HI = {
    "s_2", "s_3", "s_4",           # Temperatures that rise with degradation
    "s_11", "s_15", "s_17",        # Pressures / ratios that rise
}

# ── App setup ──────────────────────────────────────────────────────────
app = FastAPI(title="RUL Predict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup validation ────────────────────────────────────────────────
def _require(path, hint=""):
    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing required file: {path}\n"
            f"Run the ML training pipeline first: python ML/training/main.py\n{hint}"
        )


_require(os.path.join(MODEL_DIR, "predictions.npy"))
_require(os.path.join(MODEL_DIR, "true_values.npy"))
_require(os.path.join(MODEL_DIR, "uncertainty.npy"))
_require(os.path.join(MODEL_DIR, "metrics.txt"))
_require(os.path.join(DATA_DIR, "test_FD004.txt"))
_require(os.path.join(DATA_DIR, "train_FD004.txt"),
         "Training data is needed to detect constant sensors.")


# ── Load pre-computed ML results ───────────────────────────────────────
predictions = np.load(os.path.join(MODEL_DIR, "predictions.npy"))  # real-cycle RUL
true_values = np.load(os.path.join(MODEL_DIR, "true_values.npy"))  # real-cycle true RUL
uncertainty = np.load(os.path.join(MODEL_DIR, "uncertainty.npy"))  # real-cycle std (epistemic)

# ── Load raw data ──────────────────────────────────────────────────────
raw_test = load_data(os.path.join(DATA_DIR, "test_FD004.txt"))
raw_train = load_data(os.path.join(DATA_DIR, "train_FD004.txt"))
true_rul_df = pd.read_csv(
    os.path.join(DATA_DIR, "RUL_FD004.txt"),
    sep=r"\s+", header=None, names=["RUL"],
)

# Detect constant sensors using training data (same as training pipeline)
available_sensors = [s for s in SENSOR_NAMES if s in raw_train.columns]
std_dev = raw_train[available_sensors].std()
constant_sensors = std_dev[std_dev < 1e-5].index.tolist()
active_sensors = [s for s in available_sensors if s not in constant_sensors]
print(f"[*] Active sensors ({len(active_sensors)}): {active_sensors}")

# Extract operating conditions
raw_test_with_domain, _ = extract_operating_conditions(raw_test.copy(), n_clusters=6)

# Figure out which engines have enough data for the model (seq_length=50)
SEQUENCE_LENGTH = 50
engine_ids = raw_test["unit_nr"].unique()


# ── Metrics ────────────────────────────────────────────────────────────
def parse_metrics(filepath):
    metrics = {}
    if not os.path.exists(filepath):
        return metrics
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line and not line.startswith("="):
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().replace(" %", "")
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = val
    return metrics


parsed_metrics = parse_metrics(os.path.join(MODEL_DIR, "metrics.txt"))
baseline_metrics = parse_metrics(os.path.join(BASELINE_DIR, "metrics.txt"))
rnn_metrics = parse_metrics(os.path.join(RNN_DIR, "metrics.txt"))

model_config = {}
config_path = os.path.join(MODEL_DIR, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        model_config = json.load(f)


# ── Helper functions ───────────────────────────────────────────────────
def compute_status(mean_rul, std_rul, threshold=15):
    conservative = mean_rul - 1.5 * std_rul
    if conservative <= threshold:
        return "red"
    elif mean_rul <= threshold * 2:
        return "amber"
    else:
        return "green"


def rolling_mean(arr, window=15):
    """Simple rolling average to smooth noisy signals."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    # Use 'valid' mode and pad the front to maintain length
    smoothed = np.convolve(arr, kernel, mode="valid")
    # Pad front with the first few raw values
    pad = arr[: len(arr) - len(smoothed)]
    return np.concatenate([pad, smoothed])


def compute_health_index(engine_df, sensors):
    """
    Compute a monotonically-decreasing health index from sensor data.

    Strategy:
    1. Min-max normalise each sensor within this engine's range to [0, 1].
    2. Invert sensors that are known to INCREASE with degradation so that
       all normalised signals decrease as the engine degrades.
    3. Average all normalised signals.
    4. Apply rolling mean smoothing for visual clarity.
    """
    engine_sensors = engine_df[sensors].copy()

    for col in sensors:
        cmin, cmax = engine_sensors[col].min(), engine_sensors[col].max()
        if cmax > cmin:
            normalised = (engine_sensors[col] - cmin) / (cmax - cmin)
        else:
            normalised = pd.Series(0.5, index=engine_sensors.index)

        # Invert sensors that rise with degradation
        if col in SENSORS_INVERT_FOR_HI:
            normalised = 1.0 - normalised

        engine_sensors[col] = normalised

    # Average across all sensors → single health index per cycle
    hi_raw = engine_sensors.mean(axis=1).values

    # Smooth with rolling average
    hi_smooth = rolling_mean(hi_raw, window=15)

    return hi_smooth


def compute_sensor_telemetry(engine_df, sensors):
    """Build the sensor table data for an engine."""
    last_row = engine_df.iloc[-1]
    n_rows = len(engine_df)

    # Recent window for drift calculation (last 50 cycles vs previous 50)
    recent_window = min(50, n_rows // 3)

    sensor_list = []
    for sensor_id in sensors:
        meta = SENSOR_META.get(sensor_id, {"name": sensor_id, "unit": ""})
        col_data = engine_df[sensor_id]
        current_val = float(last_row[sensor_id])

        # Normalised position within operating range
        s_min, s_max = float(col_data.min()), float(col_data.max())
        norm = (current_val - s_min) / (s_max - s_min) if s_max > s_min else 0.5

        # Drift: recent window mean vs earlier window mean
        if n_rows > recent_window * 2:
            recent_mean = float(col_data.iloc[-recent_window:].mean())
            earlier_mean = float(col_data.iloc[-recent_window * 2:-recent_window].mean())
            drift = ((recent_mean - earlier_mean) / abs(earlier_mean) * 100) if earlier_mean != 0 else 0.0
        else:
            # Fallback: last value vs first value
            first_val = float(col_data.iloc[0])
            drift = ((current_val - first_val) / abs(first_val) * 100) if first_val != 0 else 0.0

        sensor_list.append({
            "id": sensor_id,
            "name": meta["name"],
            "val": round(current_val, 2),
            "unit": meta["unit"],
            "norm": round(max(0.0, min(1.0, norm)), 3),
            "drift": round(drift, 1),
        })

    return sensor_list


# ── Pre-compute all engine data at startup ─────────────────────────────
print("[*] Pre-computing engine data for all test engines...")

# Build a mapping: engine_index (0-based, matching predictions array) → engine data
fleet_data = []          # List of fleet summary items
engine_detail_cache = {} # index → full detail dict
engine_index_by_id = {}  # unit_nr → index (for lookup by engine ID)

pred_index = 0  # Tracks position in the predictions array
skipped_engines = []

for eid in engine_ids:
    engine_df = raw_test[raw_test["unit_nr"] == eid]
    total_cycles = int(engine_df["time_cycles"].max())

    # Check if this engine was included in model evaluation
    if len(engine_df) < SEQUENCE_LENGTH:
        skipped_engines.append(int(eid))
        continue

    if pred_index >= len(predictions):
        break

    idx = pred_index
    pred_index += 1

    pred_rul  = float(predictions[idx])
    std_rul   = float(uncertainty[idx])
    true_rul  = float(true_values[idx])
    status    = compute_status(pred_rul, std_rul)
    conservative = pred_rul - 1.5 * std_rul

    engine_index_by_id[int(eid)] = idx

    # Fleet summary
    fleet_data.append({
        "id": f"FD004-{int(eid):03d}",
        "engineId": int(eid),
        "index": idx,
        "rul": round(pred_rul, 1),
        "trueRul": round(true_rul, 1),
        "uncertainty": round(std_rul, 1),
        "conservativeRul": round(conservative, 1),
        "status": status,
    })

    # ── Sensor telemetry ───────────────────────────────────────────
    sensors = compute_sensor_telemetry(engine_df, active_sensors)

    # ── Health index trajectory ────────────────────────────────────
    health_index = compute_health_index(engine_df, active_sensors)
    cycles = engine_df["time_cycles"].values.tolist()

    # ── Predicted future trajectory ────────────────────────────────
    failure_threshold = 0.3
    current_health = float(health_index[-1]) if len(health_index) > 0 else 0.5
    steps_to_failure = max(int(pred_rul), 10)

    pred_cycles = []
    pred_health = []
    ci_upper = []
    ci_lower = []
    for step in range(steps_to_failure + 1):
        t = total_cycles + step
        # Linear decay from current health toward failure threshold
        h = current_health - (current_health - failure_threshold) * (step / steps_to_failure)
        pred_cycles.append(t)
        pred_health.append(round(h, 4))
        # Uncertainty fan grows with prediction horizon
        u = (std_rul / RUL_CAP) * (step / steps_to_failure)
        ci_upper.append(round(h + u, 4))
        ci_lower.append(round(h - u * 1.5, 4))

    # ── Uncertainty (honest: MC Dropout = epistemic only) ──────────
    uncertainty_data = {
        "epistemic": {
            "label": "Epistemic (MC Dropout)",
            "value": round(std_rul, 1),
            "rangeStart": round(std_rul * 0.3, 1),
            "rangeEnd": round(std_rul * 2.0, 1),
        },
        "predictive": {
            "label": "Predictive Interval",
            "value": round(std_rul * 1.96, 1),
            "rangeStart": round(std_rul * 0.5, 1),
            "rangeEnd": round(std_rul * 3.5, 1),
        },
        "conservative": {
            "label": "Conservative Est.",
            "value": round(max(0, conservative), 1),
            "rangeStart": 0,
            "rangeEnd": round(pred_rul, 1),
        },
    }

    # ── Domain info ────────────────────────────────────────────────
    engine_with_domain = raw_test_with_domain[raw_test_with_domain["unit_nr"] == eid]
    domain_id = int(engine_with_domain["domain"].mode().iloc[0]) if len(engine_with_domain) > 0 else 0

    # Cache full detail
    engine_detail_cache[idx] = {
        "id": f"FD004-{int(eid):03d}",
        "engineId": int(eid),
        "index": idx,
        "rul": round(pred_rul, 1),
        "trueRul": round(true_rul, 1),
        "uncertainty": round(std_rul, 1),
        "conservativeRul": round(conservative, 1),
        "status": status,
        "totalCycles": total_cycles,
        "sensors": sensors,
        "trajectory": {
            "cycles": cycles,
            "healthIndex": [round(float(h), 4) for h in health_index],
            "failureThreshold": failure_threshold,
        },
        "prediction": {
            "cycles": pred_cycles,
            "healthIndex": pred_health,
            "ciUpper": ci_upper,
            "ciLower": ci_lower,
        },
        "uncertaintyDecomposition": uncertainty_data,
        "domain": {
            "id": domain_id,
            "totalDomains": 6,
        },
    }

if skipped_engines:
    print(f"[!] Skipped {len(skipped_engines)} engines with < {SEQUENCE_LENGTH} cycles: {skipped_engines}")

print(f"[*] Pre-computed data for {len(fleet_data)} engines. Ready to serve.")


# ======================================================================
#  ENDPOINTS
# ======================================================================

@app.get("/api/fleet")
def get_fleet():
    """Return all test engines with predicted RUL and status."""
    return {"fleet": fleet_data, "total": len(fleet_data), "skipped": len(skipped_engines)}


@app.get("/api/engine/{engine_index}")
def get_engine_detail(engine_index: int):
    """Return detailed data for a single engine by its 0-based index."""
    if engine_index not in engine_detail_cache:
        return {"error": f"Engine index {engine_index} not found. Valid range: 0–{len(engine_detail_cache)-1}"}
    return engine_detail_cache[engine_index]


@app.get("/api/metrics")
def get_metrics():
    """Return model performance metrics and config."""
    return {
        "advanced": parsed_metrics,
        "baseline": baseline_metrics,
        "rnn": rnn_metrics,
        "config": model_config,
        "comparison": {
            "models": ["Baseline RNN", "Baseline LSTM", "Advanced CNN-LSTM"],
            "rmse": [rnn_metrics.get("RMSE", 0), baseline_metrics.get("RMSE", 0), parsed_metrics.get("RMSE", 0)],
            "nasa": [rnn_metrics.get("NASA Score", 0), baseline_metrics.get("NASA Score", 0), parsed_metrics.get("NASA Score", 0)],
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
