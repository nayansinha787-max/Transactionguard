"""
app.py - TransactionGuard Flask application
=============================================
Routes:
  GET  /              - Dashboard (HTML)
  GET  /api/health    - Application and model status
  GET  /api/stats     - Dataset and model evaluation statistics
  POST /api/predict   - Transaction fraud screening
"""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# App and path setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "transaction_models.joblib"

DATA_FULL_PATH = DATA_DIR / "creditcard_full.csv"
DATA_SAMPLE_PATH = DATA_DIR / "creditcard.csv"
DATA_PATH = DATA_FULL_PATH if DATA_FULL_PATH.exists() else DATA_SAMPLE_PATH

# Class is never a prediction input feature
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
REQUIRED_COLUMNS = FEATURES + ["Class"]


# ---------------------------------------------------------------------------
# Dataset loader (cached per process - reads CSV once)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_dataset():
    """Load and validate the credit-card dataset. Returns (df, error_string)."""
    if not DATA_PATH.exists():
        return None, f"Dataset not found at {DATA_PATH}. Put creditcard.csv in data/."

    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return None, f"Could not read dataset: {exc}"

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, "Missing required columns: " + ", ".join(missing)

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[REQUIRED_COLUMNS].isna().any().any():
        return None, "Required columns contain missing or non-numeric values."
    if df.empty:
        return None, "Dataset is empty."
    if not df["Class"].isin([0, 1]).all():
        return None, "Class column must contain only 0 and 1."

    return df, None


# ---------------------------------------------------------------------------
# Model loader (re-reads file each request so hot-reload works after training)
# ---------------------------------------------------------------------------
def get_model_bundle():
    """Load the trained model bundle from disk. Returns bundle dict or None."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except (OSError, ValueError, EOFError):
        app.logger.exception("Could not load trained model bundle")
        return None


# ---------------------------------------------------------------------------
# Route: Dashboard
# ---------------------------------------------------------------------------
@app.route("/")
def dashboard():
    df, error = load_dataset()
    stats = None

    if df is not None:
        total = len(df)
        fraud_count = int((df["Class"] == 1).sum())
        stats = {
            "total": total,
            "fraud": fraud_count,
            "genuine": int((df["Class"] == 0).sum()),
            "fraud_rate": round(fraud_count / total * 100, 6) if total else 0.0,
            "average_amount": float(df["Amount"].mean()),
            "total_columns": len(df.columns),
            "missing_cells": int(df[REQUIRED_COLUMNS].isna().sum().sum()),
            "dataset_file": DATA_PATH.name,
        }

    bundle = get_model_bundle()
    model_status = {
        "trained": bundle is not None,
        "random_forest": bundle is not None,
        "isolation_forest": bundle is not None,
        "rf_metrics": bundle.get("rf_metrics") if bundle else None,
        "iso_metrics": bundle.get("iso_metrics") if bundle else None,
        "dataset_name": bundle.get("dataset_name", "") if bundle else "",
        "train_rows": bundle.get("train_rows", 0) if bundle else 0,
    }

    return render_template("index.html", stats=stats, error=error, model_status=model_status)


@app.get("/favicon.ico")
def favicon():
    return "", 204


# ---------------------------------------------------------------------------
# Route: Health check
# ---------------------------------------------------------------------------
@app.get("/api/health")
def api_health():
    df, ds_error = load_dataset()
    bundle = get_model_bundle()
    return jsonify({
        "status": "ok" if (ds_error is None and bundle is not None) else "needs_attention",
        "dataset_loaded": df is not None,
        "dataset_error": ds_error,
        "models_trained": bundle is not None,
        "model_path": str(MODEL_PATH) if bundle else None,
    })


# ---------------------------------------------------------------------------
# Route: Dataset + model statistics
# ---------------------------------------------------------------------------
@app.get("/api/stats")
def api_stats():
    df, error = load_dataset()
    if error:
        return jsonify({"error": error}), 503

    fraud_count = int((df["Class"] == 1).sum())
    total = len(df)

    payload = {
        "dataset": {
            "file": DATA_PATH.name,
            "total_transactions": total,
            "genuine_transactions": int((df["Class"] == 0).sum()),
            "labeled_fraud_cases": fraud_count,
            "fraud_percentage": round(fraud_count / total * 100, 6) if total else 0.0,
            "average_amount": round(float(df["Amount"].mean()), 4),
            "columns": len(df.columns),
        },
        "models": None,
    }

    bundle = get_model_bundle()
    if bundle:
        payload["models"] = {
            "trained": True,
            "trained_on": bundle.get("dataset_name"),
            "train_rows": bundle.get("train_rows"),
            "test_rows": bundle.get("test_rows"),
            "random_forest": bundle.get("rf_metrics"),
            "isolation_forest": bundle.get("iso_metrics"),
        }
    else:
        payload["models"] = {
            "trained": False,
            "message": "Run 'python train_models.py' to train the models.",
        }

    return jsonify(payload)


# ---------------------------------------------------------------------------
# Route: Transaction prediction
# ---------------------------------------------------------------------------
@app.post("/api/predict")
def api_predict():
    """
    POST JSON: {"Time": 0, "V1": 0, ..., "V28": 0, "Amount": 25.5}
    Do NOT send the Class field - it is the target label, not an input.
    Predictions are screening signals, not proof of fraud.
    """
    bundle = get_model_bundle()
    if bundle is None:
        return jsonify({
            "error": "Models are not trained yet.",
            "fix": "Run: python train_models.py  then restart Flask.",
        }), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object containing the transaction features."}), 400

    # Validate all features are present
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify({"error": "Missing features: " + ", ".join(missing)}), 400

    # Reject Class if accidentally sent
    if "Class" in payload:
        return jsonify({"error": "Do not send the Class field - it is the target label, not an input."}), 400

    # Parse and validate numeric values
    try:
        row = {f: float(payload[f]) for f in FEATURES}
    except (TypeError, ValueError):
        return jsonify({"error": "Every feature value must be numeric."}), 400

    if not all(np.isfinite(v) for v in row.values()):
        return jsonify({"error": "Feature values must be finite numbers (no NaN or Inf)."}), 400
    if row["Amount"] < 0 or row["Time"] < 0:
        return jsonify({"error": "Time and Amount must not be negative."}), 400

    # Build input DataFrame preserving feature order
    X = pd.DataFrame([row], columns=FEATURES)

    rf = bundle["random_forest"]
    isolation = bundle["isolation_forest"]

    # --- Random Forest prediction ---
    rf_class = int(rf.predict(X)[0])
    fraud_idx = list(rf.classes_).index(1)
    fraud_score = float(rf.predict_proba(X)[0][fraud_idx])

    # --- Isolation Forest prediction ---
    anomaly_label = int(isolation.predict(X)[0])   # -1 = outlier, 1 = inlier
    anomaly_score = float(isolation.decision_function(X)[0])

    # --- Descriptive feature deviations (from training-set medians/stds) ---
    medians = bundle["medians"]
    scales = bundle["scales"]
    deviations = []
    for feat in FEATURES:
        scale = float(scales.get(feat, 0.0))
        if scale > 0:
            z = abs((row[feat] - float(medians[feat])) / scale)
            deviations.append({
                "feature": feat,
                "your_value": round(row[feat], 4),
                "typical_value": round(float(medians[feat]), 4),
                "absolute_z_score": round(z, 3),
            })
    deviations.sort(key=lambda d: d["absolute_z_score"], reverse=True)

    return jsonify({
        "random_forest": {
            "prediction": "flag_for_review" if rf_class == 1 else "not_flagged",
            "predicted_class": rf_class,
            "fraud_score": round(fraud_score, 6),
            "note": (
                "Supervised classifier trained on labeled examples. "
                "Score is a model probability, not a calibrated guarantee."
            ),
        },
        "isolation_forest": {
            "prediction": "anomaly_flagged" if anomaly_label == -1 else "not_flagged_as_anomaly",
            "is_anomaly": anomaly_label == -1,
            "anomaly_score": round(anomaly_score, 6),
            "note": (
                "Unsupervised detector. Lower decision_function values are more anomalous. "
                "This is not a fraud probability."
            ),
        },
        "most_unusual_features": deviations[:5],
        "disclaimer": (
            "These outputs are SCREENING SIGNALS for human review - not proof of fraud. "
            "V1-V28 are anonymized PCA features with no direct real-world meaning. "
            "This is a student ML project, not a production payment system."
        ),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
