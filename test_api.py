"""
test_api.py - TransactionGuard API tests
==========================================
Tests the Flask API without requiring a running server (uses Flask test client).
Also tests model loading independently.

Run from the project root (with .venv active):
    python test_api.py

Exit code 0 = all tests passed.
Exit code 1 = one or more tests failed.
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal test harness (no pytest dependency)
# ---------------------------------------------------------------------------

_results = {"passed": 0, "failed": 0, "errors": []}


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        _results["passed"] += 1
        print(f"  PASS  {name}")
    else:
        _results["failed"] += 1
        msg = f"  FAIL  {name}" + (f" -- {detail}" if detail else "")
        _results["errors"].append(msg)
        print(msg)


def section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


# ---------------------------------------------------------------------------
# Import app
# ---------------------------------------------------------------------------

try:
    import app as flask_app
except ImportError as exc:
    print(f"Cannot import app.py: {exc}")
    sys.exit(1)

client = flask_app.app.test_client()
flask_app.app.config["TESTING"] = True

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "transaction_models.joblib"

# ---------------------------------------------------------------------------
# 1. Health endpoint
# ---------------------------------------------------------------------------

section("GET /api/health")

resp = client.get("/api/health")
data = resp.get_json()

check("HTTP 200", resp.status_code == 200)
check("Returns JSON object", isinstance(data, dict))
check("Has 'status' key", "status" in data)
check("Has 'dataset_loaded' key", "dataset_loaded" in data)
check("Has 'models_trained' key", "models_trained" in data)
check("'status' is string", isinstance(data.get("status"), str))

# ---------------------------------------------------------------------------
# 2. Stats endpoint
# ---------------------------------------------------------------------------

section("GET /api/stats")

resp = client.get("/api/stats")
data = resp.get_json()

if resp.status_code == 200:
    check("HTTP 200", True)
    check("Has 'dataset' key", "dataset" in data)
    check("Dataset total_transactions is int", isinstance(data["dataset"].get("total_transactions"), int))
    check("Dataset fraud_percentage is number",
          isinstance(data["dataset"].get("fraud_percentage"), (int, float)))
    check("Has 'models' key", "models" in data)
else:
    check("HTTP 200 or 503", resp.status_code in (200, 503),
          f"Got {resp.status_code}")

# ---------------------------------------------------------------------------
# 3. Dashboard route
# ---------------------------------------------------------------------------

section("GET / (dashboard)")

resp = client.get("/")
check("HTTP 200", resp.status_code == 200)
check("HTML content-type", "text/html" in resp.content_type)
check("Contains brand name", b"TransactionGuard" in resp.data)
check("Contains Fraud Detection", b"Fraud Detection" in resp.data)

# ---------------------------------------------------------------------------
# 4. Predict endpoint - no model
# ---------------------------------------------------------------------------

section("POST /api/predict - various error cases")

# 4a. Wrong content type
resp = client.post("/api/predict", data="not json", content_type="text/plain")
check("Non-JSON body returns 400 or 503", resp.status_code in (400, 503))

# 4b. Empty JSON object
resp = client.post(
    "/api/predict",
    data=json.dumps({}),
    content_type="application/json",
)
check("Empty JSON returns 400 or 503", resp.status_code in (400, 503))

# 4c. Class field included (should be rejected)
payload_with_class = {f: 0.0 for f in ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]}
payload_with_class["Class"] = 1  # must not be accepted
resp = client.post(
    "/api/predict",
    data=json.dumps(payload_with_class),
    content_type="application/json",
)
check("Payload with Class field is rejected (400 or 503)", resp.status_code in (400, 503))

# ---------------------------------------------------------------------------
# 5. Predict endpoint - valid payload (only if models exist)
# ---------------------------------------------------------------------------

section("POST /api/predict - valid payload")

if not MODEL_PATH.exists():
    print("  SKIP  Models not trained yet. Run: python train_models.py")
    print("        (These tests will pass after training.)")
else:
    valid_payload = {f: 0.0 for f in ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]}
    valid_payload["Amount"] = 25.0
    valid_payload["Time"] = 1000.0

    resp = client.post(
        "/api/predict",
        data=json.dumps(valid_payload),
        content_type="application/json",
    )
    data = resp.get_json()

    check("HTTP 200", resp.status_code == 200, f"Got {resp.status_code}: {data}")
    check("Has 'random_forest' key", "random_forest" in data)
    check("Has 'isolation_forest' key", "isolation_forest" in data)
    check("RF prediction is string", isinstance(data.get("random_forest", {}).get("prediction"), str))
    check("RF fraud_score is float", isinstance(data.get("random_forest", {}).get("fraud_score"), float))
    check("IF anomaly_score is float", isinstance(data.get("isolation_forest", {}).get("anomaly_score"), float))
    check("Has disclaimer", "disclaimer" in data)
    check("Has most_unusual_features", "most_unusual_features" in data)

    # 5b. Negative Amount
    bad_payload = dict(valid_payload)
    bad_payload["Amount"] = -10.0
    resp = client.post(
        "/api/predict",
        data=json.dumps(bad_payload),
        content_type="application/json",
    )
    check("Negative Amount returns 400", resp.status_code == 400)

    # 5c. Non-numeric feature
    bad_payload2 = dict(valid_payload)
    bad_payload2["V1"] = "not_a_number"
    resp = client.post(
        "/api/predict",
        data=json.dumps(bad_payload2),
        content_type="application/json",
    )
    check("Non-numeric feature returns 400", resp.status_code == 400)

# ---------------------------------------------------------------------------
# 6. Model bundle integrity (if trained)
# ---------------------------------------------------------------------------

section("Model bundle integrity")

if not MODEL_PATH.exists():
    print("  SKIP  models/transaction_models.joblib not found.")
    print("        Run: python train_models.py  then re-run this script.")
else:
    import joblib
    bundle = joblib.load(MODEL_PATH)

    check("Bundle is a dict", isinstance(bundle, dict))
    check("Has 'random_forest'", "random_forest" in bundle)
    check("Has 'isolation_forest'", "isolation_forest" in bundle)
    check("Has 'features'", "features" in bundle)
    check("Has 'medians'", "medians" in bundle)
    check("Has 'scales'", "scales" in bundle)
    check("Has 'rf_metrics'", "rf_metrics" in bundle)
    check("Has 'iso_metrics'", "iso_metrics" in bundle)
    check("features list length is 30", len(bundle["features"]) == 30)
    check("Class not in features", "Class" not in bundle["features"])

    rf = bundle["random_forest"]
    check("RF has predict method", hasattr(rf, "predict"))
    check("RF has predict_proba method", hasattr(rf, "predict_proba"))
    check("RF classes are [0, 1]", list(rf.classes_) == [0, 1])

    iso = bundle["isolation_forest"]
    check("IF has predict method", hasattr(iso, "predict"))
    check("IF has decision_function method", hasattr(iso, "decision_function"))

    rf_m = bundle["rf_metrics"]
    check("RF precision in [0, 1]", 0.0 <= rf_m["precision"] <= 1.0)
    check("RF recall in [0, 1]", 0.0 <= rf_m["recall"] <= 1.0)
    check("RF F1 in [0, 1]", 0.0 <= rf_m["f1_score"] <= 1.0)
    check("RF avg_precision in [0, 1]", 0.0 <= rf_m["average_precision"] <= 1.0)
    check("RF confusion matrix is 2x2",
          isinstance(rf_m["confusion_matrix"], list) and len(rf_m["confusion_matrix"]) == 2)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section("Results")
total = _results["passed"] + _results["failed"]
print(f"  {_results['passed']} / {total} tests passed")

if _results["errors"]:
    print("\n  Failed tests:")
    for msg in _results["errors"]:
        print(f"    {msg}")
    sys.exit(1)
else:
    print("\n  All tests passed.")
    sys.exit(0)
