"""
train_models.py – TransactionGuard
====================================
Trains a Random Forest classifier and an Isolation Forest anomaly detector
on the credit-card fraud detection dataset.

Run from the project root:
    python train_models.py

The full dataset is preferred; the sample is used as a fallback.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)  # create models/ automatically

FULL_PATH = DATA_DIR / "creditcard_full.csv"
SAMPLE_PATH = DATA_DIR / "creditcard.csv"
DATA_PATH = FULL_PATH if FULL_PATH.exists() else SAMPLE_PATH

# Class is excluded from feature list (used only as the prediction target)
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
REQUIRED = FEATURES + ["Class"]

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str) -> None:
    """Print a prominent section separator."""
    width = 65
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found at: {DATA_PATH}\n"
            "Put creditcard.csv inside the data/ folder.\n"
            "(Optional) Full dataset: data/creditcard_full.csv"
        )

    print(f"\nLoading dataset: {DATA_PATH.name}  ({DATA_PATH.stat().st_size / 1_048_576:.1f} MB)")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # ------------------------------------------------------------------
    # 2. Validate schema
    # ------------------------------------------------------------------
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError("Missing required columns: " + ", ".join(missing_cols))

    df = df[REQUIRED].apply(pd.to_numeric, errors="coerce")

    if df.isna().any().any():
        bad = df.isna().sum()
        raise ValueError(
            "Dataset contains missing or non-numeric values in required columns:\n"
            + bad[bad > 0].to_string()
        )

    if df.empty:
        raise ValueError("Dataset is empty after loading.")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError(
            "Class column must contain only 0 (genuine) and 1 (fraud).\n"
            f"Found values: {sorted(df['Class'].unique())}"
        )

    # ------------------------------------------------------------------
    # 3. Split features from target
    # ------------------------------------------------------------------
    X = df[FEATURES]
    y = df["Class"].astype(int)

    total = len(df)
    fraud_count = int(y.sum())
    genuine_count = int((y == 0).sum())
    fraud_pct = fraud_count / total * 100 if total else 0.0

    _sep("Dataset summary")
    print(f"  Total rows       : {total:>10,}")
    print(f"  Genuine (class 0): {genuine_count:>10,}")
    print(f"  Fraud   (class 1): {fraud_count:>10,}  ({fraud_pct:.4f} %)")
    print(f"  Feature columns  : {len(FEATURES)}")

    # ------------------------------------------------------------------
    # 4. Stratified train / test split (no leakage: split before any fit)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train set: {len(X_train):,} rows  |  Test set: {len(X_test):,} rows  (stratified 80/20)")

    # ------------------------------------------------------------------
    # 5. Random Forest - supervised classifier
    # ------------------------------------------------------------------
    _sep("Training Random Forest (supervised)")
    print("  class_weight='balanced'  n_estimators=200  random_state=42")
    print("  (Training may take 30-90 s on the full dataset...)")

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",   # handles class imbalance without dropping data
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    # Safely find the probability column for class=1
    fraud_col_idx = list(rf.classes_).index(1)
    rf_prob = rf.predict_proba(X_test)[:, fraud_col_idx]

    _sep("Random Forest - evaluation on held-out test set")
    print(classification_report(
        y_test, rf_pred,
        labels=[0, 1],
        target_names=["genuine", "fraud"],
        zero_division=0,
    ))
    print("Confusion matrix (rows = true label, columns = predicted label):")
    cm = confusion_matrix(y_test, rf_pred, labels=[0, 1])
    print(f"  {'':15}  predicted_genuine  predicted_fraud")
    print(f"  {'true_genuine':15}  {cm[0, 0]:>17,}  {cm[0, 1]:>14,}")
    print(f"  {'true_fraud':15}  {cm[1, 0]:>17,}  {cm[1, 1]:>14,}")
    rf_precision = precision_score(y_test, rf_pred, zero_division=0)
    rf_recall    = recall_score(y_test, rf_pred, zero_division=0)
    rf_f1        = f1_score(y_test, rf_pred, zero_division=0)
    rf_ap        = average_precision_score(y_test, rf_prob)
    print(f"\n  Fraud precision    : {rf_precision:.4f}")
    print(f"  Fraud recall       : {rf_recall:.4f}")
    print(f"  Fraud F1-score     : {rf_f1:.4f}")
    print(f"  Average precision  : {rf_ap:.4f}  (PR-AUC summary; higher is better)")

    rf_metrics = {
        "precision": round(rf_precision, 6),
        "recall":    round(rf_recall, 6),
        "f1_score":  round(rf_f1, 6),
        "average_precision": round(rf_ap, 6),
        "confusion_matrix": cm.tolist(),
        "test_size": len(y_test),
    }

    # ------------------------------------------------------------------
    # 6. Isolation Forest - unsupervised anomaly detector
    # ------------------------------------------------------------------
    _sep("Training Isolation Forest (unsupervised - no labels used during fit)")
    print("  contamination='auto'  n_estimators=200  random_state=42")
    print("  (Class labels are NOT passed to fit(); used only for evaluation below.)")

    isolation = IsolationForest(
        n_estimators=200,
        contamination="auto",  # avoids choosing the anomaly rate from test labels
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    isolation.fit(X_train)  # deliberately unsupervised - y_train never passed

    iso_raw = isolation.predict(X_test)           # -1 = outlier, 1 = inlier
    iso_pred = (iso_raw == -1).astype(int)        # convert to 0/1 fraud labels

    _sep("Isolation Forest - evaluation (labels used ONLY here, not during training)")
    print("  Note: Isolation Forest is unsupervised. The evaluation below uses")
    print("  the test labels for reference only, not to tune the model.")
    print()
    print(classification_report(
        y_test, iso_pred,
        labels=[0, 1],
        target_names=["genuine", "fraud"],
        zero_division=0,
    ))
    cm_iso = confusion_matrix(y_test, iso_pred, labels=[0, 1])
    print("Confusion matrix:")
    print(f"  {'':15}  predicted_genuine  predicted_fraud")
    print(f"  {'true_genuine':15}  {cm_iso[0, 0]:>17,}  {cm_iso[0, 1]:>14,}")
    print(f"  {'true_fraud':15}  {cm_iso[1, 0]:>17,}  {cm_iso[1, 1]:>14,}")
    iso_precision = precision_score(y_test, iso_pred, zero_division=0)
    iso_recall    = recall_score(y_test, iso_pred, zero_division=0)
    iso_f1        = f1_score(y_test, iso_pred, zero_division=0)
    print(f"\n  Fraud precision : {iso_precision:.4f}")
    print(f"  Fraud recall    : {iso_recall:.4f}")
    print(f"  Fraud F1-score  : {iso_f1:.4f}")

    iso_metrics = {
        "precision": round(iso_precision, 6),
        "recall":    round(iso_recall, 6),
        "f1_score":  round(iso_f1, 6),
        "confusion_matrix": cm_iso.tolist(),
        "test_size": len(y_test),
        "note": "Labels were used for evaluation only; Isolation Forest was trained without them.",
    }

    # ------------------------------------------------------------------
    # 7. Preprocessing reference (computed from TRAIN set only - no leakage)
    # ------------------------------------------------------------------
    medians = X_train.median().to_dict()
    scales  = X_train.std(ddof=0).replace(0, np.nan).fillna(0).to_dict()

    # ------------------------------------------------------------------
    # 8. Save model bundle
    # ------------------------------------------------------------------
    bundle = {
        "random_forest":    rf,
        "isolation_forest": isolation,
        "features":         FEATURES,
        "medians":          medians,
        "scales":           scales,
        "rf_metrics":       rf_metrics,
        "iso_metrics":      iso_metrics,
        "dataset_name":     DATA_PATH.name,
        "train_rows":       len(X_train),
        "test_rows":        len(X_test),
        "total_rows":       total,
        "fraud_count":      fraud_count,
        "genuine_count":    genuine_count,
    }

    output_path = MODEL_DIR / "transaction_models.joblib"
    joblib.dump(bundle, output_path)

    _sep("Training complete")
    print(f"  Model bundle saved to: {output_path}")
    print()
    print("  IMPORTANT LIMITATIONS")
    print("  -------------------------------------------------------------------")
    print("  * These predictions are SCREENING SIGNALS for human review.")
    print("  * A flagged transaction does NOT prove fraud.")
    print("  * V1-V28 are anonymized PCA components; feature values have no")
    print("    direct real-world meaning.")
    print("  * Accuracy alone is misleading on this imbalanced dataset.")
    print("    Always judge by precision, recall, F1 and PR-AUC.")
    print()


if __name__ == "__main__":
    main()
