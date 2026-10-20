
from pathlib import Path

import pandas as pd
from flask import Flask, render_template

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FULL_PATH = BASE_DIR / "data" / "creditcard_full.csv"
DATA_PATH = DATA_FULL_PATH if DATA_FULL_PATH.exists() else (BASE_DIR / "data" / "creditcard.csv")

REQUIRED_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Class"]
)


def load_dataset():
    if not DATA_PATH.exists():
        return None, "Dataset not found inside the data folder."

    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None, "Could not read creditcard.csv."

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, "Missing columns: " + ", ".join(missing)

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[REQUIRED_COLUMNS].isna().any().any():
        return None, "Required columns contain missing or invalid values."

    if df.empty:
        return None, "The dataset is empty."

    if not df["Class"].isin([0, 1]).all():
        return None, "Class must contain only 0 and 1."

    stats = {
        "total": len(df),
        "fraud": int((df["Class"] == 1).sum()),
        "genuine": int((df["Class"] == 0).sum()),
        "fraud_rate": float((df["Class"] == 1).mean() * 100),
        "average_amount": float(df["Amount"].mean()),
        "total_columns": len(df.columns),
        "missing_cells": int(df[REQUIRED_COLUMNS].isna().sum().sum()),
    }

    return stats, None


@app.route("/")
def dashboard():
    stats, error = load_dataset()
    return render_template("index.html", stats=stats, error=error)


if __name__ == "__main__":
    app.run(debug=True)
