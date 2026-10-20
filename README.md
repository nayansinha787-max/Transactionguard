# TransactionGuard 🛡️

An explainable credit-card fraud detection web application built with Flask, scikit-learn, and vanilla JavaScript.

> **Student project** — predictions are screening signals for human review, not proof of fraud.

---

## Project structure

```
TransactionGuard/
├── app.py               Flask application and API routes
├── train_models.py      Train Random Forest and Isolation Forest
├── test_api.py          API and model bundle tests
├── requirements.txt     Python dependencies
├── .gitignore
├── README.md
├── data/
│   └── creditcard.csv   10 000-row sample (all 492 fraud cases included)
│   └── creditcard_full.csv   Full 284 807-row dataset (local only, git-ignored)
├── models/              Created automatically during training
│   └── transaction_models.joblib
├── static/
│   ├── css/style.css
│   └── js/app.js
└── templates/
    └── index.html
```

---

## Quick start (Windows PowerShell)

### 1 — Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If you don't have a virtual environment yet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

### 3 — (Optional) Add the full dataset

Download **creditcard.csv** from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and save it as:

```
data\creditcard_full.csv
```

The training script automatically prefers `creditcard_full.csv` when present. The bundled 10 000-row sample works too, but the full dataset gives more representative evaluation metrics.

### 4 — Train the models

```powershell
python train_models.py
```

This will:
- Validate the dataset schema and values
- Perform a stratified 80/20 train/test split
- Train a **Random Forest** classifier (`class_weight="balanced"`)
- Train an **Isolation Forest** anomaly detector (no labels used during fit)
- Print precision, recall, F1, PR-AUC and confusion matrices
- Save everything to `models\transaction_models.joblib`

Training on the full dataset takes roughly 60–90 seconds.

### 5 — Start Flask

```powershell
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

### 6 — Run the tests

```powershell
python test_api.py
```

Run this **after** training. Some tests are automatically skipped until models exist.

---

## API endpoints

### `GET /api/health`

Returns application and model status.

```json
{
  "status": "ok",
  "dataset_loaded": true,
  "dataset_error": null,
  "models_trained": true
}
```

### `GET /api/stats`

Returns dataset statistics and model evaluation metrics.

### `POST /api/predict`

Screen a single transaction. Send a JSON object with all 30 features:

```json
{
  "Time": 10000,
  "V1": -1.36,
  "V2": -0.07,
  "V3": 2.54,
  "V4": 1.38,
  "V5": -0.34,
  "V6": 0.46,
  "V7": 0.24,
  "V8": 0.10,
  "V9": 0.36,
  "V10": 0.09,
  "V11": -0.55,
  "V12": -0.62,
  "V13": -0.99,
  "V14": -0.31,
  "V15": 1.47,
  "V16": -0.47,
  "V17": 0.21,
  "V18": 0.03,
  "V19": 0.40,
  "V20": 0.25,
  "V21": -0.02,
  "V22": 0.28,
  "V23": -0.11,
  "V24": 0.07,
  "V25": 0.13,
  "V26": -0.19,
  "V27": 0.13,
  "V28": -0.02,
  "Amount": 149.62
}
```

**Do not send the `Class` field** — it is the prediction target, not an input.

Example using curl:

```powershell
$body = '{"Time":10000,"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02,"Amount":149.62}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/predict -ContentType "application/json" -Body $body
```

---

## Important limitations

| Limitation | Detail |
|---|---|
| **Imbalanced dataset** | Only ~0.17 % of transactions are fraud. Accuracy alone is meaningless — use precision, recall, F1 and PR-AUC. |
| **Anonymized features** | V1–V28 are PCA-transformed. Their values have no direct real-world meaning. |
| **Predictions are signals** | A flagged transaction must be reviewed by a human. It does not prove fraud. |
| **Sample vs full dataset** | The 10 000-row sample includes all 492 fraud cases intentionally, making it unrepresentative of real distributions. Use the full dataset for reliable evaluation. |
| **Isolation Forest is unsupervised** | Labels are used for evaluation only, never during training. Its precision/recall figures reflect how well unsupervised anomaly detection overlaps with labeled fraud — not a supervised score. |
| **Not production-ready** | This is a student demo. Do not use it in a real payment system. |
