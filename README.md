# TransactionGuard 🛡️

TransactionGuard is a modern Flask-based web application and dashboard for credit card fraud detection and transaction analytics.

## Features

- **Transaction Overview**: Real-time summary statistics of genuine vs. fraudulent transactions.
- **Fraud Rate Analytics**: Track fraud percentage and average transaction amounts.
- **Dataset Validation**: Automatic validation of column integrity, missing data, and schema compliance.
- **Modern UI**: Clean, responsive dashboard designed for transaction intelligence.

## Getting Started

### Prerequisites

- Python 3.10+
- `pip`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nayansinha787-max/Transactionguard.git
   cd Transactionguard
   ```

2. **Create and activate a virtual environment:**
   - **Windows:**
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open in browser:**
   Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Dataset Information

- The repository includes a representative lightweight sample in `data/creditcard.csv` (10,000 transactions including all 492 fraud cases) so the application works out of the box.
- To use the complete Kaggle Credit Card Fraud Detection dataset (284,807 transactions, ~144 MB), download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder as `creditcard_full.csv`. The app will automatically detect and load the full dataset when present.
