# Crypto Predictor — Calibrated Directional Probabilities

A machine learning and quantitative analysis engine that predicts next-candle directional probabilities ($P(\text{UP})$ and $P(\text{DOWN})$) for cryptocurrencies, paired with an interactive Streamlit dashboard.

---

## Description

Crypto Predictor replaces naive rule-based scoring and inflated 3-class classification with an honest machine learning pipeline. Instead of chasing impossible 70–80% 3-class accuracy (which in reality suffers from majority-class collapse where predicting "SIDEWAYS" yields ~66% accuracy with zero predictive value), this system:

- Trains two calibrated binary LightGBM estimators for $P(\text{UP})$ and $P(\text{DOWN})$ with a volatility-adjusted deadzone.
- Calibrates probabilities using out-of-fold isotonic regression to ensure predicted probabilities match true frequencies.
- Evaluates skill via Brier Score Loss, Brier Skill Score (BSS) vs climatology, and ROC-AUC.
- Implements a confidence gate based on historical out-of-fold confidence percentiles (top 20%), returning an honest **"NO CLEAR EDGE"** when confidence is insufficient.
- Employs a fee-adjusted fractional-Kelly backtest to provide an objective deployability verdict:
  - `POSITIVE NET EDGE`: Gated strategy generates positive Sharpe ratio net of fees on held-out test data.
  - `SIGNAL BUT NOT TRADEABLE`: Model demonstrates statistical edge over climatology, but cannot overcome transaction costs.
  - `NOT DEPLOYABLE`: No statistical edge over climatology.

---

## Features

- **Causal Feature Engineering**: Over 60 indicators calculated without lookahead bias, including RSI, MACD, Stochastic, CCI, Williams %R, Bollinger Bands, Ichimoku Cloud, VWAP distance, OBV slope, candle body/wick ratios, swing support/resistance, and candlestick pattern recognition.
- **Derivatives Alt-Data Integration**: Incorporates Binance perpetual funding rates (`fund_rate`, `fund_sum_24h`, `fund_z30`, `fund_trend`) causally aligned to candle close times.
- **Live Market Microstructure**: Live open interest (OI) change and order-book depth imbalance diagnostics displayed at inference.
- **Multi-Timeframe Context**: 1h models incorporate aggregated 4h and 1d trend/momentum indicators; 4h models incorporate 1d context.
- **Leakage Prevention**: Built-in verification test (`python features.py --selftest`) that corrupts future candles to guarantee strict causal integrity.
- **Walk-Forward Validation**: Expanding-window out-of-sample evaluation (never random k-fold) across deep multi-year history.
- **Streamlit Web Application**: Interactive dashboard featuring real-time directional probabilities, technical indicator breakdowns, model diagnostics, and confidence gates.

---

## Tech Stack

- **Core Logic & Modeling**: Python 3, NumPy, Pandas, Scikit-learn, LightGBM, Joblib, PyArrow
- **Data Ingestion**: Binance Public REST APIs, CoinGecko Public API
- **User Interface**: Streamlit
- **Testing & Validation**: Custom leak-free self-test framework

---

## Project Structure

```text
crypto-screener/
├── app.py                 # Streamlit web dashboard
├── build_dataset.py       # Multi-year OHLCV downloader with pagination & fallback
├── altdata.py             # Derivatives alt-data ingestion (funding rates, OI, order book)
├── features.py            # Causal feature extraction, labeling & leakage self-tests
├── train_model.py         # Walk-forward validation, calibration, backtesting & training
├── retrain.py             # Pipeline automation for scheduled updates
├── predict.py             # Classical rule-based indicator engine (baseline comparison)
├── mlutils.py             # Custom ML utilities (isotonic calibration, robust scalers)
├── requirements.txt       # Python package dependencies
├── .env.example           # Example configuration for environment variables
├── .gitignore             # Git exclusions for secrets, caches, and datasets
├── models/
│   ├── manifest.json      # Registry tracking latest model per timeframe
│   └── model_*.joblib     # Serialized LightGBM models and calibrators
└── data/                  # Local data storage (excluded from git)
    ├── raw/               # Downloaded OHLCV parquet files
    ├── alt/               # Cached funding rate parquet files
    └── features/          # Engineered feature matrix parquets
```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/crypto-screener.git
   cd crypto-screener
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data and train models** (optional if using pre-trained models):
   ```bash
   # Download historical OHLCV data
   python build_dataset.py

   # Fetch funding rate history
   python altdata.py

   # Build feature caches
   python features.py --cache

   # Train models across timeframes
   python train_model.py --all-tfs
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

---

## Environment Variables

The project functions out of the box using public, unauthenticated Binance endpoints. To increase rate limits and prevent HTTP 429 errors during heavy historical data fetching, you may optionally configure Binance API credentials.

Copy the `.env.example` template:
```bash
cp .env.example .env
```

Define the variables in your `.env` file or environment:

```env
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
```

| Variable | Required | Description |
|---|---|---|
| `BINANCE_API_KEY` | Optional | Binance API key to increase request rate limits. |
| `BINANCE_API_SECRET` | Optional | Binance API secret key. |

---

## Validation & Verification

To verify that the feature engineering pipeline has zero lookahead bias:

```bash
python features.py --selftest
```

---

## Disclaimer

This system is built for quantitative research and educational purposes. Cryptocurrency markets are highly volatile and regime-dependent. The models do not predict future returns with certainty, and directional confidence does not constitute financial advice.
