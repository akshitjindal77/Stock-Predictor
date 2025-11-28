# Stock Predictor

Full-stack ML app that predicts whether a stock will close up or down on the next trading day. It downloads historical OHLCV data, engineers technical features, runs an ensemble of four models, and serves results via FastAPI with a modern React dashboard.

## Overview
- Predicts next-day move (up/down) with probabilities and trading-style signals (Strong Buy/Buy/Weak Buy/No Trade).
- Confidence meter based on distance from 50% probability.
- 7-day rolling prediction history with bullish/bearish sentiment and accuracy.
- Feature list explorer showing all engineered inputs.
- Responsive dark UI.

## Architecture
- **Frontend:** React (Vite) + Axios calling the prediction API.
- **Backend:** FastAPI for inference.
- **Models:** Logistic Regression, Random Forest, XGBoost, MLP (ensemble average).
- **Data:** yfinance for historical OHLCV; technical/engineered features built in Python.

## Project Structure
```
backend/
  app.py           # FastAPI entrypoint
  inference.py     # Ensemble inference logic
  data_loader.py   # yfinance downloader
  features.py      # Feature engineering + labeling
  models.py        # Model factories
  train.py         # Training script
  schemas.py       # Pydantic schemas

frontend/
  src/
    App.jsx
    App.css
    api/
      client.js
      predictions.js
    main.jsx
    index.css
```

## Getting Started

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
uvicorn app:app --reload
```
API runs at `http://127.0.0.1:8000`.

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Vite dev server runs at `http://127.0.0.1:5173` (or shown in the console).

## Training (optional)
- Configure tickers/date range in `main/train.py`.
- Run `python -m main.train` to fit models and save artifacts into `main/artifacts/` (`scaler.joblib`, model joblibs).

## Training Methodology
1) **Data collection:** Download daily OHLCV for multiple large-cap tickers via yfinance (e.g., AAPL, MSFT, GOOGL, AMZN, TSLA) for a multi-year window (default 2020-01-01 to 2025-11-01).  
2) **Feature engineering:** Compute indicators/engineered features in `features.py` (EMA/MACD/RSI, Bollinger position, lags, rolling stats, ranges, volume stats, candle shapes). Drop rows with NaNs from rolling windows.  
3) **Labeling (thresholded):**  
   - Future_Return = Close[t + horizon] / Close[t] - 1  
   - Threshold = +0.2% (0.002) with horizon = 1 by default  
   - Target = 1 if Future_Return > threshold else 0  
4) **Split:** Chronological 80/20 time-series split (no shuffling) to avoid look-ahead bias.  
5) **Scaling:** Standardize features for linear/NN models (Logistic Regression, MLP); tree models consume raw values.  
6) **Model training:** Fit Logistic Regression (balanced), Random Forest, XGBoost, and MLP on the training slice.  
7) **Ensembling:** Average `P(up)` across the four models. `P(down) = 1 - P(up)`.  
8) **Evaluation:** Run on the held-out 20% time slice; report accuracy and qualitative signal mapping (Strong Buy/Buy/Weak Buy/No Trade).  
9) **Artifacts:** Save scaler + fitted models to `main/artifacts/` for reuse at inference time.  

## API Usage
- Endpoint: `POST /api/predict`
- Example payload:
```json
{
  "ticker": "AAPL",
  "start": "2020-01-01",
  "end": null,
  "threshold": 0.002,
  "horizon": 1
}
```
- Example response (abridged):
```json
{
  "ticker": "AAPL",
  "as_of": "2025-11-24",
  "prediction_for": "2025-11-25",
  "p_up": 0.63,
  "p_down": 0.37,
  "signal": "BUY",
  "history": [...],
  "metrics": {
    "recent_history_accuracy": 0.57,
    "bullish_days": 4,
    "bearish_days": 3
  },
  "meta": {
    "num_features": 37,
    "feature_names": [...]
  }
}
```

## Feature Engineering
- Trend: EMA12, EMA26, MACD line/signal/histogram, Bollinger position.
- Momentum: RSI14, lagged returns (1/2/3/5), rolling returns (3/7).
- Volatility: rolling std/min/max, High-Low and Close-Open ranges.
- Volume: MA5, std5.
- Candle shape: upper/lower wick, body size.
- Label: next-day return > +0.2% -> 1 else 0 (horizon=1 by default).

## Model Performance (representative)
| Model                | Accuracy (test) | Notes                           |
| -------------------- | -------------- | --------------------------------|
| Logistic Regression  | ~55%           | Strong baseline, low variance   |
| Random Forest        | ~56%           | Captures non-linear patterns    |
| XGBoost              | ~57%           | Best single model               |
| MLP                  | ~54%           | Adds diversity to ensemble      |

Ensemble averages probabilities for stability: `P(up) = mean(model_probs)`, `P(down) = 1 - P(up)`.

## Model Evaluation & Per-Ticker Accuracy
- **Evaluation approach:** 80/20 chronological split; no shuffling; metrics computed on the final 20% slice. Logistic/MLP use standardized features; tree models use raw features; ensemble is a simple average of `P(up)`.  
- **Metrics reported:** Accuracy and qualitative signal mapping; recent-history accuracy is also shown in the UI for the last 7 labeled days.  
- **Representative per-ticker accuracy (example run):**

| Ticker | Accuracy | Notes                     |
| ------ | -------- | ------------------------- |
| AAPL   | ~56%     | Steady patterns, lower noise |
| MSFT   | ~55%     | Similar to AAPL baseline     |
| GOOGL  | ~57%     | Benefited most from ensemble |
| AMZN   | ~54%     | More volatile, wider swings  |
| TSLA   | ~53%     | High volatility, noisier labels |

Re-run training on your data range to refresh these numbers.

## Future Improvements
- Business-day awareness and calendar adjustments.
- Multiple horizons (1/3/5 days) and per-ticker tuning.
- Richer backtests (Sharpe, max drawdown) and trade simulation.
- Deployment with CI/CD and caching/rate limiting.
- Add LSTM/Transformer variants for sequence modeling.

## Disclaimer
This project is for educational purposes only. It is not financial advice and should not be used for real trading decisions. Use at your own risk.
