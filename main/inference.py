import os
import datetime
from typing import Dict, Any, Optional

import numpy as np
from joblib import load

from .data_loader import download_stock_data
from .features import build_dataset
from .train import ensemble_predict_proba, TRADE_THRESHOLD


ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

SCALER_PATH   = os.path.join(ARTIFACT_DIR, "scaler.joblib")
LR_PATH       = os.path.join(ARTIFACT_DIR, "lr_model.joblib")
XGB_PATH      = os.path.join(ARTIFACT_DIR, "xgb_model.joblib")
LGB_PATH      = os.path.join(ARTIFACT_DIR, "lgb_model.joblib")
MLP_PATH      = os.path.join(ARTIFACT_DIR, "mlp_model.joblib")

DEFAULT_THRESHOLD = 0.002
DEFAULT_HORIZON = 1
DEFAULT_START = "2020-01-01"


class ModelArtifacts:
    """
    Simple container for all loaded models + scaler.
    """

    def __init__(self, scaler, lr_model, xgb_model, lgb_model, mlp_model):
        self.scaler = scaler
        self.lr_model = lr_model
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.mlp_model = mlp_model


def load_artifacts() -> ModelArtifacts:
    """
    Load scaler + models from disk.
    Make sure train.py has saved them to main/artifacts/ first.
    """
    if not os.path.isdir(ARTIFACT_DIR):
        raise FileNotFoundError(
            f"Artifact directory not found: {ARTIFACT_DIR}. "
            "Did you run training and save the models?"
        )

    scaler = load(SCALER_PATH)
    lr_model = load(LR_PATH)
    xgb_model = load(XGB_PATH)
    lgb_model = load(LGB_PATH)
    mlp_model = load(MLP_PATH)

    return ModelArtifacts(
        scaler=scaler,
        lr_model=lr_model,
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        mlp_model=mlp_model,
    )


def _signal_from_proba(p_up: float) -> str:
    """
    Map probability of up-move to a human-readable trading signal.
    You can tweak thresholds later if you want.
    """
    if p_up >= 0.60:
        return "STRONG BUY"
    elif p_up >= 0.55:
        return "BUY"
    elif p_up >= 0.50:
        return "WEAK BUY / HOLD"
    else:
        return "NO TRADE / SHORT BIAS"


def predict_next_day_ensemble_api(
    ticker: str,
    *,
    start: str = DEFAULT_START,
    end: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
    horizon: int = DEFAULT_HORIZON,
) -> Dict[str, Any]:
    """
    High-level inference function for your backend.

    - Loads trained models + scaler
    - Downloads fresh data for `ticker`
    - Rebuilds features (same pipeline as training)
    - Uses 4-model ensemble to get P(up)
    - Also computes a 7-day rolling history of predictions vs. actual labels
    - Returns a JSON-friendly dictionary for your API/React UI.
    """

    if end is None:
        # yfinance end date is exclusive; bump by 1 day to include today's bar
        end = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # 1) Load models
    artifacts = load_artifacts()

    # 2) Get price data
    df = download_stock_data(ticker, start, end)

    # 3) Build features/labels exactly as in training
    #    (we keep y and future_ret so that we can compute history/accuracy)
    X_t, y_t, feature_cols, future_ret_t = build_dataset(
        df,
        threshold=threshold,
        horizon=horizon,
        drop_future_nan=False,  # keep most recent row even though target is NaN
    )

    if len(X_t) == 0:
        raise ValueError(
            f"Not enough data to build features for ticker {ticker}. "
            f"Try an earlier start date."
        )

    # 4) Take the most recent row as input (today’s features)
    latest_features = X_t.iloc[[-1]]  # shape: (1, n_features)
    latest_date = latest_features.index[-1]

    # Prediction day = "next" day after the last feature row
    try:
        base_date = latest_date.date()  # pandas.Timestamp
    except AttributeError:
        # Fallback if it's already a date/str
        if isinstance(latest_date, datetime.date):
            base_date = latest_date
        else:
            base_date = datetime.datetime.strptime(str(latest_date), "%Y-%m-%d").date()

    prediction_for = (base_date + datetime.timedelta(days=1)).isoformat()

    # 5) Ensemble probability for the next day (main prediction)
    proba_up_arr = ensemble_predict_proba(
        latest_features,
        lr_model=artifacts.lr_model,
        xgb_model=artifacts.xgb_model,
        lgb_model=artifacts.lgb_model,
        mlp_model=artifacts.mlp_model,
        scaler=artifacts.scaler,
    )
    proba_up = float(proba_up_arr[0])
    proba_down = float(1.0 - proba_up)

    signal = _signal_from_proba(proba_up)

    # 6) Build a 7-day rolling history of predictions vs actual labels
    HISTORY_WINDOW = 7
    history_records: list[Dict[str, Any]] = []

    # use the last N rows where we actually have labels
    labeled_mask = y_t.notna()
    labeled_X = X_t.loc[labeled_mask]
    labeled_y = y_t.loc[labeled_mask]
    recent_X = labeled_X.tail(HISTORY_WINDOW)
    correct = 0

    for idx, row in recent_X.iterrows():
        row_df = row.to_frame().T  # keep as DataFrame
        p_up_hist = float(
            ensemble_predict_proba(
                row_df,
                lr_model=artifacts.lr_model,
                xgb_model=artifacts.xgb_model,
                lgb_model=artifacts.lgb_model,
                mlp_model=artifacts.mlp_model,
                scaler=artifacts.scaler,
            )[0]
        )
        p_down_hist = float(1.0 - p_up_hist)

        actual_label = int(labeled_y.loc[idx])      # 1 = up, 0 = down
        pred_label = int(p_up_hist >= 0.5)    # simple 0.5 cutoff

        if pred_label == actual_label:
            correct += 1

        history_records.append(
            {
                "date": str(idx),
                "p_up": p_up_hist,
                "p_down": p_down_hist,
                "actual_label": actual_label,
                "pred_label": pred_label,
            }
        )

    history_accuracy = (
        float(correct) / len(history_records) if history_records else None
    )

    bullish_days = sum(1 for rec in history_records if rec["p_up"] >= 0.5)
    bearish_days = len(history_records) - bullish_days

    # 7) Return JSON-friendly payload
    return {
        "ticker": ticker.upper(),
        "as_of": str(latest_date),         # last date we have features for
        "prediction_for": prediction_for,  # date the model is predicting
        "start": start,
        "end": end,
        "p_up": proba_up,
        "p_down": proba_down,
        "signal": signal,
        "config": {
            "threshold_for_labeling": {
                "strong_buy": 0.60,
                "buy": 0.55,
                "weak_buy": 0.50,
            },
            "trade_threshold_for_backtest": TRADE_THRESHOLD,
            "feature_threshold": threshold,
            "horizon_days": horizon,
        },
        "meta": {
            "num_features": len(feature_cols),
            "feature_names": list(feature_cols),
        },
        "history": history_records,
        "metrics": {
            "recent_history_accuracy": history_accuracy,
            "history_window": len(history_records),
            "bullish_days": bullish_days,
            "bearish_days": bearish_days,
        },
    }



if __name__ == "__main__":
    result = predict_next_day_ensemble_api("AAPL")
    from pprint import pprint
    pprint(result)
