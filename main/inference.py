import os
import datetime
from typing import Dict, Any, Optional

import numpy as np
from joblib import load

from .data_loader import download_stock_data
from .features import build_dataset
from .train import ensemble_predict_proba, TRADE_THRESHOLD


# Where trained artifacts are stored (you will save them from train.py)
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

SCALER_PATH   = os.path.join(ARTIFACT_DIR, "scaler.joblib")
LR_PATH       = os.path.join(ARTIFACT_DIR, "lr_model.joblib")
XGB_PATH      = os.path.join(ARTIFACT_DIR, "xgb_model.joblib")
LGB_PATH      = os.path.join(ARTIFACT_DIR, "lgb_model.joblib")
MLP_PATH      = os.path.join(ARTIFACT_DIR, "mlp_model.joblib")

# Optional: if you want to reuse same threshold / horizon everywhere
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
    - Returns a JSON-friendly dictionary for your API/React UI.
    """

    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")

    # 1) Load models
    artifacts = load_artifacts()

    # 2) Get price data
    df = download_stock_data(ticker, start, end)

    # 3) Build features/labels exactly as in training
    X_t, _, feature_cols, _ = build_dataset(
        df,
        threshold=threshold,
        horizon=horizon,
    )

    if len(X_t) == 0:
        raise ValueError(
            f"Not enough data to build features for ticker {ticker}. "
            f"Try an earlier start date."
        )

    # 4) Take the most recent row as input
    latest_features = X_t.iloc[[-1]]  # shape: (1, n_features)
    latest_date = latest_features.index[-1]

    # 5) Ensemble probability
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

    # 6) Return JSON-friendly payload
    return {
        "ticker": ticker,
        "as_of": str(latest_date),  # e.g. '2025-11-25'
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
    }


if __name__ == "__main__":
    # Quick manual test:
    result = predict_next_day_ensemble_api("AAPL")
    from pprint import pprint
    pprint(result)
