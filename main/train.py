import os
import itertools
import numpy as np
import pandas as pd
from joblib import dump
from typing import Any

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data_loader import download_stock_data
from .features import build_dataset
from .models import (
    make_logistic_regression,
    make_random_forest,
    make_xgboost,
    make_lightgbm,
    make_mlp,
    HAS_XGBOOST,
    HAS_LIGHTGBM,
)

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed; Transformer model will be skipped.")

# Trading / tuning config
TRADE_THRESHOLD = 0.60        # probability threshold to open a trade
MIN_TRADES_FOR_MODEL = 30     # ignore configs with too few trades (to avoid overfitting)
SEQ_LEN = 30                  # number of past days for the Transformer window

# Horizons we support
HORIZONS = [1, 3, 5]

# Where model artifacts are stored (same path as before)
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def backtest_with_threshold(y_true, proba, returns, threshold=TRADE_THRESHOLD):
    """
    Backtest long-only strategy:
      - Go long when P(up) >= threshold
      - Flat otherwise
    Metrics:
      - accuracy: classification accuracy using this threshold
      - num_trades: number of long trades taken
      - win_rate: fraction of trades with positive return
      - total_return: sum of returns on trade days
      - avg_return: average return per trade
    """
    proba = np.asarray(proba)
    y_true = np.asarray(y_true)
    returns = np.asarray(returns)

    trade_mask = proba >= threshold
    num_trades = int(trade_mask.sum())

    # Classification labels at this threshold
    y_pred = (proba >= threshold).astype(int)
    accuracy = float((y_pred == y_true).mean())

    if num_trades == 0:
        return {
            "accuracy": accuracy,
            "num_trades": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "avg_return": 0.0,
        }

    trade_returns = returns[trade_mask]
    total_return = float(trade_returns.sum())
    wins = int((trade_returns > 0).sum())
    win_rate = wins / num_trades
    avg_return = total_return / num_trades

    return {
        "accuracy": accuracy,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "total_return": total_return,
        "avg_return": avg_return,
    }


def ensemble_predict_proba(
    X_raw,
    *,
    lr_model,
    xgb_model,
    lgb_model,
    mlp_model,
    scaler,
):
    """
    Compute ensemble probability P(up) by averaging member model probabilities.

    - lr_model, mlp_model use scaled features
    - xgb_model, lgb_model use raw features
    Any model can be None and will be skipped.
    """
    proba_members = []

    # Scale once
    X_scaled = scaler.transform(X_raw)

    # Logistic Regression (scaled)
    if lr_model is not None:
        proba_members.append(lr_model.predict_proba(X_scaled)[:, 1])

    # XGBoost (raw)
    if xgb_model is not None:
        proba_members.append(xgb_model.predict_proba(X_raw)[:, 1])

    # LightGBM (raw)
    if lgb_model is not None:
        proba_members.append(lgb_model.predict_proba(X_raw)[:, 1])

    # MLP (scaled)
    if mlp_model is not None:
        proba_members.append(mlp_model.predict_proba(X_scaled)[:, 1])

    if not proba_members:
        raise RuntimeError("No models available for ensemble prediction.")

    # Simple average
    return np.mean(proba_members, axis=0)


def time_series_split(X, y, future_ret=None, train_ratio=0.8):
    """
    Time-ordered split. If future_ret is provided, it is split in the same way.
    """
    split_index = int(len(X) * train_ratio)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if future_ret is None:
        return X_train, X_test, y_train, y_test

    future_ret_train = future_ret.iloc[:split_index]
    future_ret_test = future_ret.iloc[split_index:]
    return X_train, X_test, y_train, y_test, future_ret_train, future_ret_test


def build_sequence_dataset(X, y, future_ret, seq_len=SEQ_LEN):
    """
    Turn tabular time series into overlapping sequences of length seq_len.
    """
    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(np.int64)
    r_arr = future_ret.values.astype(np.float32)

    X_seq = []
    y_seq = []
    r_seq = []

    for i in range(seq_len, len(X_arr)):
        X_seq.append(X_arr[i - seq_len: i])  # window of last seq_len days
        y_seq.append(y_arr[i])               # label for "today"
        r_seq.append(r_arr[i])               # return for "today"

    return np.stack(X_seq), np.array(y_seq), np.array(r_seq)


def evaluate_model(name: str, y_test, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def predict_next_day_for_ticker(
    ticker,
    model,
    start="2023-01-01",
    end=None,
    *,
    threshold=0.002,
    horizon=1,
):
    import datetime

    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")

    print(
        f"\nBuilding prediction for next day of {ticker} "
        f"using data from {start} to {end}..."
    )

    df = download_stock_data(ticker, start, end)

    # Reuse exact same feature pipeline
    X_t, _, feature_cols, _ = build_dataset(
        df,
        threshold=threshold,
        horizon=horizon,
    )

    if len(X_t) == 0:
        print("Not enough data to build features for this ticker.")
        return

    latest_features = X_t.iloc[[-1]]  # last row as 1-sample DataFrame
    proba_up = model.predict_proba(latest_features)[0, 1]
    proba_down = 1 - proba_up

    print(f"Model prediction for {ticker}:")
    print(f"  P(up tomorrow)   = {proba_up:.3f}")
    print(f"  P(down tomorrow) = {proba_down:.3f}")

    # Map probability to trading-style signal
    if proba_up >= 0.60:
        print("  Signal: STRONG BUY (high-confidence up move)")
    elif proba_up >= 0.55:
        print("  Signal: BUY (moderate edge)")
    elif proba_up >= 0.50:
        print("  Signal: WEAK BUY / HOLD (slight edge)")
    else:
        print("  Signal: NO TRADE or SHORT (model not confident up)")


def predict_next_day_ensemble(
    ticker,
    *,
    lr_model,
    xgb_model,
    lgb_model,
    mlp_model,
    scaler,
    start="2023-01-01",
    end=None,
    threshold=0.002,
    horizon=1,
):
    """
    Next-day prediction using the 4-model ensemble:
    Logistic Regression + XGBoost + LightGBM + MLP
    """
    import datetime

    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")

    print(
        f"\nBuilding ENSEMBLE prediction for next day of {ticker} "
        f"using data from {start} to {end}..."
    )

    df = download_stock_data(ticker, start, end)

    X_t, _, feature_cols, _ = build_dataset(
        df,
        threshold=threshold,
        horizon=horizon,
    )

    if len(X_t) == 0:
        print("Not enough data to build features for this ticker.")
        return

    latest_features = X_t.iloc[[-1]]  # last row as 1-sample DataFrame

    proba_up = ensemble_predict_proba(
        latest_features,
        lr_model=lr_model,
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        mlp_model=mlp_model,
        scaler=scaler,
    )[0]

    proba_down = 1.0 - proba_up

    print(f"Ensemble prediction for {ticker}:")
    print(f"  P(up tomorrow)   = {proba_up:.3f}")
    print(f"  P(down tomorrow) = {proba_down:.3f}")

    if proba_up >= 0.60:
        print("  Signal: STRONG BUY (high-confidence up move)")
    elif proba_up >= 0.55:
        print("  Signal: BUY (moderate edge)")
    elif proba_up >= 0.50:
        print("  Signal: WEAK BUY / HOLD (slight edge)")
    else:
        print("  Signal: NO TRADE or SHORT (model not confident up)")


def tune_xgboost_accuracy_old(
    X_train,
    y_train,
    X_test,
    y_test,
    future_ret_test,
):
    """
    Old accuracy-based XGBoost tuner (kept for reference).
    Not used in the main training loop, but left here in case you want it.
    """
    from xgboost import XGBClassifier

    param_grid = {
        "max_depth": [3, 4, 5],
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    base_params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    best_model: Any = None
    best_params = None
    best_acc = -1.0

    print("\n=== XGBoost Hyperparameter Tuning (ACCURACY-BASED, OLD) ===")

    for max_depth in param_grid["max_depth"]:
        for n_estimators in param_grid["n_estimators"]:
            for learning_rate in param_grid["learning_rate"]:
                for subsample in param_grid["subsample"]:
                    for colsample_bytree in param_grid["colsample_bytree"]:
                        params = {
                            "max_depth": max_depth,
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "subsample": subsample,
                            "colsample_bytree": colsample_bytree,
                        }

                        model = XGBClassifier(**base_params, **params)
                        model.fit(X_train, y_train)

                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)

                        print(f"Params {params} → Acc: {acc:.3f}")

                        if acc > best_acc:
                            best_acc = acc
                            best_model = model
                            best_params = params

    print("\n=== Best XGBoost config (by accuracy, OLD) ===")
    print("Best accuracy:", best_acc)
    print("Best params:", best_params)

    return best_model, best_params


# ---------------------------------------------------------------------------
# Training for a single horizon
# ---------------------------------------------------------------------------

def save_artifacts_for_horizon(
    scaler,
    lr_model,
    xgb_clf,
    lgb_clf,
    mlp_clf,
    horizon: int,
):
    """
    Save model artifacts with horizon-specific filenames, e.g.:
    scaler_h1.joblib, lr_model_h3.joblib, etc.
    """
    suffix = f"_h{horizon}"  # e.g. _h1, _h3, _h5

    dump(scaler, os.path.join(ARTIFACT_DIR, f"scaler{suffix}.joblib"))
    dump(lr_model, os.path.join(ARTIFACT_DIR, f"lr_model{suffix}.joblib"))

    if xgb_clf is not None:
        dump(xgb_clf, os.path.join(ARTIFACT_DIR, f"xgb_model{suffix}.joblib"))

    if lgb_clf is not None:
        dump(lgb_clf, os.path.join(ARTIFACT_DIR, f"lgb_model{suffix}.joblib"))

    if mlp_clf is not None:
        dump(mlp_clf, os.path.join(ARTIFACT_DIR, f"mlp_model{suffix}.joblib"))

    print(f"\nArtifacts for horizon={horizon} saved to: {ARTIFACT_DIR}")


def train_for_horizon(
    horizon: int,
    *,
    tickers,
    start: str,
    end: str,
    threshold: float,
):
    """
    Train the full pipeline (LR, RF, XGB, LGBM, MLP, ensemble) for a single horizon,
    evaluate it, and save horizon-specific artifacts.
    """
    print("\n--------------------------------------------------")
    print(f"Training models for horizon = {horizon} day(s)")
    print("--------------------------------------------------")

    xgb_clf = None
    lgb_clf = None
    mlp_clf = None

    X_list = []
    y_list = []
    future_list = []

    for t in tickers:
        print(f"Downloading data for {t}...")
        df_t = download_stock_data(t, start, end)

        X_t, y_t, feature_cols, future_ret_t = build_dataset(
            df_t,
            threshold=threshold,
            horizon=horizon,
        )

        if len(X_t) == 0:
            print(
                f"  Warning: no usable data for {t} after feature engineering. "
                "Skipping."
            )
            continue

        X_list.append(X_t)
        y_list.append(y_t)
        future_list.append(future_ret_t)

    if not X_list:
        print("No usable data for any ticker; skipping this horizon.")
        return

    # Concatenate all tickers into one big dataset
    X = pd.concat(X_list, axis=0)
    y = pd.concat(y_list, axis=0)
    future_ret = pd.concat(future_list, axis=0)

    # Sort each piece independently to keep alignment
    X = X.sort_index(kind="mergesort")
    y = y.sort_index(kind="mergesort")
    future_ret = future_ret.sort_index(kind="mergesort")

    print("Dataset built.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Time-series split
    X_train, X_test, y_train, y_test, future_ret_train, future_ret_test = (
        time_series_split(X, y, future_ret, train_ratio=0.8)
    )

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    # Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = make_logistic_regression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    evaluate_model("Logistic Regression (balanced)", y_test, y_pred_lr)

    # Random Forest
    rf_model = make_random_forest()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    evaluate_model("Random Forest", y_test, y_pred_rf)

    # XGBoost tuned by P&L
    if HAS_XGBOOST:
        print("\n=== XGBoost Hyperparameter Tuning (by P&L) ===")

        xgb_param_grid = {
            "max_depth": [3, 4, 5],
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        param_names = list(xgb_param_grid.keys())
        param_values = list(xgb_param_grid.values())

        best_score = -np.inf
        best_params = None
        best_stats = None
        best_model = None

        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))

            model = make_xgboost(
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
            )

            model.fit(X_train, y_train)
            proba_test = model.predict_proba(X_test)[:, 1]

            stats = backtest_with_threshold(
                y_true=y_test,
                proba=proba_test,
                returns=future_ret_test,
                threshold=TRADE_THRESHOLD,
            )

            acc = stats["accuracy"]
            num_trades = stats["num_trades"]
            win_rate = stats["win_rate"]
            total_ret = stats["total_return"]
            avg_ret = stats["avg_return"]

            print(
                f"Params {params} → "
                f"Acc: {acc:.3f}, Trades: {num_trades}, "
                f"Win: {win_rate:.3f}, "
                f"TotRet: {total_ret:.3f}, AvgRet: {avg_ret:.4f}"
            )

            if num_trades >= MIN_TRADES_FOR_MODEL and total_ret > best_score:
                best_score = total_ret
                best_params = params
                best_stats = stats
                best_model = model

        print(
            "\n=== Best XGBoost config (by total_return at "
            f"threshold {TRADE_THRESHOLD:.2f}) ==="
        )
        print("Best params:", best_params)
        print("Best stats:", best_stats)

        xgb_clf = best_model

        # Final evaluation of tuned XGBoost
        print("\n=== XGBoost (tuned, best by P&L) ===")
        y_pred_xgb = xgb_clf.predict(X_test)
        evaluate_model("XGBoost (tuned by P&L)", y_test, y_pred_xgb)

        # Threshold analysis for multiple thresholds
        thresholds = [0.50, 0.55, 0.60, 0.65]
        proba_test = xgb_clf.predict_proba(X_test)[:, 1]

        print("\n=== Threshold analysis (XGBoost, tuned) ===")
        for th in thresholds:
            stats = backtest_with_threshold(
                y_true=y_test,
                proba=proba_test,
                returns=future_ret_test,
                threshold=th,
            )
            print(
                f"Threshold {th:.2f} → "
                f"Accuracy: {stats['accuracy']:.3f}, "
                f"Trades: {stats['num_trades']}, "
                f"Win rate: {stats['win_rate']:.3f}, "
                f"Total return: {stats['total_return']:.3f}, "
                f"Avg/trade: {stats['avg_return']:.4f}"
            )
    else:
        print("XGBoost not available; skipping XGBoost tuning.")
        xgb_clf = None

    # LightGBM tuned by P&L
    if HAS_LIGHTGBM:
        print("\n=== LightGBM Hyperparameter Tuning (by P&L) ===")

        lgb_param_grid = {
            "n_estimators": [200, 400],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 63],
            "max_depth": [-1, 7],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        lgb_param_names = list(lgb_param_grid.keys())
        lgb_param_values = list(lgb_param_grid.values())

        lgb_best_score = -np.inf
        lgb_best_params = None
        lgb_best_stats = None
        lgb_best_model: Any = None

        for values in itertools.product(*lgb_param_values):
            params = dict(zip(lgb_param_names, values))

            model = make_lightgbm(**params)
            model.fit(X_train, y_train)

            proba_test_lgb = model.predict_proba(X_test)[:, 1]

            stats = backtest_with_threshold(
                y_true=y_test,
                proba=proba_test_lgb,
                returns=future_ret_test,
                threshold=TRADE_THRESHOLD,
            )

            acc = stats["accuracy"]
            num_trades = stats["num_trades"]
            win_rate = stats["win_rate"]
            total_ret = stats["total_return"]
            avg_ret = stats["avg_return"]

            print(
                f"LGBM Params {params} → "
                f"Acc: {acc:.3f}, Trades: {num_trades}, "
                f"Win: {win_rate:.3f}, TotRet: {total_ret:.3f}, "
                f"AvgRet: {avg_ret:.4f}"
            )

            if num_trades >= MIN_TRADES_FOR_MODEL and total_ret > lgb_best_score:
                lgb_best_score = total_ret
                lgb_best_params = params
                lgb_best_stats = stats
                lgb_best_model = model

        print(
            "\n=== Best LightGBM config (by total_return at "
            f"threshold {TRADE_THRESHOLD:.2f}) ==="
        )
        print("Best LGBM params:", lgb_best_params)
        print("Best LGBM stats:", lgb_best_stats)

        lgb_clf = lgb_best_model
        y_pred_lgb = lgb_clf.predict(X_test)
        evaluate_model("LightGBM (tuned by P&L)", y_test, y_pred_lgb)
    else:
        print("\nLightGBM not installed, skipping LightGBM model.")
        lgb_clf = None

    # MLP tuned by P&L
    print("\n=== MLP Hyperparameter Tuning (by P&L) ===")

    mlp_param_grid = {
        "hidden_layer_sizes": [(64, 32), (128, 64)],
        "alpha": [1e-4, 1e-3],
        "learning_rate_init": [1e-3, 5e-4],
    }

    mlp_param_names = list(mlp_param_grid.keys())
    mlp_param_values = list(mlp_param_grid.values())

    mlp_best_score = -np.inf
    mlp_best_params = None
    mlp_best_stats = None
    mlp_best_model: Any = None

    for values in itertools.product(*mlp_param_values):
        params = dict(zip(mlp_param_names, values))

        model = make_mlp(**params)
        model.fit(X_train_scaled, y_train)

        proba_test_mlp = model.predict_proba(X_test_scaled)[:, 1]

        stats = backtest_with_threshold(
            y_true=y_test,
            proba=proba_test_mlp,
            returns=future_ret_test,
            threshold=TRADE_THRESHOLD,
        )

        acc = stats["accuracy"]
        num_trades = stats["num_trades"]
        win_rate = stats["win_rate"]
        total_ret = stats["total_return"]
        avg_ret = stats["avg_return"]

        print(
            f"MLP Params {params} → "
            f"Acc: {acc:.3f}, Trades: {num_trades}, "
            f"Win: {win_rate:.3f}, TotRet: {total_ret:.3f}, "
            f"AvgRet: {avg_ret:.4f}"
        )

        if num_trades >= MIN_TRADES_FOR_MODEL and total_ret > mlp_best_score:
            mlp_best_score = total_ret
            mlp_best_params = params
            mlp_best_stats = stats
            mlp_best_model = model

    print(
        "\n=== Best MLP config (by total_return at "
        f"threshold {TRADE_THRESHOLD:.2f}) ==="
    )
    print("Best MLP params:", mlp_best_params)
    print("Best MLP stats:", mlp_best_stats)

    mlp_clf = mlp_best_model
    y_pred_mlp = mlp_clf.predict(X_test_scaled)
    evaluate_model("MLPClassifier (tuned by P&L)", y_test, y_pred_mlp)

    # 4-model Ensemble: LR + XGB + LGBM + MLP
    print("\n=== Ensemble (LR + XGB + LGBM + MLP) ===")

    proba_ens = ensemble_predict_proba(
        X_test,
        lr_model=lr_model,
        xgb_model=xgb_clf,
        lgb_model=lgb_clf,
        mlp_model=mlp_clf,
        scaler=scaler,
    )

    ens_stats = backtest_with_threshold(
        y_true=y_test,
        proba=proba_ens,
        returns=future_ret_test,
        threshold=TRADE_THRESHOLD,
    )

    print(
        f"Backtest @threshold {TRADE_THRESHOLD:.2f} → "
        f"Accuracy: {ens_stats['accuracy']:.3f}, "
        f"Trades: {ens_stats['num_trades']}, "
        f"Win rate: {ens_stats['win_rate']:.3f}, "
        f"Total return: {ens_stats['total_return']:.3f}, "
        f"Avg/trade: {ens_stats['avg_return']:.4f}"
    )

    # Also view as a pure classifier with 0.5 cutoff
    y_pred_ens = (proba_ens >= 0.5).astype(int)
    evaluate_model("Ensemble (prob-average, 0.5 cutoff)", y_test, y_pred_ens)

    # Simple per-ticker backtest using final ensemble
    print("\n=== Per-ticker ensemble backtest (test split per ticker) ===")
    for t in tickers:
        df_t = download_stock_data(t, start, end)

        X_t, y_t, _, future_ret_t = build_dataset(
            df_t,
            threshold=threshold,
            horizon=horizon,
        )

        if len(X_t) == 0:
            print(f"{t}: no usable data after feature engineering.")
            continue

        split_idx_t = int(len(X_t) * 0.8)
        X_t_train, X_t_test = X_t.iloc[:split_idx_t], X_t.iloc[split_idx_t:]
        y_t_train, y_t_test = y_t.iloc[:split_idx_t], y_t.iloc[split_idx_t:]
        future_ret_t_test = future_ret_t.iloc[split_idx_t:]

        if len(X_t_test) == 0:
            print(f"{t}: no test data window.")
            continue

        proba_t = ensemble_predict_proba(
            X_t_test,
            lr_model=lr_model,
            xgb_model=xgb_clf,
            lgb_model=lgb_clf,
            mlp_model=mlp_clf,
            scaler=scaler,
        )

        stats_t = backtest_with_threshold(
            y_true=y_t_test,
            proba=proba_t,
            returns=future_ret_t_test,
            threshold=TRADE_THRESHOLD,
        )

        print(
            f"{t}: Acc={stats_t['accuracy']:.3f}, "
            f"Trades={stats_t['num_trades']}, "
            f"Win={stats_t['win_rate']:.3f}, "
            f"TotRet={stats_t['total_return']:.3f}, "
            f"AvgRet={stats_t['avg_return']:.4f}"
        )

    # Optional: per-ticker next-day ensemble predictions (for logs)
    print("\n=== Next-day predictions (4-model ENSEMBLE) ===")
    for t in tickers:
        predict_next_day_ensemble(
            ticker=t,
            lr_model=lr_model,
            xgb_model=xgb_clf,
            lgb_model=lgb_clf,
            mlp_model=mlp_clf,
            scaler=scaler,
            start=start,
            end=None,
            threshold=threshold,
            horizon=horizon,
        )

    # Save artifacts for THIS horizon
    save_artifacts_for_horizon(
        scaler=scaler,
        lr_model=lr_model,
        xgb_clf=xgb_clf,
        lgb_clf=lgb_clf,
        mlp_clf=mlp_clf,
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Multi-ticker training config
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start = "2020-01-01"
    end = "2025-11-20"
    threshold = 0.002  # +0.2% move

    for horizon in HORIZONS:
        train_for_horizon(
            horizon=horizon,
            tickers=tickers,
            start=start,
            end=end,
            threshold=threshold,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
