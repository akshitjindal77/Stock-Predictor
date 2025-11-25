import pandas as pd
import os
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

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
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed; Transformer model will be skipped.")
# Trading / tuning config
TRADE_THRESHOLD = 0.60        # probability threshold to open a trade
MIN_TRADES_FOR_MODEL = 30     # ignore configs with too few trades (to avoid overfitting)
SEQ_LEN = 30   # number of past days for the Transformer window

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

    # Long when proba >= threshold
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

def ensemble_predict_proba(X_raw,*, lr_model, xgb_model, lgb_model, mlp_model, scaler):
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


def time_series_split(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    return X_train, X_test, y_train, y_test

def build_sequence_dataset(X, y, future_ret, seq_len=SEQ_LEN):
    #Turn tabular time series into overlapping sequences of length seq_len.
    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(np.int64)
    r_arr = future_ret.values.astype(np.float32)

    X_seq = []
    y_seq = []
    r_seq = []

    for i in range(seq_len, len(X_arr)):
        X_seq.append(X_arr[i - seq_len : i])  # window of last seq_len days
        y_seq.append(y_arr[i])                # label for "today"
        r_seq.append(r_arr[i])                # return for "today"

    return np.stack(X_seq), np.array(y_seq), np.array(r_seq)


def evaluate_model(name, y_test, y_pred):
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

    print(f"\nBuilding prediction for next day of {ticker} using data from {start} to {end}...")

    df = download_stock_data(ticker, start, end)

    # Reuse exact same feature pipeline
    X_t, _, feature_cols, _ = build_dataset(df, threshold=threshold, horizon=horizon)

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

    print(f"\nBuilding ENSEMBLE prediction for next day of {ticker} using data from {start} to {end}...")

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


def tune_xgboost(X_train, y_train, X_test, y_test, future_ret_test):
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

    best_model = None
    best_params = None
    best_acc = -1.0

    print("\n=== XGBoost Hyperparameter Tuning (ACCURACY-BASED, OLD) ===")

    results = []

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

                        # Use a fixed threshold 0.60 for P/L evaluation
                        y_proba = model.predict_proba(X_test)[:, 1]
                        trade_mask = (y_proba >= 0.60)
                        trades = trade_mask.sum()

                        if trades > 0:
                            wins = ((trade_mask) & (y_test == 1)).sum()
                            win_rate = wins / trades
                            trade_returns = future_ret_test[trade_mask]
                            total_return = trade_returns.sum()
                            avg_return = trade_returns.mean()
                        else:
                            win_rate = 0.0
                            total_return = 0.0
                            avg_return = 0.0

                        results.append(
                            (acc, win_rate, total_return, avg_return, params)
                        )

                        print(
                            f"Params {params} → "
                            f"Acc: {acc:.3f}, Trades: {trades}, "
                            f"Win: {win_rate:.3f}, "
                            f"TotRet: {total_return:.3f}, AvgRet: {avg_return:.4f}"
                        )

                        if acc > best_acc:
                            best_acc = acc
                            best_model = model
                            best_params = params

    print("\n=== Best XGBoost config (by accuracy, OLD) ===")
    print("Best accuracy:", best_acc)
    print("Best params:", best_params)

    return best_model, best_params

class SimpleTransformerClassifier(nn.Module):
    """
    Very simple Transformer-based classifier for tabular data.

    We:
      - Project the feature vector to a d_model embedding
      - Treat it as a sequence of length 1
      - Run a TransformerEncoder
      - Use a linear head to predict a logit for "up" (1)
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (batch, seq, feature)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (batch, input_dim)
        """
        x = self.input_proj(x)     # (batch, d_model)
        x = x.unsqueeze(1)         # (batch, 1, d_model)  -> seq_len = 1
        x = self.encoder(x)        # (batch, 1, d_model)
        x = x[:, 0, :]             # (batch, d_model)
        logits = self.cls_head(x)  # (batch, 1)
        return logits.squeeze(-1)  # (batch,)


def tune_transformer(
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    future_ret_test,
    trade_threshold=TRADE_THRESHOLD,
    min_trades=MIN_TRADES_FOR_MODEL,
):
    """
    Tune a small Transformer model by total_return at the fixed trading threshold.
    """

    if not HAS_TORCH:
        print("\nPyTorch not available; skipping Transformer tuning.")
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert data to tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    X_test_t  = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_np = y_test.values  # keep as numpy for backtesting

    future_ret_np = future_ret_test.values

    input_dim = X_train_scaled.shape[1]

    # Small hyperparameter grid
    param_grid = [
        {"d_model": 64, "num_layers": 2, "lr": 1e-3},
        {"d_model": 128, "num_layers": 2, "lr": 1e-3},
        {"d_model": 64, "num_layers": 3, "lr": 5e-4},
    ]

    best_score = -np.inf
    best_params = None
    best_stats = None
    best_state_dict = None

    print("\n=== Transformer Hyperparameter Tuning (by P&L) ===")

    for params in param_grid:
        d_model = params["d_model"]
        num_layers = params["num_layers"]
        lr = params["lr"]

        model = SimpleTransformerClassifier(
            input_dim=input_dim,
            d_model=d_model,
            nhead=4,
            num_layers=num_layers,
            dim_feedforward=4 * d_model,
            dropout=0.1,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Simple training loop
        n_epochs = 10
        batch_size = 256

        model.train()
        n_train = X_train_t.size(0)

        for epoch in range(n_epochs):
            # shuffle indices
            perm = torch.randperm(n_train)
            X_train_t = X_train_t[perm]
            y_train_t = y_train_t[perm]

            for start in range(0, n_train, batch_size):
                end = start + batch_size
                xb = X_train_t[start:end]
                yb = y_train_t[start:end]

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # Evaluate on test set (probabilities)
        model.eval()
        with torch.no_grad():
            logits_test = model(X_test_t).cpu().numpy()
            proba_test = 1 / (1 + np.exp(-logits_test))  # sigmoid

        # Backtest at fixed threshold
        stats = backtest_with_threshold(
            y_true=y_test_np,
            proba=proba_test,
            returns=future_ret_np,
            threshold=trade_threshold,
        )

        acc = stats["accuracy"]
        num_trades = stats["num_trades"]
        win_rate = stats["win_rate"]
        total_ret = stats["total_return"]
        avg_ret = stats["avg_return"]

        print(
            f"Transformer Params {params} → "
            f"Acc: {acc:.3f}, Trades: {num_trades}, "
            f"Win: {win_rate:.3f}, TotRet: {total_ret:.3f}, AvgRet: {avg_ret:.4f}"
        )

        # Selection criterion: maximize total_return with enough trades
        if num_trades >= min_trades and total_ret > best_score:
            best_score = total_ret
            best_params = params
            best_stats = stats
            best_state_dict = model.state_dict()

    if best_params is None:
        print("\nNo Transformer config met the min_trades requirement.")
        return None, None, None

    print(
        "\n=== Best Transformer config (by total_return at "
        f"threshold {trade_threshold:.2f}) ==="
    )
    print("Best Transformer params:", best_params)
    print("Best Transformer stats:", best_stats)

    # Rebuild best model and load weights
    best_model = SimpleTransformerClassifier(
        input_dim=input_dim,
        d_model=best_params["d_model"],
        nhead=4,
        num_layers=best_params["num_layers"],
        dim_feedforward=4 * best_params["d_model"],
        dropout=0.1,
    ).to(device)

    best_model.load_state_dict(best_state_dict)
    best_model.eval()

    # Return model + stats + device (so we can reuse it below)
    return best_model, best_stats, device

def main():

    xgb_clf = None
    lgb_clf = None
    mlp_clf = None
    # === MULTI-TICKER TRAINING ===
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start = "2020-01-01"
    end = "2025-11-01"
    # Target configuration matches build_dataset in features.py
    threshold = 0.002  # +0.2% move
    horizon = 1        # predict 1 day ahead

    X_list = []
    y_list = []
    future_list = []

    for t in tickers:
        print(f"Downloading data for {t}...")
        df_t = download_stock_data(t, start, end)

        X_t, y_t, feature_cols, future_ret_t = build_dataset(
            df_t, threshold=threshold, horizon=horizon
        )

        # In case some ticker has no valid rows after indicators:
        if len(X_t) == 0:
            print(f"  Warning: no usable data for {t} after feature engineering. Skipping.")
            continue

        X_list.append(X_t)
        y_list.append(y_t)
        future_list.append(future_ret_t)

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

    # Split
    X_train, X_test, y_train, y_test = time_series_split(X, y, train_ratio=0.8)
    split_index = int(len(X) * 0.8)
    future_ret_train = future_ret.iloc[:split_index]
    future_ret_test = future_ret.iloc[split_index:]

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
        import itertools

        # Define param grid
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

        print("\n=== XGBoost Hyperparameter Tuning (by P&L) ===")

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

            # Backtest at the fixed trading threshold
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
                f"Win: {win_rate:.3f}, TotRet: {total_ret:.3f}, AvgRet: {avg_ret:.4f}"
            )

            # Selection criterion: maximize total_return,
            # but ignore configs with too few trades
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

        # Now best_model is your tuned model
        xgb_clf = best_model

        # ---------- Final evaluation of tuned XGBoost ----------
        print("\n=== XGBoost (tuned, best by P&L) ===")
        y_pred_xgb = xgb_clf.predict(X_test)
        evaluate_model("XGBoost (tuned by P&L)", y_test, y_pred_xgb)

        # ---------- Threshold analysis for multiple thresholds ----------
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

                # ---------- Next-day predictions per ticker (ENSEMBLE) ----------
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


    else:
        print("XGBoost not available; skipping XGBoost tuning.")
        xgb_clf = None

    # ---------- LightGBM tuned by P&L ----------
    if HAS_LIGHTGBM:
        import itertools

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
        lgb_best_model = None

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
                f"Win: {win_rate:.3f}, TotRet: {total_ret:.3f}, AvgRet: {avg_ret:.4f}"
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

        # Final evaluation for best LightGBM
        lgb_clf = lgb_best_model
        y_pred_lgb = lgb_clf.predict(X_test)
        evaluate_model("LightGBM (tuned by P&L)", y_test, y_pred_lgb)

    else:
        print("\nLightGBM not installed, skipping LightGBM model.")


    # ---------- MLP tuned by P&L ----------
    print("\n=== MLP Hyperparameter Tuning (by P&L) ===")
    import itertools

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
    mlp_best_model = None

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
            f"Win: {win_rate:.3f}, TotRet: {total_ret:.3f}, AvgRet: {avg_ret:.4f}"
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

        # ---------- 4-model Ensemble: LR + XGB + LGBM + MLP ----------
    print("\n=== Ensemble (LR + XGB + LGBM + MLP) ===")

    proba_ens = ensemble_predict_proba(
        X_test,
        lr_model=lr_model,
        xgb_model=xgb_clf,
        lgb_model=lgb_clf,
        mlp_model=mlp_clf,
        scaler=scaler,
    )

    # Backtest using your trading threshold (P(up) >= TRADE_THRESHOLD)
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

    #     # === Transformer (sequence model, 30-day window) ===
    # try:
    #     import torch
    #     import torch.nn as nn
    #     import torch.optim as optim
    #     from torch.utils.data import TensorDataset, DataLoader
    #     HAS_TORCH = True
    # except ImportError:
    #     HAS_TORCH = False

    # if HAS_TORCH:
    #     print("\n=== Transformer Hyperparameter Tuning (by P&L) ===")

    #     # 1) Scale features for the Transformer
    #     scaler_tf = StandardScaler()
    #     X_all_scaled = scaler_tf.fit_transform(X)
    #     X_all_scaled = pd.DataFrame(X_all_scaled, index=X.index, columns=X.columns)

    #     # 2) Build 30-day sequence dataset
    #     X_seq, y_seq, ret_seq = build_sequence_dataset(
    #         X_all_scaled, y, future_ret, seq_len=SEQ_LEN
    #     )

    #     n_seq = len(X_seq)
    #     split_idx_seq = int(n_seq * 0.8)

    #     X_train_seq = X_seq[:split_idx_seq]
    #     X_test_seq  = X_seq[split_idx_seq:]
    #     y_train_seq = y_seq[:split_idx_seq]
    #     y_test_seq  = y_seq[split_idx_seq:]
    #     ret_test_seq = ret_seq[split_idx_seq:]

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     X_train_t = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    #     y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).float().to(device)
    #     X_test_t  = torch.tensor(X_test_seq,  dtype=torch.float32).to(device)
    #     y_test_t  = torch.tensor(y_test_seq,  dtype=torch.float32).float().to(device)

    #     num_features = X_seq.shape[-1]

    #     class SimpleTransformerClassifier(nn.Module):
    #         def __init__(self, num_features, d_model=64, num_layers=2,
    #                      nhead=4, dim_feedforward=128, dropout=0.1):
    #             super().__init__()
    #             self.input_proj = nn.Linear(num_features, d_model)
    #             encoder_layer = nn.TransformerEncoderLayer(
    #                 d_model=d_model,
    #                 nhead=nhead,
    #                 dim_feedforward=dim_feedforward,
    #                 dropout=dropout,
    #                 batch_first=True,
    #             )
    #             self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    #             self.cls_head = nn.Linear(d_model, 1)

    #         def forward(self, x):
    #             # x: (batch, seq_len, num_features)
    #             x = self.input_proj(x)      # (batch, seq_len, d_model)
    #             x = self.encoder(x)         # (batch, seq_len, d_model)
    #             x = x[:, -1, :]             # use last time step as summary
    #             logits = self.cls_head(x).squeeze(-1)  # (batch,)
    #             return logits

    #     def train_one_transformer(model, lr=1e-3, epochs=10, batch_size=64):
    #         model.to(device)
    #         criterion = nn.BCEWithLogitsLoss()
    #         optimizer = optim.Adam(model.parameters(), lr=lr)

    #         dataset = TensorDataset(X_train_t, y_train_t)
    #         loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    #         model.train()
    #         for _ in range(epochs):
    #             for xb, yb in loader:
    #                 optimizer.zero_grad()
    #                 logits = model(xb)
    #                 loss = criterion(logits, yb)
    #                 loss.backward()
    #                 optimizer.step()

    #     def evaluate_transformer(model):
    #         model.eval()
    #         with torch.no_grad():
    #             logits = model(X_test_t)
    #             proba = torch.sigmoid(logits).cpu().numpy()

    #         # use sequence labels and returns for backtest
    #         stats = backtest_with_threshold(
    #             y_true=y_test_seq,
    #             proba=proba,
    #             returns=ret_test_seq,
    #             threshold=TRADE_THRESHOLD,
    #         )
    #         return stats, proba

    #     # 3) Hyperparameter search for the Transformer
    #     tf_param_grid = [
    #         {"d_model": 64,  "num_layers": 2, "lr": 1e-3},
    #         {"d_model": 128, "num_layers": 2, "lr": 1e-3},
    #         {"d_model": 64,  "num_layers": 3, "lr": 5e-4},
    #     ]

    #     best_tf_score = -np.inf
    #     best_tf_params = None
    #     best_tf_stats = None
    #     best_tf_model = None
    #     best_tf_proba = None

    #     for params in tf_param_grid:
    #         model_tf = SimpleTransformerClassifier(
    #             num_features=num_features,
    #             d_model=params["d_model"],
    #             num_layers=params["num_layers"],
    #             nhead=4,
    #             dim_feedforward=128,
    #             dropout=0.1,
    #         )

    #         train_one_transformer(
    #             model_tf,
    #             lr=params["lr"],
    #             epochs=10,
    #             batch_size=64,
    #         )

    #         stats, proba = evaluate_transformer(model_tf)

    #         acc = stats["accuracy"]
    #         num_trades = stats["num_trades"]
    #         win_rate = stats["win_rate"]
    #         total_ret = stats["total_return"]
    #         avg_ret = stats["avg_return"]

    #         print(
    #             f"Transformer Params {params} → "
    #             f"Acc: {acc:.3f}, Trades: {num_trades}, "
    #             f"Win: {win_rate:.3f}, TotRet: {total_ret:.3f}, AvgRet: {avg_ret:.4f}"
    #         )

    #         if num_trades >= MIN_TRADES_FOR_MODEL and total_ret > best_tf_score:
    #             best_tf_score = total_ret
    #             best_tf_params = params
    #             best_tf_stats = stats
    #             best_tf_model = model_tf
    #             best_tf_proba = proba

    #     print("\n=== Best Transformer config (by total_return at "
    #           f"threshold {TRADE_THRESHOLD:.2f}) ===")
    #     print("Best Transformer params:", best_tf_params)
    #     print("Best Transformer stats:", best_tf_stats)

    #     # Final classification metrics for the best Transformer
    #     if best_tf_model is not None:
    #         best_tf_model.eval()
    #         with torch.no_grad():
    #             logits = best_tf_model(X_test_t)
    #             proba = torch.sigmoid(logits).cpu().numpy()

    #         y_pred_labels = (proba >= 0.5).astype(int)

    #         print("\n=== Transformer (tuned by P&L, 30-day window) ===")
    #         print("Accuracy:", accuracy_score(y_test_seq, y_pred_labels))
    #         print("\nClassification Report:")
    #         print(classification_report(y_test_seq, y_pred_labels))
    #         print("Confusion Matrix:")
    #         print(confusion_matrix(y_test_seq, y_pred_labels))
    # else:
    #     print("\nPyTorch not installed; skipping Transformer model.")


    # Save artifacts for inference
    artifact_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    dump(scaler,   os.path.join(artifact_dir, "scaler.joblib"))
    dump(lr_model, os.path.join(artifact_dir, "lr_model.joblib"))

    if xgb_clf is not None:
        dump(xgb_clf, os.path.join(artifact_dir, "xgb_model.joblib"))

    if lgb_clf is not None:
        dump(lgb_clf, os.path.join(artifact_dir, "lgb_model.joblib"))

    if mlp_clf is not None:
        dump(mlp_clf, os.path.join(artifact_dir, "mlp_model.joblib"))

    print(f"\nArtifacts saved to: {artifact_dir}")

    print("\nDone.")

if __name__ == "__main__":
    main()
