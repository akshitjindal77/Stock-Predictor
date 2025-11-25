import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data_loader import download_stock_data
from .features import build_dataset
from .models import (
    make_logistic_regression,
    make_random_forest,
    make_xgboost,
    HAS_XGBOOST,
)


def time_series_split(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    return X_train, X_test, y_train, y_test


def evaluate_model(name, y_test, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def predict_next_day_for_ticker(ticker, model, start="2023-01-01", end=None, *, threshold=0.002, horizon=1):
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
        "max_depth":      [3, 4, 5],
        "n_estimators":   [100, 200],
        "learning_rate":  [0.05, 0.1],
        "subsample":      [0.8, 1.0],
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

    print("\n=== XGBoost Hyperparameter Tuning ===")

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

    print("\n=== Best XGBoost config ===")
    print("Best accuracy:", best_acc)
    print("Best params:", best_params)

    return best_model, best_params


def main():

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

    # Sort each piece independently; using .loc with duplicated date indices would
    # duplicate rows and break alignment between X and y.
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

    if HAS_XGBOOST:
        best_xgb, best_params = tune_xgboost(
            X_train, y_train, X_test, y_test, future_ret_test
        )

        y_pred_xgb = best_xgb.predict(X_test)
        evaluate_model("XGBoost (tuned)", y_test, y_pred_xgb)

        y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
        print("\nXGBoost (tuned) probability examples (first 10):")
        print(y_proba_xgb[:10])

        thresholds = [0.50, 0.55, 0.60, 0.65]
        print("\n=== Threshold analysis (XGBoost, tuned) ===")
        for thresh in thresholds:
            y_pred_thresh = (y_proba_xgb >= thresh).astype(int)
            acc = accuracy_score(y_test, y_pred_thresh)

            trade_mask = (y_pred_thresh == 1)
            trades = trade_mask.sum()

            if trades > 0:
                wins = ((y_pred_thresh == 1) & (y_test == 1)).sum()
                win_rate = wins / trades
                trade_returns = future_ret_test[trade_mask]
                total_return = trade_returns.sum()
                avg_trade_return = trade_returns.mean()
            else:
                win_rate = 0.0
                total_return = 0.0
                avg_trade_return = 0.0

            print(
                f"Threshold {thresh:.2f} → "
                f"Accuracy: {acc:.3f}, Trades: {trades}, "
                f"Win rate: {win_rate:.3f}, "
                f"Total return: {total_return:.3f}, "
                f"Avg/trade: {avg_trade_return:.4f}"
            )

        # 5.3: Predict next day for each ticker with the tuned model
        print("\n=== Next-day predictions (tuned XGBoost) ===")
        for t in tickers:
            try:
                predict_next_day_for_ticker(
                    t,
                    best_xgb,
                    start="2023-01-01",
                    threshold=threshold,
                    horizon=horizon,
                )
            except Exception as exc:
                print(f"  Skipping {t}: {exc}")

    print("\nDone.")



if __name__ == "__main__":
    main()
