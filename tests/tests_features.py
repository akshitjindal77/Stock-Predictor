import pandas as pd
from main.features import build_dataset

def make_dummy_df(n=60):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = pd.Series(range(100, 100 + n), index=idx)
    df = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Adj Close": prices,
            "Volume": 1_000_000,
        },
        index=idx,
    )
    return df

def test_build_dataset_shapes():
    df = make_dummy_df()
    X, y, feature_cols, future_ret = build_dataset(
        df, threshold=0.002, horizon=1
    )
    assert len(X) == len(y) == len(future_ret)
    assert not X.isna().any().any()
    assert len(feature_cols) == X.shape[1]
