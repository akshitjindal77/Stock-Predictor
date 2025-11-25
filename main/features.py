import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add core technical indicators to the dataframe:
    Returns, EMAs, MACD, RSI, Bollinger Bands, Volatility.
    """
    df = df.copy()

    # --- Daily return ---
    df["Return"] = df["Close"].pct_change()

    # --- EMAs ---
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # --- MACD ---
    df["MACD_Line"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]

    # --- RSI 14 ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands (20) ---
    middle = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    lower_band = middle - 2 * std
    upper_band = middle + 2 * std
    width = upper_band - lower_band

    df["BB_Middle"] = middle
    df["BB_Upper"] = upper_band
    df["BB_Lower"] = lower_band
    df["BB_Width"] = width
    df["BB_Position"] = (df["Close"] - lower_band) / width

    # --- Volatility ---
    df["Vol_20"] = df["Return"].rolling(20).std()

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add more advanced engineered features:
    lags, ratios, rolling stats, ranges, volume features, candle features.
    """
    df = df.copy()

    # --- Lagged returns (momentum) ---
    df["Return_1"] = df["Return"].shift(1)
    df["Return_2"] = df["Return"].shift(2)
    df["Return_3"] = df["Return"].shift(3)
    df["Return_5"] = df["Return"].shift(5)

    # --- Rolling returns (short-term momentum) ---
    df["Rolling_Return_3"] = df["Return"].rolling(3).mean()
    df["Rolling_Return_7"] = df["Return"].rolling(7).mean()

    # --- Trend ratios ---
    df["EMA_ratio"] = df["EMA_12"] / df["EMA_26"]
    df["Price_to_EMA12"] = df["Close"] / df["EMA_12"]
    df["Price_to_EMA26"] = df["Close"] / df["EMA_26"]

    # --- Rolling price stats (5-day window) ---
    df["Rolling_mean_5"] = df["Close"].rolling(5).mean()
    df["Rolling_std_5"] = df["Close"].rolling(5).std()
    df["Rolling_min_5"] = df["Close"].rolling(5).min()
    df["Rolling_max_5"] = df["Close"].rolling(5).max()

    # --- Intraday ranges ---
    df["High_Low_Range"] = df["High"] - df["Low"]
    df["Close_Open_Range"] = df["Close"] - df["Open"]

    # --- Volume features ---
    df["Vol_MA_5"] = df["Volume"].rolling(5).mean()
    df["Vol_std_5"] = df["Volume"].rolling(5).std()

    # --- Candle shape features ---
    df["Upper_wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["Lower_wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["Body_size"] = (df["Close"] - df["Open"]).abs()

    return df


def add_target(df: pd.DataFrame, threshold: float = 0.0, horizon: int = 1,) -> pd.DataFrame:
    df = df.copy()
    # future price after `horizon` days
    future_price = df["Close"].shift(-horizon)

    # future % return over that horizon
    future_return = future_price / df["Close"] - 1

    df["Future_Return"] = future_return
    df["Target"] = (future_return > threshold).astype(int)

    return df



def build_dataset(df: pd.DataFrame, threshold: float = 0.002, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    """
    Build the ML dataset from raw price data.

    threshold: e.g. 0.002 for +0.2% move
    horizon:   how many days ahead to predict (1, 3, 5, etc.)
    """
    df = add_technical_indicators(df)
    df = add_engineered_features(df)

    # Add target + Future_Return based on chosen horizon and threshold
    df = add_target(df, threshold=threshold, horizon=horizon)

    # All feature columns we want to use
    feature_cols = [
        # core
        "Return",
        "EMA_12", "EMA_26",
        "MACD_Line", "MACD_Signal", "MACD_Hist",
        "RSI_14",
        "BB_Position",
        "Vol_20",

        # lags
        "Return_1", "Return_2", "Return_3", "Return_5",
        "Rolling_Return_3", "Rolling_Return_7",

        # ratios
        "EMA_ratio", "Price_to_EMA12", "Price_to_EMA26",

        # rolling stats
        "Rolling_mean_5", "Rolling_std_5", "Rolling_min_5", "Rolling_max_5",

        # ranges
        "High_Low_Range", "Close_Open_Range",

        # volume
        "Vol_MA_5", "Vol_std_5",

        # candle structure
        "Upper_wick", "Lower_wick", "Body_size",
    ]

    X = df[feature_cols]
    y = df["Target"]
    future_ret = df["Future_Return"]

    # Drop rows with any NaNs (from rolling windows, shift, etc.)
    data = pd.concat([X, y, future_ret], axis=1).dropna()
    X = data[feature_cols]
    y = data["Target"]
    future_ret = data["Future_Return"]

    return X, y, feature_cols, future_ret

