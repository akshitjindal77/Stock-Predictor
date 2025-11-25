# stock_predictor/models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def make_logistic_regression():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        n_jobs=-1,
    )


def make_random_forest():
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


def make_xgboost():
    """
    XGBoost model (if xgboost is installed).
    """
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
