from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# --- XGBoost ---
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    XGBClassifier = None
    HAS_XGBOOST = False

# --- LightGBM ---
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    LGBMClassifier = None
    HAS_LIGHTGBM = False

# --- MLP (neural network) ---
from sklearn.neural_network import MLPClassifier


def make_logistic_regression():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1,
    )


def make_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )


def make_xgboost(
    max_depth=3,
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
):
    if not HAS_XGBOOST:
        raise RuntimeError("XGBoost is not installed.")
    return XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )


def make_lightgbm(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
):
    if not HAS_LIGHTGBM:
        raise RuntimeError("LightGBM is not installed.")
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1,
        verbose=-1, 
    )


def make_mlp(
    hidden_layer_sizes=(128, 64),
    alpha=1e-4,
    learning_rate_init=1e-3,
):
    # Simple feedforward neural net
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=200,
        random_state=42,
    )
