"""Reproducible model training and out-of-fold evaluation."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

from construction_cycle.data import CATEGORICAL_FEATURES, FEATURES

MODEL_PARAMS = dict(iterations=600, depth=5, learning_rate=0.03, random_seed=42)


def new_model(iterations: int = 600) -> CatBoostRegressor:
    return CatBoostRegressor(
        **{**MODEL_PARAMS, "iterations": iterations}, verbose=False, allow_writing_files=False,
    )


def make_splits(data: pd.DataFrame, folds: int = 5) -> list:
    """Require sufficient examples to represent every plan in every fold."""
    if folds < 2 or data["plan_code"].value_counts().min() < folds or len(data) < 2 * folds:
        raise ValueError("Use at least two folds, two validation rows per fold, and one example per plan per fold.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return list(cv.split(data[FEATURES], data["plan_code"]))


def cross_validate(data: pd.DataFrame, splits: list, iterations: int = 600):
    """Evaluate CatBoost and a training-fold median baseline on identical splits."""
    rows = []
    oof = np.full(len(data), np.nan)
    y = data["cycle_days"]
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        model = new_model(iterations)
        model.fit(data.iloc[train_idx][FEATURES], y.iloc[train_idx], cat_features=CATEGORICAL_FEATURES)
        predictions = model.predict(data.iloc[val_idx][FEATURES])
        oof[val_idx] = predictions
        baseline = np.full(len(val_idx), y.iloc[train_idx].median())
        rows.append({
            "fold": fold,
            "train_r2": r2_score(y.iloc[train_idx], model.predict(data.iloc[train_idx][FEATURES])),
            "validation_r2": r2_score(y.iloc[val_idx], predictions),
            "mae_days": mean_absolute_error(y.iloc[val_idx], predictions),
            "rmse_days": np.sqrt(mean_squared_error(y.iloc[val_idx], predictions)),
            "baseline_mae_days": mean_absolute_error(y.iloc[val_idx], baseline),
            "baseline_rmse_days": np.sqrt(mean_squared_error(y.iloc[val_idx], baseline)),
        })
    return pd.DataFrame(rows), oof


def manager_analysis(data: pd.DataFrame, splits: list, iterations: int = 600) -> pd.DataFrame:
    """Summarize actual-minus-expected duration without treating it as causal."""
    features = [c for c in FEATURES if c != "site_manager"]
    categorical = [c for c in CATEGORICAL_FEATURES if c != "site_manager"]
    expected = np.full(len(data), np.nan)
    for train_idx, val_idx in splits:
        model = new_model(iterations)
        model.fit(data.iloc[train_idx][features], data.iloc[train_idx]["cycle_days"], cat_features=categorical)
        expected[val_idx] = model.predict(data.iloc[val_idx][features])
    residuals = data.assign(residual=data["cycle_days"] - expected)
    results = residuals.groupby("site_manager")["residual"].agg(["count", "mean", "median", "std"])
    results["se"] = results["std"] / np.sqrt(results["count"])
    # Descriptive normal intervals do not account for model or assignment uncertainty.
    results["ci_low"] = results["mean"] - 1.96 * results["se"]
    results["ci_high"] = results["mean"] + 1.96 * results["se"]
    return results.sort_values("mean")
