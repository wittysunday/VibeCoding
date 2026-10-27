"""Gradient-boosted classifier that scores each stock's chance of a good forward return."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .indicators import FEATURES


@dataclass
class ModelConfig:
    horizon: int = 20            # trading days to hold
    target_return: float = 0.05  # "good buy" = forward return above this
    test_fraction: float = 0.25  # most recent share of dates held out for validation
    min_history: int = 60        # rows a symbol needs before it is scored


def add_target(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    df = df.copy()
    fwd = df.groupby("symbol")["close"].shift(-cfg.horizon) / df["close"] - 1
    df["fwd_return"] = fwd
    df["target"] = np.where(fwd.isna(), np.nan, (fwd > cfg.target_return).astype(float))
    df["history_len"] = df.groupby("symbol").cumcount() + 1
    return df


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=42,
    )


def train_and_validate(df: pd.DataFrame, cfg: ModelConfig):
    """Walk-forward split: train on older dates, validate on the most recent ones.

    A gap of `horizon` days separates train and test so no training label
    overlaps the test period.
    """
    data = df[(df["history_len"] >= cfg.min_history) & df["target"].notna()]
    dates = np.sort(data["date"].unique())
    if len(dates) < 4 * cfg.horizon:
        raise ValueError(f"Need at least {4 * cfg.horizon} trading days of labelled history; got {len(dates)}.")

    split = int(len(dates) * (1 - cfg.test_fraction))
    train_end, test_start = dates[max(split - cfg.horizon, 1)], dates[split]
    train, test = data[data["date"] < train_end], data[data["date"] >= test_start]

    model = _new_model().fit(train[FEATURES], train["target"])
    test = test.assign(prob=model.predict_proba(test[FEATURES])[:, 1])

    top = test[test["prob"] >= test["prob"].quantile(0.9)]
    metrics = {
        "train_rows": len(train), "test_rows": len(test),
        "train_period": f"{pd.Timestamp(train['date'].min()).date()} to {pd.Timestamp(train['date'].max()).date()}",
        "test_period": f"{pd.Timestamp(test['date'].min()).date()} to {pd.Timestamp(test['date'].max()).date()}",
        "base_rate": test["target"].mean(),
        "auc": roc_auc_score(test["target"], test["prob"]) if test["target"].nunique() > 1 else np.nan,
        "top_decile_hit_rate": top["target"].mean(),
        "top_decile_avg_fwd_return": top["fwd_return"].mean(),
        "all_avg_fwd_return": test["fwd_return"].mean(),
    }

    # Refit on every labelled row for live scoring.
    final = _new_model().fit(data[FEATURES], data["target"])
    return final, test, metrics


def feature_importance(model, df: pd.DataFrame, n_repeats: int = 5) -> pd.Series:
    from sklearn.inspection import permutation_importance

    sample = df.dropna(subset=["target"]).tail(5000)
    r = permutation_importance(model, sample[FEATURES], sample["target"], n_repeats=n_repeats,
                               random_state=0, scoring="roc_auc")
    return pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)
