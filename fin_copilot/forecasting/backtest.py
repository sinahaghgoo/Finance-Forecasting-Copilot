from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fin_copilot.config import ForecastConfig
from fin_copilot.forecasting.features import TARGET
from fin_copilot.forecasting.model import QuantileForecaster


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom < 1e-9, 1e-9, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    loss = np.maximum(quantile * diff, (quantile - 1) * diff)
    return float(np.mean(loss))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    return float(np.mean(inside))


@dataclass
class BacktestResult:
    per_fold: pd.DataFrame
    per_series_fold: pd.DataFrame
    predictions: pd.DataFrame

    def summary(self) -> pd.Series:
        return self.per_fold.mean(numeric_only=True).round(4)


def walk_forward(
    panel: pd.DataFrame,
    cfg: ForecastConfig,
    n_folds: int,
    fold_horizon_days: int,
) -> BacktestResult:
    panel = panel.sort_values(["category", "region", "date"]).reset_index(drop=True)
    dates = np.sort(panel["date"].unique())
    end_date = pd.Timestamp(dates[-1])

    fold_rows, series_rows, pred_rows = [], [], []

    for fold in range(n_folds):
        offset = (n_folds - 1 - fold) * fold_horizon_days
        split = end_date - pd.Timedelta(days=offset + fold_horizon_days)
        test_start = split + pd.Timedelta(days=1)
        test_end = split + pd.Timedelta(days=fold_horizon_days)

        train = panel[panel["date"] <= split].copy()
        test = panel[(panel["date"] >= test_start) & (panel["date"] <= test_end)].copy()
        if train.empty or test.empty:
            continue

        model = QuantileForecaster(quantiles=cfg.quantiles, params=cfg.lgbm).fit(train)
        pred = model.predict_recursive(train, horizon_days=fold_horizon_days)
        merged = pred.merge(
            test[["category", "region", "date", TARGET]], on=["category", "region", "date"], how="inner"
        )
        merged["fold"] = fold

        y = merged[TARGET].values
        p = merged["point"].values
        lo = merged[f"q{int(min(cfg.quantiles)*100):02d}"].values
        hi = merged[f"q{int(max(cfg.quantiles)*100):02d}"].values

        fold_rows.append({
            "fold": fold,
            "train_end": split.date().isoformat(),
            "test_start": test_start.date().isoformat(),
            "test_end": test_end.date().isoformat(),
            "mape": mape(y, p),
            "smape": smape(y, p),
            "rmse": rmse(y, p),
            "pinball_10": pinball_loss(y, lo, min(cfg.quantiles)),
            "pinball_90": pinball_loss(y, hi, max(cfg.quantiles)),
            "coverage_80": coverage(y, lo, hi),
        })

        for (cat, reg), sub in merged.groupby(["category", "region"]):
            series_rows.append({
                "fold": fold,
                "category": cat,
                "region": reg,
                "mape": mape(sub[TARGET].values, sub["point"].values),
                "coverage_80": coverage(
                    sub[TARGET].values,
                    sub[f"q{int(min(cfg.quantiles)*100):02d}"].values,
                    sub[f"q{int(max(cfg.quantiles)*100):02d}"].values,
                ),
            })

        pred_rows.append(merged)

    return BacktestResult(
        per_fold=pd.DataFrame(fold_rows),
        per_series_fold=pd.DataFrame(series_rows),
        predictions=pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame(),
    )
