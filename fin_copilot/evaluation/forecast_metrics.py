from __future__ import annotations

import pandas as pd

from fin_copilot.forecasting.backtest import BacktestResult


def summary_table(result: BacktestResult) -> pd.DataFrame:
    if result.per_fold.empty:
        return pd.DataFrame()
    agg = (
        result.per_fold[["mape", "smape", "rmse", "pinball_10", "pinball_90", "coverage_80"]]
        .agg(["mean", "std"])
        .round(4)
    )
    return agg


def per_series_summary(result: BacktestResult) -> pd.DataFrame:
    if result.per_series_fold.empty:
        return pd.DataFrame()
    return (
        result.per_series_fold.groupby(["category", "region"])
        .agg(mape_mean=("mape", "mean"), mape_std=("mape", "std"),
             coverage_mean=("coverage_80", "mean"))
        .round(4)
        .reset_index()
    )
