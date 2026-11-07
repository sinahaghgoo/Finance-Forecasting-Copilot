from __future__ import annotations

import pandas as pd

from fin_copilot.evaluation.forecast_metrics import per_series_summary, summary_table
from fin_copilot.forecasting.backtest import BacktestResult


def _fake_result() -> BacktestResult:
    per_fold = pd.DataFrame([
        {"fold": 0, "mape": 5.0, "smape": 4.5, "rmse": 100.0,
         "pinball_10": 10.0, "pinball_90": 12.0, "coverage_80": 0.80},
        {"fold": 1, "mape": 7.0, "smape": 6.0, "rmse": 120.0,
         "pinball_10": 11.0, "pinball_90": 13.0, "coverage_80": 0.70},
    ])
    per_series = pd.DataFrame([
        {"fold": 0, "category": "A", "region": "EU", "mape": 4.0, "coverage_80": 0.75},
        {"fold": 1, "category": "A", "region": "EU", "mape": 6.0, "coverage_80": 0.70},
        {"fold": 0, "category": "B", "region": "EU", "mape": 10.0, "coverage_80": 0.60},
    ])
    return BacktestResult(per_fold=per_fold, per_series_fold=per_series, predictions=pd.DataFrame())


def test_summary_table_has_mean_and_std_rows():
    s = summary_table(_fake_result())
    assert set(["mean", "std"]).issubset(s.index)
    assert s.loc["mean", "mape"] == 6.0


def test_per_series_summary_groups_by_pair():
    s = per_series_summary(_fake_result())
    assert set(s.columns) >= {"category", "region", "mape_mean", "coverage_mean"}
    a_eu = s[(s["category"] == "A") & (s["region"] == "EU")].iloc[0]
    assert a_eu["mape_mean"] == 5.0


def test_summary_empty_result_returns_empty():
    empty = BacktestResult(per_fold=pd.DataFrame(), per_series_fold=pd.DataFrame(), predictions=pd.DataFrame())
    assert summary_table(empty).empty
    assert per_series_summary(empty).empty
