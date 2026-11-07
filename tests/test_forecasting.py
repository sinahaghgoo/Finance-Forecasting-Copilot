from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_copilot.config import LGBMParams
from fin_copilot.data.generate import generate_panel
from fin_copilot.forecasting.backtest import coverage, mape, pinball_loss, walk_forward
from fin_copilot.forecasting.features import build_features
from fin_copilot.forecasting.model import QuantileForecaster


def _small_panel():
    return generate_panel("2022-01-01", "2024-02-28", ["Footwear", "Apparel"], ["EMEA"], seed=11)


def test_features_add_expected_columns():
    df = generate_panel("2022-01-01", "2023-12-31", ["Footwear"], ["EMEA"], seed=3)
    feats = build_features(df)
    for col in ["year", "month", "dow", "sin_doy", "cos_doy", "lag_7", "lag_365", "roll_mean_7", "roll_std_28"]:
        assert col in feats.columns


def test_lag_365_is_shifted_correctly():
    df = generate_panel("2022-01-01", "2023-12-31", ["Footwear"], ["EMEA"], seed=3)
    feats = build_features(df).sort_values("date").reset_index(drop=True)
    row = feats.iloc[400]
    expected_lag = feats.loc[feats["date"] == row["date"] - pd.Timedelta(days=365), "revenue"].iloc[0]
    assert np.isclose(row["lag_365"], expected_lag)


def test_quantile_forecaster_fits_and_predicts():
    df = _small_panel()
    params = LGBMParams(n_estimators=120, num_leaves=31, min_data_in_leaf=20)
    model = QuantileForecaster(quantiles=[0.1, 0.5, 0.9], params=params).fit(df)
    pred = model.predict_recursive(df, horizon_days=7)
    assert len(pred) == 2 * 7
    assert (pred["q10"] <= pred["q50"] + 1e-6).mean() > 0.9
    assert (pred["q50"] <= pred["q90"] + 1e-6).mean() > 0.9


def test_mape_sanity():
    y = np.array([100.0, 200.0, 300.0])
    assert mape(y, y) == pytest.approx(0.0, abs=1e-6)
    assert mape(y, y * 1.1) == pytest.approx(10.0, abs=1e-6)


def test_pinball_asymmetric():
    y = np.array([10.0, 10.0])
    under = np.array([8.0, 8.0])
    over = np.array([12.0, 12.0])
    assert pinball_loss(y, under, 0.9) > pinball_loss(y, under, 0.1)
    assert pinball_loss(y, over, 0.1) > pinball_loss(y, over, 0.9)


def test_walk_forward_returns_metrics():
    df = _small_panel()
    from fin_copilot.config import ForecastConfig

    cfg = ForecastConfig(
        horizon_days=7,
        quantiles=[0.1, 0.5, 0.9],
        lgbm=LGBMParams(n_estimators=120, num_leaves=31, min_data_in_leaf=20),
    )
    result = walk_forward(df, cfg, n_folds=2, fold_horizon_days=7)
    assert not result.per_fold.empty
    assert set(["mape", "rmse", "coverage_80"]).issubset(result.per_fold.columns)
    assert (result.per_fold["mape"] < 50).all()


def test_coverage_bounds():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.0, 1.0, 2.0, 5.0])
    hi = np.array([2.0, 3.0, 4.0, 10.0])
    assert coverage(y, lo, hi) == 0.75
