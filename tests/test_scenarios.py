from __future__ import annotations

import pandas as pd

from fin_copilot.scenarios.gross_profit import ScenarioParams, apply_scenario, run_scenario


def _baseline():
    return pd.DataFrame({
        "category": ["Footwear"] * 3,
        "region": ["EMEA"] * 3,
        "date": pd.date_range("2024-01-01", periods=3, freq="D"),
        "revenue": [100.0, 110.0, 120.0],
        "units_sold": [10.0, 11.0, 12.0],
        "avg_price": [10.0, 10.0, 10.0],
        "cogs": [50.0, 55.0, 60.0],
        "cogs_rate": [0.5, 0.5, 0.5],
        "gross_profit": [50.0, 55.0, 60.0],
    })


def test_no_change_leaves_numbers_untouched():
    b = _baseline()
    s = apply_scenario(b, ScenarioParams())
    pd.testing.assert_series_equal(s["revenue"], b["revenue"], check_names=False)
    pd.testing.assert_series_equal(s["gross_profit"], b["gross_profit"], check_names=False)


def test_price_increase_without_elasticity_lifts_gross_profit():
    b = _baseline()
    s = apply_scenario(b, ScenarioParams(price_change_pct=0.10, price_elasticity=0.0))
    assert (s["revenue"] > b["revenue"]).all()
    assert (s["gross_profit"] > b["gross_profit"]).all()


def test_price_increase_with_unit_elasticity_keeps_revenue_but_lifts_gp():
    b = _baseline()
    s = apply_scenario(b, ScenarioParams(price_change_pct=0.10, price_elasticity=-1.0))
    assert abs(s["revenue"].sum() - b["revenue"].sum() * 0.99) < 1e-6
    assert s["gross_profit"].sum() > b["gross_profit"].sum()


def test_cogs_increase_reduces_gross_profit():
    b = _baseline()
    s = apply_scenario(b, ScenarioParams(cogs_change_pct=0.20))
    assert (s["cogs"] > b["cogs"]).all()
    assert (s["gross_profit"] < b["gross_profit"]).all()


def test_run_scenario_summary_shape():
    hist = pd.DataFrame({
        "category": ["Footwear"] * 90,
        "region": ["EMEA"] * 90,
        "date": pd.date_range("2024-01-01", periods=90, freq="D"),
        "revenue": [1000.0] * 90,
        "units_sold": [100.0] * 90,
        "avg_price": [10.0] * 90,
        "cogs": [500.0] * 90,
        "gross_profit": [500.0] * 90,
    })
    pred = pd.DataFrame({
        "category": ["Footwear"] * 14,
        "region": ["EMEA"] * 14,
        "date": pd.date_range("2024-04-01", periods=14, freq="D"),
        "point": [1000.0] * 14,
        "q10": [900.0] * 14,
        "q50": [1000.0] * 14,
        "q90": [1100.0] * 14,
    })
    r = run_scenario(hist, pred, ScenarioParams(price_change_pct=0.05, price_elasticity=-1.0))
    s = r.summary()
    assert set(["baseline_revenue", "scenario_revenue", "delta_gross_profit", "delta_gross_profit_pct"]).issubset(s)
    assert s["delta_gross_profit"] > 0
