from __future__ import annotations

import pandas as pd

from fin_copilot.data.generate import generate_panel


def test_panel_shape_and_columns():
    df = generate_panel("2023-01-01", "2023-06-30", ["Footwear", "Apparel"], ["EMEA"], seed=1)
    expected_cols = {"date", "category", "region", "revenue", "units_sold", "avg_price", "cogs", "gross_profit"}
    assert expected_cols.issubset(df.columns)
    assert len(df) == 2 * 181


def test_panel_values_are_positive():
    df = generate_panel("2023-01-01", "2023-03-31", ["Footwear"], ["EMEA"], seed=1)
    assert (df["revenue"] > 0).all()
    assert (df["units_sold"] > 0).all()
    assert (df["avg_price"] > 0).all()
    assert (df["gross_profit"] > 0).all()
    assert (df["gross_profit"] < df["revenue"]).all()


def test_seed_is_reproducible():
    a = generate_panel("2023-01-01", "2023-02-01", ["Footwear"], ["EMEA"], seed=7)
    b = generate_panel("2023-01-01", "2023-02-01", ["Footwear"], ["EMEA"], seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_covid_dip_present():
    df = generate_panel("2020-01-01", "2020-12-31", ["Apparel"], ["EMEA"], seed=42)
    df = df.set_index("date")
    pre = df.loc["2020-02-01":"2020-02-28", "revenue"].mean()
    dip = df.loc["2020-04-01":"2020-04-30", "revenue"].mean()
    assert dip < pre * 0.95
