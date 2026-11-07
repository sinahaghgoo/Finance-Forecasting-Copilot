from __future__ import annotations

import pandas as pd
import pytest

from fin_copilot.data import online_retail


def test_assign_category_keyword_matching():
    assert online_retail._assign_category("WHITE HANGING HEART T-LIGHT HOLDER") == "Decor"
    assert online_retail._assign_category("LUNCH BAG RED RETROSPOT") == "Bags & Storage"
    assert online_retail._assign_category("SET OF 3 CAKE TINS PANTRY") == "Bags & Storage"
    assert online_retail._assign_category("NOTEBOOK A5 BLACK") == "Stationery"
    assert online_retail._assign_category("GIFT WRAP RED RIBBON") == "Gifts & Cards"
    assert online_retail._assign_category("RETRO FLORAL MUG") == "Home & Kitchen"
    assert online_retail._assign_category("MYSTERY ITEM XYZ") == "Other"
    assert online_retail._assign_category(None) == "Other"
    assert online_retail._assign_category("") == "Other"


@pytest.fixture
def fake_transactions(monkeypatch):
    dates = pd.date_range("2010-01-01", "2010-12-31", freq="D")
    rows = []
    for d in dates:
        rows.extend([
            {"invoice": "I1", "stock_code": "A1", "description": "HEART DECORATION", "quantity": 2,
             "invoice_date": d, "price": 5.0, "customer_id": 1, "country": "United Kingdom", "sheet": "x"},
            {"invoice": "I2", "stock_code": "B1", "description": "LUNCH BAG RED", "quantity": 1,
             "invoice_date": d, "price": 3.5, "customer_id": 2, "country": "Germany", "sheet": "x"},
            {"invoice": "CX1", "stock_code": "A1", "description": "HEART DECORATION", "quantity": -2,
             "invoice_date": d, "price": 5.0, "customer_id": 3, "country": "United Kingdom", "sheet": "x"},
        ])
    tx = pd.DataFrame(rows)

    def fake_download_raw(force: bool = False):
        return online_retail.RAW_XLSX

    def fake_load_raw():
        return tx

    monkeypatch.setattr(online_retail, "download_raw", fake_download_raw)
    monkeypatch.setattr(online_retail, "_load_raw", fake_load_raw)
    return tx


def test_to_panel_filters_credit_notes_and_builds_schema(fake_transactions):
    panel = online_retail.to_panel(save=False)
    expected = {"date", "category", "region", "revenue", "units_sold", "avg_price", "cogs", "gross_profit"}
    assert expected.issubset(panel.columns)
    assert (panel["revenue"] > 0).all()
    assert "Other" not in panel["category"].unique()


def test_to_panel_drops_low_volume_countries(fake_transactions):
    panel = online_retail.to_panel(save=False)
    assert set(panel["region"].unique()).issubset({"United Kingdom", "Germany"})
