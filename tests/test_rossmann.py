from __future__ import annotations

import pandas as pd
import pytest

from fin_copilot.data import rossmann


@pytest.fixture
def fake_raw(monkeypatch):
    dates = pd.date_range("2013-01-01", "2013-06-30", freq="D")
    rng_seed = 0

    def stores_for(storetype: str, assortment: str, count: int):
        nonlocal rng_seed
        base_store = rng_seed
        rng_seed += count
        return [{"Store": base_store + i, "StoreType": storetype, "Assortment": assortment} for i in range(count)]

    big_group = stores_for("a", "a", 50)
    small_group = stores_for("b", "a", 5)
    stores = pd.DataFrame(big_group + small_group)

    train_rows = []
    for d in dates:
        for s in big_group:
            train_rows.append({"Store": s["Store"], "Date": d, "Sales": 1000 + (s["Store"] % 7) * 100,
                               "Customers": 100, "Open": 1})
        for s in small_group:
            train_rows.append({"Store": s["Store"], "Date": d, "Sales": 500, "Customers": 50, "Open": 1})
    for s in big_group[:5]:
        train_rows.append({"Store": s["Store"], "Date": pd.Timestamp("2013-07-01"),
                           "Sales": 10, "Customers": 2, "Open": 1})

    train = pd.DataFrame(train_rows)

    def fake_load_raw():
        return train, stores

    def fake_download_raw(force: bool = False):
        return rossmann.RAW_DIR

    monkeypatch.setattr(rossmann, "_load_raw", fake_load_raw)
    monkeypatch.setattr(rossmann, "download_raw", fake_download_raw)
    return train, stores


def test_small_store_groups_are_dropped(fake_raw, tmp_path):
    panel = rossmann.to_panel(save=False)
    pairs = panel[["category", "region"]].drop_duplicates().apply(tuple, axis=1).tolist()
    assert ("Standard", "Basic") in pairs
    assert ("Boutique", "Basic") not in pairs


def test_low_coverage_days_are_dropped(fake_raw):
    panel = rossmann.to_panel(save=False)
    assert pd.Timestamp("2013-07-01") not in panel["date"].values


def test_panel_derived_columns(fake_raw):
    panel = rossmann.to_panel(save=False)
    assert (panel["avg_price"] == (panel["revenue"] / panel["units_sold"]).round(2)).all()
    assert (panel["cogs"] < panel["revenue"]).all()
    assert (panel["gross_profit"] == (panel["revenue"] - panel["cogs"])).all()


def test_category_and_region_mapping_uses_friendly_names(fake_raw):
    panel = rossmann.to_panel(save=False)
    assert set(panel["category"].unique()).issubset(set(rossmann.STORETYPE_NAMES.values()))
    assert set(panel["region"].unique()).issubset(set(rossmann.ASSORTMENT_NAMES.values()))
