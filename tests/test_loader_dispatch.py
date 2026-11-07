from __future__ import annotations

import pandas as pd
import pytest

from fin_copilot.data import loader


def test_default_source_is_synthetic(monkeypatch):
    monkeypatch.delenv("FINCOPILOT_DATA_SOURCE", raising=False)
    assert loader._source() == "synthetic"


def test_env_var_selects_source(monkeypatch):
    monkeypatch.setenv("FINCOPILOT_DATA_SOURCE", "ROSSMANN")
    assert loader._source() == "rossmann"


def test_unknown_source_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("FINCOPILOT_DATA_SOURCE", "quarterly_reports")
    with pytest.raises(ValueError, match="Unknown FINCOPILOT_DATA_SOURCE"):
        loader.load(path=tmp_path / "nothing.parquet")


def test_rossmann_source_dispatches(monkeypatch):
    called = {}

    def fake_rossmann_load():
        called["ok"] = True
        return pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "category": ["x"],
                             "region": ["y"], "revenue": [1.0], "units_sold": [1.0],
                             "avg_price": [1.0], "cogs": [0.5], "gross_profit": [0.5]})

    monkeypatch.setenv("FINCOPILOT_DATA_SOURCE", "rossmann")
    monkeypatch.setattr("fin_copilot.data.rossmann.load", fake_rossmann_load)
    df = loader.load()
    assert called.get("ok") is True
    assert len(df) == 1


def test_online_retail_source_dispatches(monkeypatch):
    called = {}

    def fake_or_load():
        called["ok"] = True
        return pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "category": ["x"],
                             "region": ["y"], "revenue": [1.0], "units_sold": [1.0],
                             "avg_price": [1.0], "cogs": [0.5], "gross_profit": [0.5]})

    monkeypatch.setenv("FINCOPILOT_DATA_SOURCE", "online_retail")
    monkeypatch.setattr("fin_copilot.data.online_retail.load", fake_or_load)
    df = loader.load()
    assert called.get("ok") is True
    assert len(df) == 1
