from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fin_copilot.forecasting.features import GROUP_COLS


@dataclass
class ScenarioParams:
    price_change_pct: float = 0.0
    cogs_change_pct: float = 0.0
    volume_change_pct: float = 0.0
    price_elasticity: float = -1.0


@dataclass
class ScenarioResult:
    baseline: pd.DataFrame
    scenario: pd.DataFrame
    delta: pd.DataFrame
    params: ScenarioParams

    def summary(self) -> dict:
        s = self.scenario.sum(numeric_only=True)
        b = self.baseline.sum(numeric_only=True)
        return {
            "baseline_revenue": float(b["revenue"]),
            "baseline_gross_profit": float(b["gross_profit"]),
            "scenario_revenue": float(s["revenue"]),
            "scenario_gross_profit": float(s["gross_profit"]),
            "delta_revenue": float(s["revenue"] - b["revenue"]),
            "delta_gross_profit": float(s["gross_profit"] - b["gross_profit"]),
            "delta_revenue_pct": float((s["revenue"] - b["revenue"]) / b["revenue"] * 100),
            "delta_gross_profit_pct": float((s["gross_profit"] - b["gross_profit"]) / b["gross_profit"] * 100),
        }


def _baseline_from_history(history: pd.DataFrame, forecast: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    recent = history.sort_values("date").groupby(GROUP_COLS, observed=True).tail(90)
    rates = recent.groupby(GROUP_COLS, observed=True).agg(
        avg_price=("avg_price", "mean"),
        cogs_rate=("cogs", lambda x: x.sum() / recent.loc[x.index, "revenue"].sum()),
    ).reset_index()

    base = forecast[GROUP_COLS + ["date", "point"]].rename(columns={"point": "revenue"})
    base = base.merge(rates, on=GROUP_COLS, how="left")
    base["units_sold"] = base["revenue"] / base["avg_price"]
    base["cogs"] = base["revenue"] * base["cogs_rate"]
    base["gross_profit"] = base["revenue"] - base["cogs"]
    return base[GROUP_COLS + ["date", "revenue", "units_sold", "avg_price", "cogs", "cogs_rate", "gross_profit"]]


def apply_scenario(baseline: pd.DataFrame, params: ScenarioParams) -> pd.DataFrame:
    price_factor = 1 + params.price_change_pct
    cogs_factor = 1 + params.cogs_change_pct
    volume_factor = max(
        (1 + params.price_elasticity * params.price_change_pct) * (1 + params.volume_change_pct),
        0.0,
    )

    scen = baseline.copy()
    scen["avg_price"] = baseline["avg_price"] * price_factor
    scen["units_sold"] = baseline["units_sold"] * volume_factor
    scen["revenue"] = baseline["revenue"] * price_factor * volume_factor
    scen["cogs"] = baseline["cogs"] * volume_factor * cogs_factor
    scen["cogs_rate"] = np.where(scen["revenue"] > 0, scen["cogs"] / scen["revenue"], np.nan)
    scen["gross_profit"] = scen["revenue"] - scen["cogs"]
    return scen


def run_scenario(history: pd.DataFrame, forecast: pd.DataFrame, params: ScenarioParams) -> ScenarioResult:
    horizon_days = forecast["date"].nunique()
    baseline = _baseline_from_history(history, forecast, horizon_days)
    scenario = apply_scenario(baseline, params)
    delta = scenario.copy()
    for col in ["revenue", "units_sold", "cogs", "gross_profit"]:
        delta[col] = scenario[col] - baseline[col]
    return ScenarioResult(baseline=baseline, scenario=scenario, delta=delta, params=params)
