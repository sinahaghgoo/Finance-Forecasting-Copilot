from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool

from fin_copilot.agent.context import current_context
from fin_copilot.scenarios.gross_profit import ScenarioParams, run_scenario

MAX_HORIZON = 90


def _validate_pair(category: str, region: str) -> tuple[str, str]:
    ctx = current_context()
    if category not in ctx.categories:
        raise ValueError(f"Unknown category '{category}'. Valid: {ctx.categories}")
    if region not in ctx.regions:
        raise ValueError(f"Unknown region '{region}'. Valid: {ctx.regions}")
    return category, region


def _slice(category: str, region: str) -> pd.DataFrame:
    ctx = current_context()
    return ctx.panel[(ctx.panel["category"] == category) & (ctx.panel["region"] == region)]


@tool
def list_dimensions() -> dict:
    """Return available product categories and regions, plus the earliest and latest date in the data."""
    ctx = current_context()
    return {
        "categories": ctx.categories,
        "regions": ctx.regions,
        "data_start": str(ctx.panel["date"].min().date()),
        "data_end": str(ctx.panel["date"].max().date()),
    }


@tool
def historical_metric(
    category: str,
    region: str,
    start_date: str,
    end_date: str,
    metric: str = "revenue",
) -> dict:
    """Return the total and daily average of a metric for a (category, region) between two ISO dates.

    Valid metrics: revenue, units_sold, cogs, gross_profit, avg_price.
    """
    _validate_pair(category, region)
    allowed = {"revenue", "units_sold", "cogs", "gross_profit", "avg_price"}
    if metric not in allowed:
        raise ValueError(f"Unknown metric '{metric}'. Valid: {sorted(allowed)}")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    sub = _slice(category, region)
    sub = sub[(sub["date"] >= start) & (sub["date"] <= end)]
    if sub.empty:
        raise ValueError(f"No data between {start_date} and {end_date}.")

    return {
        "category": category,
        "region": region,
        "metric": metric,
        "start_date": start_date,
        "end_date": end_date,
        "n_days": int(len(sub)),
        "total": float(sub[metric].sum()),
        "mean_daily": float(sub[metric].mean()),
        "min_daily": float(sub[metric].min()),
        "max_daily": float(sub[metric].max()),
    }


@tool
def forecast_revenue(category: str, region: str, horizon_days: int = 28) -> dict:
    """Forecast daily revenue for a (category, region) over the next horizon_days.

    Returns total forecast revenue plus a median point estimate and 80% prediction
    interval (q10, q90) at daily granularity.
    """
    _validate_pair(category, region)
    if not (1 <= horizon_days <= MAX_HORIZON):
        raise ValueError(f"horizon_days must be between 1 and {MAX_HORIZON}, got {horizon_days}")

    ctx = current_context()
    history = ctx.panel
    pred = ctx.model.predict_recursive(history, horizon_days=horizon_days)
    sub = pred[(pred["category"] == category) & (pred["region"] == region)].sort_values("date")

    daily = [
        {
            "date": str(row["date"].date()),
            "point": float(row["point"]),
            "q10": float(row["q10"]),
            "q90": float(row["q90"]),
        }
        for _, row in sub.iterrows()
    ]
    return {
        "category": category,
        "region": region,
        "horizon_days": horizon_days,
        "total_point": float(sub["point"].sum()),
        "total_q10": float(sub["q10"].sum()),
        "total_q90": float(sub["q90"].sum()),
        "mean_daily_point": float(sub["point"].mean()),
        "daily": daily[:7],
    }


@tool
def gross_profit_scenario(
    category: str,
    region: str,
    horizon_days: int = 28,
    price_change_pct: float = 0.0,
    cogs_change_pct: float = 0.0,
    price_elasticity: float = -1.0,
    volume_change_pct: float = 0.0,
) -> dict:
    """Run a what-if scenario on gross profit for a (category, region) over horizon_days.

    Inputs are decimal fractions (e.g. 0.05 for +5%). price_elasticity is typically
    negative; -1.0 means a 1% price increase reduces units sold by 1%.
    """
    _validate_pair(category, region)
    if not (1 <= horizon_days <= MAX_HORIZON):
        raise ValueError(f"horizon_days must be between 1 and {MAX_HORIZON}")

    ctx = current_context()
    history = ctx.panel
    pred = ctx.model.predict_recursive(history, horizon_days=horizon_days)
    pred = pred[(pred["category"] == category) & (pred["region"] == region)]
    hist_pair = _slice(category, region)

    params = ScenarioParams(
        price_change_pct=price_change_pct,
        cogs_change_pct=cogs_change_pct,
        volume_change_pct=volume_change_pct,
        price_elasticity=price_elasticity,
    )
    result = run_scenario(hist_pair, pred, params)
    s = result.summary()
    return {
        "category": category,
        "region": region,
        "horizon_days": horizon_days,
        "params": {
            "price_change_pct": price_change_pct,
            "cogs_change_pct": cogs_change_pct,
            "price_elasticity": price_elasticity,
            "volume_change_pct": volume_change_pct,
        },
        **{k: round(v, 2) for k, v in s.items()},
    }


ALL_TOOLS = [list_dimensions, historical_metric, forecast_revenue, gross_profit_scenario]
