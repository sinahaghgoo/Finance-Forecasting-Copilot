from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORY_PROFILES = {
    "Footwear":    {"base": 420_000, "cogs_rate": 0.48, "yearly_amp": 0.22, "weekend": 0.10},
    "Apparel":     {"base": 310_000, "cogs_rate": 0.52, "yearly_amp": 0.28, "weekend": 0.14},
    "Accessories": {"base": 140_000, "cogs_rate": 0.44, "yearly_amp": 0.18, "weekend": 0.08},
}

REGION_PROFILES = {
    "EMEA":     {"mult": 1.00, "trend_per_year": 0.04, "phase_shift": 0.0},
    "Americas": {"mult": 1.15, "trend_per_year": 0.05, "phase_shift": 0.08},
    "APAC":     {"mult": 0.85, "trend_per_year": 0.07, "phase_shift": -0.12},
}

HOLIDAYS = [
    ("black_friday", 11, 25, 7, 1.55),
    ("boxing_week", 12, 26, 7, 1.35),
    ("summer_sale", 7, 10, 10, 1.25),
    ("back_to_school", 8, 25, 7, 1.20),
    ("spring_launch", 3, 20, 5, 1.18),
]


def _holiday_effect(dates: pd.DatetimeIndex) -> np.ndarray:
    effect = np.ones(len(dates))
    for _, month, day, duration, strength in HOLIDAYS:
        for yr in dates.year.unique():
            start = pd.Timestamp(year=int(yr), month=month, day=day)
            end = start + pd.Timedelta(days=duration)
            mask = (dates >= start) & (dates < end)
            days_in = (dates[mask] - start).days.values
            peak = np.exp(-((days_in - duration / 2) ** 2) / (2 * (duration / 3) ** 2))
            effect[mask] *= 1 + (strength - 1) * peak
    return effect


def _covid_shock(dates: pd.DatetimeIndex) -> np.ndarray:
    effect = np.ones(len(dates))
    start = pd.Timestamp("2020-03-15")
    end = pd.Timestamp("2020-08-01")
    mask = (dates >= start) & (dates <= end)
    if mask.any():
        t = (dates[mask] - start).days.values / (end - start).days
        effect[mask] *= 1 - 0.35 * np.exp(-((t - 0.3) ** 2) / 0.05)
    return effect


def _build_series(dates: pd.DatetimeIndex, category: str, region: str, rng: np.random.Generator) -> pd.DataFrame:
    cp = CATEGORY_PROFILES[category]
    rp = REGION_PROFILES[region]
    n = len(dates)

    years_from_start = (dates - dates[0]).days.values / 365.25
    trend = cp["base"] * rp["mult"] * (1 + rp["trend_per_year"] * years_from_start)

    doy = dates.dayofyear.values
    yearly = 1 + cp["yearly_amp"] * np.sin(2 * np.pi * (doy / 365.25 + rp["phase_shift"]))

    is_weekend = (dates.dayofweek >= 5).astype(float)
    weekly = 1 + cp["weekend"] * is_weekend

    holiday = _holiday_effect(dates)
    covid = _covid_shock(dates)

    noise = rng.normal(1.0, 0.045, size=n)

    revenue = trend * yearly * weekly * holiday * covid * noise
    revenue = np.clip(revenue, a_min=1_000, a_max=None)

    price_base = {"Footwear": 92.0, "Apparel": 55.0, "Accessories": 28.0}[category]
    price_drift = 1 + 0.025 * years_from_start + rng.normal(0, 0.01, size=n)
    avg_price = price_base * price_drift
    units_sold = revenue / avg_price
    cogs = revenue * (cp["cogs_rate"] + rng.normal(0, 0.012, size=n))
    gross_profit = revenue - cogs

    return pd.DataFrame({
        "date": dates,
        "category": category,
        "region": region,
        "revenue": revenue.round(2),
        "units_sold": units_sold.round(1),
        "avg_price": avg_price.round(2),
        "cogs": cogs.round(2),
        "gross_profit": gross_profit.round(2),
    })


def generate_panel(start: str, end: str, categories: list[str], regions: list[str], seed: int = 42) -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq="D")
    rng = np.random.default_rng(seed)
    frames = []
    for cat in categories:
        for reg in regions:
            frames.append(_build_series(dates, cat, reg, rng))
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["category", "region", "date"]).reset_index(drop=True)
    return out
