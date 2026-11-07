from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from fin_copilot.config import LGBMParams
from fin_copilot.forecasting.features import (
    GROUP_COLS,
    LAGS,
    ROLLING_WINDOWS,
    TARGET,
    build_features,
    feature_columns,
)


@dataclass
class QuantileForecaster:
    quantiles: list[float]
    params: LGBMParams
    models: dict[float, lgb.Booster] | None = None
    feature_cols: list[str] | None = None

    def _lgb_params(self, quantile: float) -> dict:
        return {
            "objective": "quantile",
            "alpha": quantile,
            "metric": "quantile",
            "num_leaves": self.params.num_leaves,
            "learning_rate": self.params.learning_rate,
            "min_data_in_leaf": self.params.min_data_in_leaf,
            "feature_fraction": self.params.feature_fraction,
            "bagging_fraction": self.params.bagging_fraction,
            "bagging_freq": self.params.bagging_freq,
            "verbose": self.params.verbose,
        }

    def fit(self, train: pd.DataFrame) -> QuantileForecaster:
        feats = build_features(train)
        feats = feats.dropna(subset=[f"lag_{max(LAGS)}"])
        cols = feature_columns(feats)
        y = feats[TARGET].values
        X = feats[cols]
        cat_cols = [c for c in GROUP_COLS if c in cols]
        dtrain = lgb.Dataset(X, label=y, categorical_feature=cat_cols, free_raw_data=False)
        self.models = {}
        for q in self.quantiles:
            booster = lgb.train(
                self._lgb_params(q),
                dtrain,
                num_boost_round=self.params.n_estimators,
            )
            self.models[q] = booster
        self.feature_cols = cols
        return self

    def _predict_row(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        return {q: self.models[q].predict(X[self.feature_cols]) for q in self.quantiles}

    def predict_recursive(self, history: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
        assert self.models is not None, "Call fit() first."
        hist = history.copy()
        rows = []
        last_date = hist["date"].max()
        groups = hist[GROUP_COLS].drop_duplicates().to_dict(orient="records")

        for step in range(1, horizon_days + 1):
            step_date = last_date + pd.Timedelta(days=step)
            new_rows = pd.DataFrame(
                [{**g, "date": step_date, TARGET: np.nan} for g in groups]
            )
            frame = pd.concat([hist, new_rows], ignore_index=True)
            feats = build_features(frame)
            step_mask = feats["date"] == step_date
            preds = self._predict_row(feats.loc[step_mask])
            sorted_qs = sorted(self.quantiles)
            stacked = np.sort(np.stack([preds[q] for q in sorted_qs], axis=1), axis=1)
            preds = {q: stacked[:, i] for i, q in enumerate(sorted_qs)}
            median_q = 0.5 if 0.5 in self.quantiles else sorted_qs[len(sorted_qs) // 2]
            point = preds[median_q]

            step_frame = feats.loc[step_mask, GROUP_COLS + ["date"]].copy()
            for q, vals in preds.items():
                step_frame[f"q{int(q*100):02d}"] = vals
            step_frame["point"] = point
            rows.append(step_frame)

            fill = feats.loc[step_mask, GROUP_COLS + ["date"]].copy()
            fill[TARGET] = point
            hist = pd.concat([hist, fill], ignore_index=True)

        return pd.concat(rows, ignore_index=True).sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)


def minimum_history_days(lags: list[int] = LAGS, windows: list[int] = ROLLING_WINDOWS) -> int:
    return max(max(lags), max(windows) + 1) + 1
