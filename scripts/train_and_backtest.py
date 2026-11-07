"""Train the forecaster, run a walk-forward backtest, and save artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fin_copilot.agent.context import MODEL_PATH, _train_and_cache  # noqa: E402
from fin_copilot.config import ARTIFACTS, load_config  # noqa: E402
from fin_copilot.data.loader import load  # noqa: E402
from fin_copilot.evaluation.forecast_metrics import summary_table  # noqa: E402
from fin_copilot.forecasting.backtest import walk_forward  # noqa: E402


def main() -> None:
    cfg = load_config()
    df = load()
    print(f"Loaded {len(df):,} rows")

    print("Running walk-forward backtest...")
    result = walk_forward(df, cfg.forecast, cfg.backtest.n_folds, cfg.backtest.fold_horizon_days)
    print("Per-fold metrics:")
    print(result.per_fold.round(3).to_string(index=False))
    print("Summary:")
    print(summary_table(result))

    ARTIFACTS.mkdir(exist_ok=True)
    result.per_fold.to_csv(ARTIFACTS / "backtest_folds.csv", index=False)
    result.per_series_fold.to_csv(ARTIFACTS / "backtest_series.csv", index=False)
    (ARTIFACTS / "backtest_summary.json").write_text(
        json.dumps(summary_table(result).to_dict(), indent=2)
    )
    print(f"Saved fold results to {ARTIFACTS}")

    print("Training final model on full data and caching...")
    _train_and_cache(df)
    print(f"Cached forecaster at {MODEL_PATH}")


if __name__ == "__main__":
    main()
