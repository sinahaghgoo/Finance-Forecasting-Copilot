from __future__ import annotations

import argparse
import json
import sys

from fin_copilot.config import ARTIFACTS, load_config
from fin_copilot.data.loader import build_and_save, load
from fin_copilot.evaluation.agent_eval import aggregate, evaluate_agent
from fin_copilot.evaluation.cases import build_eval_cases
from fin_copilot.evaluation.forecast_metrics import per_series_summary, summary_table
from fin_copilot.forecasting.backtest import walk_forward
from fin_copilot.forecasting.model import QuantileForecaster


def _cmd_build_data(_args):
    df = build_and_save()
    print(f"Wrote {len(df):,} rows")


def _cmd_backtest(args):
    cfg = load_config()
    df = load()
    result = walk_forward(df, cfg.forecast, n_folds=args.folds, fold_horizon_days=args.horizon)
    print("Per-fold:")
    print(result.per_fold.round(3).to_string(index=False))
    print("\nSummary:")
    print(summary_table(result))
    print("\nPer series:")
    print(per_series_summary(result))
    ARTIFACTS.mkdir(exist_ok=True)
    result.per_fold.to_csv(ARTIFACTS / "backtest_folds.csv", index=False)
    result.per_series_fold.to_csv(ARTIFACTS / "backtest_series.csv", index=False)
    result.predictions.to_parquet(ARTIFACTS / "backtest_predictions.parquet", index=False)
    print(f"\nSaved artifacts to {ARTIFACTS}")


def _cmd_forecast(args):
    cfg = load_config()
    df = load()
    model = QuantileForecaster(quantiles=cfg.forecast.quantiles, params=cfg.forecast.lgbm).fit(df)
    pred = model.predict_recursive(df, horizon_days=args.horizon)
    out = pred if args.category is None and args.region is None else pred[
        ((pred["category"] == args.category) if args.category else True)
        & ((pred["region"] == args.region) if args.region else True)
    ]
    print(out.round(2).to_string(index=False))


def _cmd_agent(args):
    from fin_copilot.agent.graph import build_agent
    agent = build_agent()
    result = agent.invoke(args.question)
    print("Answer:\n" + result["answer"])
    print("\nTool calls:")
    for tc in result["tool_calls"]:
        print(f"  {tc['name']}({json.dumps(tc['args'])})")


def _cmd_eval_agent(args):
    from fin_copilot.agent.graph import build_agent
    df = load()
    cases = build_eval_cases(df)[: args.limit] if args.limit else build_eval_cases(df)
    agent = build_agent()
    rows = evaluate_agent(agent.invoke, cases)
    out_path = ARTIFACTS / "agent_eval.json"
    ARTIFACTS.mkdir(exist_ok=True)
    payload = {
        "aggregate": aggregate(rows),
        "rows": [
            {
                "question": r.question,
                "answer": r.answer,
                "tool_calls": r.tool_calls,
                "hallucination_rate": r.hallucination_rate,
                "accuracy_ok": r.accuracy_ok,
                "tool_coverage": r.tool_coverage,
            }
            for r in rows
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"\nFull report saved to {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fincopilot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-data").set_defaults(func=_cmd_build_data)

    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--folds", type=int, default=6)
    p_bt.add_argument("--horizon", type=int, default=28)
    p_bt.set_defaults(func=_cmd_backtest)

    p_fc = sub.add_parser("forecast")
    p_fc.add_argument("--horizon", type=int, default=28)
    p_fc.add_argument("--category", default=None)
    p_fc.add_argument("--region", default=None)
    p_fc.set_defaults(func=_cmd_forecast)

    p_ag = sub.add_parser("ask")
    p_ag.add_argument("question")
    p_ag.set_defaults(func=_cmd_agent)

    p_ev = sub.add_parser("eval-agent")
    p_ev.add_argument("--limit", type=int, default=None)
    p_ev.set_defaults(func=_cmd_eval_agent)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
