from pathlib import Path
import pandas as pd

from config import Config
from data.market_loader import load_returns
from baselines.historical import historical_simulation_forecast
from baselines.arch_models import run_arch_baselines
from baselines.neural_garchnet import neural_garchnet_forecast
from experiments.run_baselines import summarize_results


PERIODS = [
    ("Period I", "2005-01-01"),
    ("Period II", "2007-01-01"),
    ("Period III", "2013-01-01"),
    ("Period IV", "2016-01-01"),
]


def run_one_period(cfg: Config, period_name: str, start_date: str) -> pd.DataFrame:
    returns = load_returns(
        symbol=cfg.ticker,
        start=start_date,
        end=cfg.end_date,
        source=cfg.data_source,
        price_col=cfg.price_col,
        scale=cfg.return_scale,
    )

    required_obs = cfg.train_window + cfg.test_window
    if len(returns) < required_obs:
        raise ValueError(
            f"{period_name} has only {len(returns)} returns, but needs {required_obs}."
        )

    returns = returns.iloc[:required_obs].copy()

    frames = []

    frames.append(
        historical_simulation_forecast(
            returns=returns,
            train_window=cfg.train_window,
            alphas=cfg.alphas,
        )
    )

    frames.append(
        run_arch_baselines(
            returns=returns,
            train_window=cfg.train_window,
            alphas=cfg.alphas,
            dist=cfg.distribution,
            refit_every=20,
        )
    )

    frames.append(
        neural_garchnet_forecast(
            returns=returns,
            train_window=cfg.train_window,
            p=cfg.p,
            alphas=cfg.alphas,
            mode="original",
            distribution=cfg.distribution,
            lstm_hidden=cfg.lstm_hidden,
            fc_layers=cfg.fc_layers,
            dropout=0.0,
            epochs_full=cfg.epochs_full,
            epochs_warmstart=cfg.epochs_warmstart,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            patience=cfg.patience,
            grad_clip=cfg.grad_clip,
            val_frac=cfg.val_frac,
            device=cfg.device,
            refit_every=20,
        )
    )

    frames.append(
        neural_garchnet_forecast(
            returns=returns,
            train_window=cfg.train_window,
            p=cfg.p,
            alphas=cfg.alphas,
            mode="plus",
            distribution=cfg.distribution,
            lstm_hidden=cfg.lstm_hidden,
            fc_layers=cfg.fc_layers,
            dropout=cfg.dropout,
            epochs_full=cfg.epochs_full,
            epochs_warmstart=cfg.epochs_warmstart,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            patience=cfg.patience,
            grad_clip=cfg.grad_clip,
            val_frac=cfg.val_frac,
            device=cfg.device,
            refit_every=20,
        )
    )

    out = pd.concat(frames, ignore_index=True)
    out["period"] = period_name
    out["period_start"] = start_date

    return out


def save_aggregate_neural_comparison(summary: pd.DataFrame, results_dir: Path) -> None:
    neural = summary[summary["model"].isin(["Original GARCHNet", "GARCHNet++"])].copy()
    if neural.empty:
        return

    aggregate = (
        neural.groupby("model")[["exceptions", "GPL score", "LLF", "CRLF", "CFLF"]]
        .sum()
        .reset_index()
    )
    aggregate.to_csv(results_dir / "garchnet_vs_plus_aggregate.csv", index=False)


def main() -> None:
    cfg = Config()
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []

    for period_name, start_date in PERIODS:
        print(f"\nRunning {period_name}: {start_date}")
        frame = run_one_period(cfg, period_name, start_date)
        all_frames.append(frame)

    forecasts = pd.concat(all_frames, ignore_index=True)

    summaries = []
    for period, group in forecasts.groupby("period"):
        summary = summarize_results(group)
        summary.insert(0, "period", period)
        summaries.append(summary)

    summary = pd.concat(summaries, ignore_index=True)

    forecasts.to_csv(results_dir / "paper_window_forecasts.csv", index=False)
    summary.to_csv(results_dir / "paper_window_summary.csv", index=False)
    save_aggregate_neural_comparison(summary, results_dir)

    print(summary.to_string(index=False))
    print(f"\nSaved forecasts to {results_dir / 'paper_window_forecasts.csv'}")
    print(f"Saved summary to {results_dir / 'paper_window_summary.csv'}")


if __name__ == "__main__":
    main()
