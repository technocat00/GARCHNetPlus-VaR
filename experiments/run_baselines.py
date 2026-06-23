"""
experiments/run_baselines.py

End-to-end baseline runner.

Example:
    python -m experiments.run_baselines
"""

from pathlib import Path
import pandas as pd

from config import Config
from data.market_loader import load_returns
from baselines.historical import historical_simulation_forecast
from baselines.arch_models import run_arch_baselines
from baselines.neural_garchnet import neural_garchnet_forecast
from evaluation.backtester import Backtester


def summarize_results(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model, alpha), group in forecasts.groupby(["model", "alpha"]):
        bt = Backtester(alpha=alpha)
        var_stats = bt.test_var(group["return"].values, group["VaR"].values)
        es_stats = bt.test_es(group["return"].values, group["VaR"].values, group["ES"].values)
        loss_stats = bt.loss_functions(group["return"].values, group["VaR"].values)

        rows.append({
            "model": model,
            "alpha": alpha,
            "exceptions": var_stats["H (exceptions)"],
            "alpha_hat": var_stats["alpha_hat"],
            "UC p-value": var_stats["UC p-value"],
            "CC p-value": var_stats["CC p-value"],
            "DQ p-value": var_stats["DQ p-value"],
            "GPL score": var_stats["GPL score"],
            "ES p-value": es_stats["p_value"],
            "ES residual mean": es_stats["residual_mean"],
            "LLF": loss_stats["LLF (Lopez quadratic)"],
            "CRLF": loss_stats["CRLF (Caporin regulator)"],
            "CFLF": loss_stats["CFLF (Caporin firm)"],
        })

    return pd.DataFrame(rows).sort_values(["alpha", "GPL score"])


def main():
    cfg = Config()
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    returns = load_returns(
        symbol=cfg.ticker,
        start=cfg.start_date,
        end=cfg.end_date,
        source=cfg.data_source,
        price_col=cfg.price_col,
        scale=cfg.return_scale,
    )

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

    forecasts = pd.concat(frames, ignore_index=True)
    summary = summarize_results(forecasts)

    forecasts.to_csv(Path(cfg.results_dir) / "forecast_panel.csv", index=False)
    summary.to_csv(Path(cfg.results_dir) / "baseline_summary.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
