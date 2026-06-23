from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _aggregate_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    comp = summary[summary["model"].isin(["Original GARCHNet", "GARCHNet++"])].copy()
    aggregate = (
        comp.groupby("model")[["exceptions", "GPL score", "LLF", "CRLF", "CFLF"]]
        .sum()
        .reset_index()
    )
    return aggregate


def plot_exception_comparison(summary: pd.DataFrame) -> None:
    comp = summary[summary["model"].isin(["Original GARCHNet", "GARCHNet++"])].copy()
    pivot = comp.pivot(index="period", columns="model", values="exceptions")
    period_order = ["Period I", "Period II", "Period III", "Period IV"]
    pivot = pivot.loc[[p for p in period_order if p in pivot.index]]

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("VaR Exceptions by Period")
    ax.set_xlabel("Out-of-sample period")
    ax.set_ylabel("Number of VaR exceptions")
    ax.axhline(252 * 0.025, linestyle="--", linewidth=1)
    ax.text(len(pivot) - 0.95, 252 * 0.025 + 0.4, "Expected ≈ 6.3", fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "var_exceptions_comparison.png", dpi=180)
    plt.close()


def plot_original_vs_plus(aggregate: pd.DataFrame) -> None:
    orig = aggregate[aggregate["model"] == "Original GARCHNet"].iloc[0]
    plus = aggregate[aggregate["model"] == "GARCHNet++"].iloc[0]

    metrics = ["exceptions", "GPL score", "LLF", "CRLF", "CFLF"]
    normalized = pd.DataFrame({
        "Metric": ["VaR breaches" if m == "exceptions" else m for m in metrics],
        "Original GARCHNet": [100 for _ in metrics],
        "GARCHNet++": [plus[m] / orig[m] * 100 for m in metrics],
    })

    ax = normalized.set_index("Metric").plot(kind="bar", figsize=(9, 5))
    ax.set_title("Aggregate Metrics: Original GARCHNet vs GARCHNet++")
    ax.set_ylabel("Metric value normalized to Original GARCHNet = 100")
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "original_vs_plus_normalized.png", dpi=180)
    plt.close()

    reduction = pd.DataFrame({
        "Metric": ["VaR breaches", "GPL score", "LLF", "CRLF", "CFLF"],
        "Reduction (%)": [
            (orig["exceptions"] - plus["exceptions"]) / orig["exceptions"] * 100,
            (orig["GPL score"] - plus["GPL score"]) / orig["GPL score"] * 100,
            (orig["LLF"] - plus["LLF"]) / orig["LLF"] * 100,
            (orig["CRLF"] - plus["CRLF"]) / orig["CRLF"] * 100,
            (orig["CFLF"] - plus["CFLF"]) / orig["CFLF"] * 100,
        ],
    })

    ax = reduction.plot(kind="bar", x="Metric", y="Reduction (%)", legend=False, figsize=(8, 4.5))
    ax.set_title("Percentage Reduction of GARCHNet++ Relative to Original GARCHNet")
    ax.set_ylabel("Reduction (%)")
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "aggregate_reduction.png", dpi=180)
    plt.close()


def plot_forecast_paths() -> None:
    forecast_path = RESULTS_DIR / "paper_window_forecasts.csv"
    if not forecast_path.exists():
        print("Skipping forecast path plot: results/paper_window_forecasts.csv not found.")
        return

    df = pd.read_csv(forecast_path)

    required = {"period", "model", "return", "VaR"}
    missing = required - set(df.columns)
    if missing:
        print(f"Skipping forecast path plot. Missing columns: {sorted(missing)}")
        return

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        x_col = "date"
    else:
        df["step"] = df.groupby(["period", "model"]).cumcount()
        x_col = "step"

    periods = ["Period I", "Period II", "Period III", "Period IV"]
    models = ["Original GARCHNet", "GARCHNet++"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    axes = axes.ravel()

    for ax, period in zip(axes, periods):
        part = df[df["period"] == period].copy()

        if part.empty:
            ax.set_title(period)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        base = part[part["model"] == "Original GARCHNet"]
        if base.empty:
            base = part.drop_duplicates(subset=[x_col])

        ax.plot(base[x_col], base["return"], label="Actual return", linewidth=1)

        for model in models:
            mdf = part[part["model"] == model]
            if not mdf.empty:
                ax.plot(mdf[x_col], mdf["VaR"], label=model, linewidth=1.2)

        ax.set_title(period)
        ax.axhline(0, linewidth=0.8)
        ax.tick_params(axis="x", rotation=25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("One-Day-Ahead 2.5% VaR Forecasts", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(ASSETS_DIR / "var_forecast_paths.png", dpi=180)
    plt.close()


def main() -> None:
    summary_path = RESULTS_DIR / "paper_window_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Missing results/paper_window_summary.csv. Run python -m experiments.run_paper_windows first.")

    summary = pd.read_csv(summary_path)

    aggregate_path = RESULTS_DIR / "garchnet_vs_plus_aggregate.csv"
    if aggregate_path.exists():
        aggregate = pd.read_csv(aggregate_path)
    else:
        aggregate = _aggregate_from_summary(summary)
        aggregate.to_csv(aggregate_path, index=False)

    plot_exception_comparison(summary)
    plot_original_vs_plus(aggregate)
    plot_forecast_paths()

    print("Saved plots to assets/")


if __name__ == "__main__":
    main()
