"""
baselines/historical.py

Historical Simulation VaR/ES baseline.
"""

import numpy as np
import pandas as pd


def historical_simulation_forecast(
    returns: pd.Series,
    train_window: int,
    alphas,
) -> pd.DataFrame:
    if not isinstance(returns, pd.Series):
        returns = pd.Series(np.asarray(returns).reshape(-1))

    rows = []
    values = returns.values.astype(float)
    dates = returns.index

    for t in range(train_window, len(values)):
        hist = values[t - train_window:t]
        actual = values[t]

        sigma2 = float(np.var(hist, ddof=1))

        for alpha in alphas:
            var = float(np.quantile(hist, alpha))
            tail = hist[hist <= var]
            es = float(tail.mean()) if len(tail) else var

            rows.append({
                "date": dates[t],
                "model": "Historical Simulation",
                "alpha": alpha,
                "return": actual,
                "sigma2": sigma2,
                "VaR": var,
                "ES": es,
            })

    return pd.DataFrame(rows)
