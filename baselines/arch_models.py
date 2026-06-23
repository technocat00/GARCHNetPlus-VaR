"""
baselines/arch_models.py

Classical econometric baselines:
    - GARCH(1,1)
    - EGARCH(1,1)
    - GJR-GARCH(1,1)

Requires:
    pip install arch
"""

import math
import warnings
import numpy as np
import pandas as pd
from scipy import stats


def _normal_var_es(sigma, alpha):
    q = stats.norm.ppf(alpha)
    var = sigma * q
    es = -sigma * stats.norm.pdf(q) / alpha
    return float(var), float(es)


def _student_t_var_es(sigma, alpha, nu):
    nu = max(float(nu), 2.05)
    scale = math.sqrt((nu - 2.0) / nu)
    q = stats.t.ppf(alpha, df=nu)
    var = sigma * scale * q
    pdf_q = stats.t.pdf(q, df=nu)
    es_z = -(pdf_q * (nu + q ** 2) / (nu - 1.0)) / alpha
    es = sigma * scale * es_z
    return float(var), float(es)


def arch_forecast(
    returns: pd.Series,
    model_name: str,
    train_window: int,
    alphas,
    dist: str = "t",
    refit_every: int = 20,
) -> pd.DataFrame:
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("Install arch first: pip install arch") from exc

    if not isinstance(returns, pd.Series):
        returns = pd.Series(np.asarray(returns).reshape(-1))

    model_name_clean = model_name.upper()
    values = returns.values.astype(float)
    dates = returns.index
    rows = []

    fit = None
    last_fit_t = None

    for t in range(train_window, len(values)):
        should_refit = fit is None or (t - last_fit_t) >= refit_every

        if should_refit:
            window = values[t - train_window:t]

            if model_name_clean == "GARCH":
                am = arch_model(window, mean="Zero", vol="GARCH", p=1, q=1, dist=dist)
            elif model_name_clean == "EGARCH":
                am = arch_model(window, mean="Zero", vol="EGARCH", p=1, q=1, dist=dist)
            elif model_name_clean in ("GJR-GARCH", "GJR"):
                am = arch_model(window, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist=dist)
            else:
                raise ValueError("model_name must be one of: GARCH, EGARCH, GJR-GARCH")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = am.fit(disp="off", show_warning=False)

            last_fit_t = t

        fc = fit.forecast(horizon=1, reindex=False)
        sigma2 = float(fc.variance.values[-1, 0])
        sigma = math.sqrt(max(sigma2, 1e-12))
        nu = fit.params.get("nu", None)

        for alpha in alphas:
            if dist == "t" and nu is not None:
                var, es = _student_t_var_es(sigma, alpha, nu)
            else:
                var, es = _normal_var_es(sigma, alpha)

            rows.append({
                "date": dates[t],
                "model": model_name_clean,
                "alpha": alpha,
                "return": values[t],
                "sigma2": sigma2,
                "VaR": var,
                "ES": es,
            })

    return pd.DataFrame(rows)


def run_arch_baselines(
    returns: pd.Series,
    train_window: int,
    alphas,
    dist: str = "t",
    refit_every: int = 20,
) -> pd.DataFrame:
    frames = []

    for model_name in ["GARCH", "EGARCH", "GJR-GARCH"]:
        frames.append(
            arch_forecast(
                returns=returns,
                model_name=model_name,
                train_window=train_window,
                alphas=alphas,
                dist=dist,
                refit_every=refit_every,
            )
        )

    return pd.concat(frames, ignore_index=True)
