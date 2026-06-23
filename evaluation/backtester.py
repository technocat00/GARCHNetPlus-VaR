"""
evaluation/backtester.py

VaR and Expected Shortfall forecasting/backtesting for GARCHNet++.

Includes:
    - Parametric VaR and ES for normal and Student-t
    - Monte Carlo VaR and ES for Hansen skewed-t
    - Kupiec UC test
    - Christoffersen CC test
    - Dynamic Quantile test
    - Practical ES exceedance residual diagnostic
"""

import math
import numpy as np
from scipy import stats


class Backtester:
    def __init__(self, alpha: float = 0.025, seed: int = 1):
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def var_and_es(
        self,
        sigma2: np.ndarray,
        distribution: str,
        eta: np.ndarray = None,
        lam: np.ndarray = None,
        mu: float = 0.0,
    ):
        sigma2 = np.asarray(sigma2, dtype=float).reshape(-1)

        if eta is not None:
            eta = np.asarray(eta, dtype=float).reshape(-1)

        if lam is not None:
            lam = np.asarray(lam, dtype=float).reshape(-1)

        sigma = np.sqrt(np.maximum(sigma2, 1e-12))
        N = len(sigma)
        var = np.zeros(N)
        es = np.zeros(N)
        a = self.alpha

        if distribution == "normal":
            q = stats.norm.ppf(a)
            var = mu + sigma * q
            es = mu - sigma * stats.norm.pdf(q) / a

        elif distribution == "t":
            for i in range(N):
                n = float(eta[i]) if eta is not None else 5.0
                n = max(n, 2.05)
                scale = math.sqrt((n - 2.0) / n)
                q_t = stats.t.ppf(a, df=n)
                var[i] = mu + sigma[i] * scale * q_t
                f_q = stats.t.pdf(q_t, df=n)
                es_z = -(f_q * (n + q_t ** 2) / (n - 1.0)) / a
                es[i] = mu + sigma[i] * scale * es_z

        elif distribution == "skewed_t":
            for i in range(N):
                n = float(eta[i]) if eta is not None else 5.0
                l = float(lam[i]) if lam is not None else 0.0
                samples = self._sample_hansen(n, l, n_samples=20_000)
                q_mc = np.quantile(samples, a)
                var[i] = mu + sigma[i] * q_mc
                tail = samples[samples < q_mc]
                es[i] = mu + sigma[i] * (tail.mean() if len(tail) > 0 else q_mc)

        else:
            raise ValueError("distribution must be one of: normal, t, skewed_t")

        return var, es

    def test_var(self, returns: np.ndarray, var: np.ndarray) -> dict:
        returns = np.asarray(returns, dtype=float).reshape(-1)
        var = np.asarray(var, dtype=float).reshape(-1)

        if len(returns) != len(var):
            raise ValueError("returns and var must have the same length.")

        hits = (returns < var).astype(int)
        n = int(hits.sum())
        N = len(hits)

        return {
            "H (exceptions)": n,
            "alpha_hat": round(n / N, 4),
            "UC p-value": round(self._uc(N, n), 4),
            "CC p-value": round(self._cc(hits, N, n), 4),
            "DQ p-value": round(self._dq(hits, returns, var), 4),
            "GPL score": round(self._gpl(returns, var), 4),
        }

    def test_es(
        self,
        returns: np.ndarray,
        var: np.ndarray,
        es: np.ndarray,
    ) -> dict:
        returns = np.asarray(returns, dtype=float).reshape(-1)
        var = np.asarray(var, dtype=float).reshape(-1)
        es = np.asarray(es, dtype=float).reshape(-1)

        if not (len(returns) == len(var) == len(es)):
            raise ValueError("returns, var, and es must have the same length.")

        exc_mask = returns < var
        n_exc = int(exc_mask.sum())

        if n_exc == 0:
            return {
                "n_exceptions": 0,
                "mean_return_exc": None,
                "mean_es_exc": None,
                "residual_mean": None,
                "residual_std": None,
                "t_statistic": None,
                "p_value": None,
                "es_adequate": True,
                "verdict": "No VaR exceptions; ES not testable",
            }

        r_exc = returns[exc_mask]
        es_exc = es[exc_mask]
        residuals = r_exc / (es_exc + 1e-12)

        if n_exc < 2:
            return {
                "n_exceptions": n_exc,
                "mean_return_exc": round(float(r_exc.mean()), 6),
                "mean_es_exc": round(float(es_exc.mean()), 6),
                "residual_mean": round(float(residuals.mean()), 4),
                "residual_std": None,
                "t_statistic": None,
                "p_value": None,
                "es_adequate": None,
                "verdict": "Too few VaR exceptions for ES t-test",
            }

        t_stat, p_val = stats.ttest_1samp(residuals, popmean=1.0)
        adequate = bool(p_val > 0.05)

        return {
            "n_exceptions": n_exc,
            "mean_return_exc": round(float(r_exc.mean()), 6),
            "mean_es_exc": round(float(es_exc.mean()), 6),
            "residual_mean": round(float(residuals.mean()), 4),
            "residual_std": round(float(residuals.std(ddof=1)), 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 4),
            "es_adequate": adequate,
            "verdict": "ES adequate; H0 not rejected" if adequate else "ES rejected; re-examine model",
        }

    def loss_functions(self, returns: np.ndarray, var: np.ndarray) -> dict:
        returns = np.asarray(returns, dtype=float).reshape(-1)
        var = np.asarray(var, dtype=float).reshape(-1)

        hits = returns < var
        llf = float(np.where(hits, 1.0 + (var - returns) ** 2, 0.0).sum())
        crlf = float(np.where(hits, np.abs(1.0 - returns / (var + 1e-12)), 0.0).sum())
        cflf = float(np.abs(1.0 - returns / (var + 1e-12)).sum())

        return {
            "LLF (Lopez quadratic)": round(llf, 4),
            "CRLF (Caporin regulator)": round(crlf, 4),
            "CFLF (Caporin firm)": round(cflf, 4),
            "GPL (Gneiting 2011)": round(self._gpl(returns, var), 4),
        }

    def _uc_lr(self, N: int, n: int) -> float:
        a = self.alpha
        ah = n / N

        def slog(x):
            return math.log(max(x, 1e-300))

        log_l0 = (N - n) * slog(1.0 - a) + n * slog(a)

        if ah <= 0.0:
            log_l1 = (N - n) * slog(1.0)
        elif ah >= 1.0:
            log_l1 = n * slog(1.0)
        else:
            log_l1 = (N - n) * slog(1.0 - ah) + n * slog(ah)

        return -2.0 * (log_l0 - log_l1)

    def _uc(self, N: int, n: int) -> float:
        lr = self._uc_lr(N, n)
        return float(1.0 - stats.chi2.cdf(lr, df=1))

    def _cc(self, hits: np.ndarray, N: int, n: int) -> float:
        n00 = int(((hits[:-1] == 0) & (hits[1:] == 0)).sum())
        n01 = int(((hits[:-1] == 0) & (hits[1:] == 1)).sum())
        n10 = int(((hits[:-1] == 1) & (hits[1:] == 0)).sum())
        n11 = int(((hits[:-1] == 1) & (hits[1:] == 1)).sum())

        def slog(x):
            return math.log(max(x, 1e-300))

        pi = n / N
        pi01 = n01 / max(n00 + n01, 1)
        pi11 = n11 / max(n10 + n11, 1)

        log_l_ind = (
            (n00 + n10) * slog(1.0 - pi)
            + (n01 + n11) * slog(pi)
        )

        log_l_dep = (
            n00 * slog(1.0 - pi01)
            + n01 * slog(pi01)
            + n10 * slog(1.0 - pi11)
            + n11 * slog(pi11)
        )

        lr_ind = -2.0 * (log_l_ind - log_l_dep)
        lr_cc = self._uc_lr(N, n) + lr_ind

        return float(1.0 - stats.chi2.cdf(lr_cc, df=2))

    def _dq(self, hits, returns, var, lags: int = 4) -> float:
        a = self.alpha
        H = hits.astype(float) - a
        T = len(H)

        if T <= lags + 1:
            return float("nan")

        cols = [np.ones(T)]

        for k in range(1, lags + 1):
            cols.append(np.concatenate([np.zeros(k), H[:-k]]))

        cols.append(var)
        X = np.column_stack(cols)

        try:
            dq = (H @ X @ np.linalg.pinv(X.T @ X) @ X.T @ H) / (a * (1.0 - a))
            return float(1.0 - stats.chi2.cdf(dq, df=lags + 2))
        except Exception:
            return float("nan")

    def _gpl(self, returns: np.ndarray, var: np.ndarray) -> float:
        a = self.alpha
        scores = np.where(
            returns <= var,
            (1.0 - a) * (var - returns),
            a * (returns - var),
        )
        return float(scores.mean())

    def _sample_hansen(self, eta: float, lam: float, n_samples: int) -> np.ndarray:
        eta = max(float(eta), 2.05)
        lam = float(np.clip(lam, -0.995, 0.995))

        c = (
            math.exp(math.lgamma((eta + 1.0) / 2.0) - math.lgamma(eta / 2.0))
            / math.sqrt(math.pi * (eta - 2.0))
        )
        a = 4.0 * lam * c * (eta - 2.0) / (eta - 1.0)
        b = math.sqrt(max(1.0 + 3.0 * lam ** 2 - a ** 2, 1e-10))

        w = self.rng.standard_t(df=eta, size=n_samples)
        y = math.sqrt((eta - 2.0) / eta) * w

        z = np.where(
            y < 0,
            ((1.0 - lam) * y - a) / b,
            ((1.0 + lam) * y - a) / b,
        )

        return z
