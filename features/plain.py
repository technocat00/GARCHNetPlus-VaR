"""
features/plain.py

Plain return-only features for reproducing the paper's Original GARCHNet baseline.
"""

import numpy as np


def build_plain_features(returns: np.ndarray) -> np.ndarray:
    returns = np.asarray(returns)

    if returns.ndim != 1:
        raise ValueError("returns must be a 1-D array.")

    r = returns.astype(np.float32)

    if np.isnan(r).any() or np.isinf(r).any():
        raise ValueError("returns contains NaN or infinite values.")

    return r.reshape(-1, 1)
