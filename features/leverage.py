"""
features/leverage.py

Leverage-aware input features for GARCHNet++.

Original GARCHNet feeds only raw returns to the LSTM.
This extension uses three features per timestep:

1. r_t
2. r_t^2
3. r_t^2 * I(r_t < 0)

The third feature captures asymmetric volatility response, inspired by
GJR-GARCH leverage effects.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def build_leverage_features(returns: np.ndarray) -> np.ndarray:
    returns = np.asarray(returns)

    if returns.ndim != 1:
        raise ValueError("returns must be a 1-D array. For multiple assets, process each asset separately.")

    r = returns.astype(np.float32)

    if np.isnan(r).any() or np.isinf(r).any():
        raise ValueError("returns contains NaN or infinite values. Drop missing values before feature construction.")

    r2 = r ** 2
    leverage = np.where(r < 0, r2, 0.0).astype(np.float32)

    return np.column_stack([r, r2, leverage]).astype(np.float32)


def make_sequences(features: np.ndarray, p: int):
    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError("features must have shape (T, input_dim).")

    if p <= 0:
        raise ValueError("p must be positive.")

    T = len(features)

    if T <= p:
        raise ValueError("sequence length p must be smaller than the number of observations.")

    X = np.stack([features[t - p:t] for t in range(p, T)]).astype(np.float32)
    y = features[p:, 0].astype(np.float32)

    return X, y


class ReturnDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
