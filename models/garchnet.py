"""
models/garchnet.py

GARCHNet++ with leverage-aware inputs.

Input:
    [r_t, r_t^2, r_t^2 * I(r_t < 0)]

Output heads:
    normal   -> sigma2
    t        -> sigma2, eta
    skewed_t -> sigma2, eta, lambda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GARCHNet(nn.Module):
    def __init__(
        self,
        p: int,
        input_dim: int = 3,
        lstm_hidden: int = 100,
        fc_dims: list = None,
        distribution: str = "t",
        dropout: float = 0.1,
        min_eta: float = 2.05,
        max_eta: float = 50.0,
    ):
        super().__init__()

        if fc_dims is None:
            fc_dims = [64, 32]

        if distribution not in ("normal", "t", "skewed_t"):
            raise ValueError("distribution must be one of: normal, t, skewed_t")

        self.p = p
        self.distribution = distribution
        self.min_eta = min_eta
        self.max_eta = max_eta

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
        )

        layers = []
        in_d = lstm_hidden

        for out_d in fc_dims:
            layers.extend([
                nn.Linear(in_d, out_d),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_d = out_d

        self.fc = nn.Sequential(*layers)
        self.last_dim = in_d

        self.sigma_head = nn.Linear(self.last_dim, 1)

        if distribution in ("t", "skewed_t"):
            self.eta_head = nn.Linear(self.last_dim, 1)

        if distribution == "skewed_t":
            self.lam_head = nn.Linear(self.last_dim, 1)

    def forward(self, x: torch.Tensor) -> dict:
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]
        shared = self.fc(h)

        out = {}

        out["sigma2"] = F.softplus(self.sigma_head(shared)) + 1e-6

        if self.distribution in ("t", "skewed_t"):
            eta_raw = torch.sigmoid(self.eta_head(shared))
            out["eta"] = self.min_eta + (self.max_eta - self.min_eta) * eta_raw

        if self.distribution == "skewed_t":
            out["lambda"] = 0.99 * torch.tanh(self.lam_head(shared))

        return out
