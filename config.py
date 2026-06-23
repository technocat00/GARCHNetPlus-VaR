"""
config.py

Single source of truth for the GARCHNet++ experiment.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    ticker: str = "SPY"
    data_source: str = "yfinance"
    start_date: str = "2005-01-01"
    end_date: str = "2024-12-31"
    price_col: str = "Close"
    return_scale: float = 100.0

    p: int = 20
    input_dim: int = 3
    lstm_hidden: int = 100
    fc_layers: List[int] = field(default_factory=lambda: [64, 32])
    distribution: str = "t"
    dropout: float = 0.1

    train_window: int = 1000
    test_window: int = 252
    epochs_full: int = 20
    epochs_warmstart: int = 5
    lr: float = 3e-4
    batch_size: int = 512
    patience: int = 5
    grad_clip: float = 1.0
    val_frac: float = 0.1
    seed: int = 1
    device: str = "auto"

    alphas: List[float] = field(default_factory=lambda: [0.025])
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
