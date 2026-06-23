"""
baselines/neural_garchnet.py

Neural baselines:
    - Original GARCHNet from the paper: input_dim=1, return-only features
    - GARCHNet++ proposed extension: input_dim=3, leverage-aware features
"""

import copy
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from models.garchnet import GARCHNet
from training.losses import get_loss_fn
from features.plain import build_plain_features
from features.leverage import build_leverage_features, make_sequences, ReturnDataset
from evaluation.backtester import Backtester


def _device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _train_one_window(
    model,
    X,
    y,
    distribution,
    epochs,
    lr,
    batch_size,
    patience,
    grad_clip,
    val_frac,
    device,
):
    dataset = ReturnDataset(X, y)

    val_size = max(1, int(len(dataset) * val_frac))
    train_size = len(dataset) - val_size

    if train_size <= 0:
        raise ValueError("Training window too small for validation split.")

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(1),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    loss_fn = get_loss_fn(distribution)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0

    model.to(device)

    for _ in range(epochs):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(epsilon=yb, **out)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                out = model(xb)
                val_loss = loss_fn(epsilon=yb, **out)
                val_losses.append(float(val_loss.detach().cpu()))

        val_mean = float(np.mean(val_losses))

        if val_mean < best_val:
            best_val = val_mean
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return model


def neural_garchnet_forecast(
    returns: pd.Series,
    train_window: int,
    p: int,
    alphas,
    mode: str,
    distribution: str = "t",
    lstm_hidden: int = 100,
    fc_layers=None,
    dropout: float = 0.1,
    epochs_full: int = 300,
    epochs_warmstart: int = 50,
    lr: float = 3e-4,
    batch_size: int = 512,
    patience: int = 10,
    grad_clip: float = 1.0,
    val_frac: float = 0.1,
    device: str = "auto",
    refit_every: int = 20,
) -> pd.DataFrame:
    if fc_layers is None:
        fc_layers = [64, 32]

    if not isinstance(returns, pd.Series):
        returns = pd.Series(np.asarray(returns).reshape(-1))

    if mode not in ("original", "plus"):
        raise ValueError("mode must be either 'original' or 'plus'.")

    dev = _device(device)
    values = returns.values.astype(float)
    dates = returns.index

    input_dim = 1 if mode == "original" else 3
    model_label = "Original GARCHNet" if mode == "original" else "GARCHNet++"

    model = None
    rows = []
    bt = {alpha: Backtester(alpha=alpha) for alpha in alphas}

    for t in range(train_window, len(values)):
        should_refit = model is None or ((t - train_window) % refit_every == 0)

        if should_refit:
            train_returns = values[t - train_window:t]

            if mode == "original":
                feats = build_plain_features(train_returns)
            else:
                feats = build_leverage_features(train_returns)

            X, y = make_sequences(feats, p)

            if model is None:
                model = GARCHNet(
                    p=p,
                    input_dim=input_dim,
                    lstm_hidden=lstm_hidden,
                    fc_dims=fc_layers,
                    distribution=distribution,
                    dropout=dropout,
                )
                epochs = epochs_full
            else:
                epochs = epochs_warmstart

            model = _train_one_window(
                model=model,
                X=X,
                y=y,
                distribution=distribution,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                patience=patience,
                grad_clip=grad_clip,
                val_frac=val_frac,
                device=dev,
            )

        hist = values[t - p:t]

        if mode == "original":
            pred_feats = build_plain_features(hist)
        else:
            pred_feats = build_leverage_features(hist)

        x_pred = torch.from_numpy(pred_feats.reshape(1, p, input_dim).astype(np.float32)).to(dev)

        model.eval()
        with torch.no_grad():
            out = model(x_pred)

        sigma2 = float(out["sigma2"].detach().cpu().reshape(-1)[0])
        eta = None
        lam = None

        if "eta" in out:
            eta = np.array([float(out["eta"].detach().cpu().reshape(-1)[0])])

        if "lambda" in out:
            lam = np.array([float(out["lambda"].detach().cpu().reshape(-1)[0])])

        for alpha in alphas:
            var, es = bt[alpha].var_and_es(
                sigma2=np.array([sigma2]),
                distribution=distribution,
                eta=eta,
                lam=lam,
            )

            rows.append({
                "date": dates[t],
                "model": model_label,
                "alpha": alpha,
                "return": values[t],
                "sigma2": sigma2,
                "VaR": float(var[0]),
                "ES": float(es[0]),
            })

    return pd.DataFrame(rows)
