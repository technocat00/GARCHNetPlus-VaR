"""
training/losses.py

Negative log-likelihood losses for GARCHNet++.

Supported distributions:
    normal
    t
    skewed_t
"""

import math
import torch


EPS = 1e-8


def _flat(x):
    return x.reshape(-1)


def nll_normal(epsilon: torch.Tensor, sigma2: torch.Tensor, **_) -> torch.Tensor:
    epsilon = _flat(epsilon)
    sigma2 = _flat(sigma2).clamp_min(EPS)

    return 0.5 * (
        math.log(2.0 * math.pi)
        + sigma2.log()
        + epsilon.pow(2) / sigma2
    ).mean()


def nll_t(
    epsilon: torch.Tensor,
    sigma2: torch.Tensor,
    eta: torch.Tensor,
    **_,
) -> torch.Tensor:
    epsilon = _flat(epsilon)
    sigma2 = _flat(sigma2).clamp_min(EPS)
    eta = _flat(eta).clamp_min(2.05)

    term1 = torch.lgamma((eta + 1.0) / 2.0)
    term2 = torch.lgamma(eta / 2.0)
    term3 = 0.5 * torch.log(math.pi * (eta - 2.0) * sigma2)
    term4 = ((eta + 1.0) / 2.0) * torch.log1p(
        epsilon.pow(2) / (sigma2 * (eta - 2.0))
    )

    return (term2 + term3 + term4 - term1).mean()


def nll_skewed_t(
    epsilon: torch.Tensor,
    sigma2: torch.Tensor,
    eta: torch.Tensor,
    lam: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    if lam is None:
        lam = kwargs.get("lambda")

    if lam is None:
        raise ValueError("skewed_t loss requires `lam` or `lambda` parameter.")

    epsilon = _flat(epsilon)
    sigma2 = _flat(sigma2).clamp_min(EPS)
    eta = _flat(eta).clamp_min(2.05)
    lam = _flat(lam).clamp(-0.995, 0.995)

    sigma = sigma2.sqrt()

    c = torch.exp(
        torch.lgamma((eta + 1.0) / 2.0)
        - 0.5 * torch.log(math.pi * (eta - 2.0))
        - torch.lgamma(eta / 2.0)
    )

    a = 4.0 * lam * c * (eta - 2.0) / (eta - 1.0)
    b = torch.sqrt((1.0 + 3.0 * lam.pow(2) - a.pow(2)).clamp_min(EPS))

    z = epsilon / sigma
    threshold = -a / b

    inner_left = (a + b * z) / (1.0 - lam)
    inner_right = (a + b * z) / (1.0 + lam)

    inner = torch.where(z < threshold, inner_left, inner_right)

    log_lik = (
        torch.log(b)
        + torch.log(c)
        - torch.log(sigma)
        - ((eta + 1.0) / 2.0) * torch.log1p(inner.pow(2) / (eta - 2.0))
    )

    return -log_lik.mean()


_REGISTRY = {
    "normal": nll_normal,
    "t": nll_t,
    "skewed_t": nll_skewed_t,
}


def get_loss_fn(distribution: str):
    if distribution not in _REGISTRY:
        raise ValueError(f"Unknown distribution '{distribution}'. Choose from {list(_REGISTRY)}")
    return _REGISTRY[distribution]
