"""Random-matrix effective rank: where does the sample-noise bulk end?

A sample covariance built from ``N`` observations of a ``D``-dimensional input spreads its
eigenvalues over the Marchenko-Pastur bulk ``[sigma^2 (1-sqrt(g))^2, sigma^2 (1+sqrt(g))^2]``
(``g = D/N``) *even when the population covariance is pure noise*. Directions below the upper
edge therefore carry no evidence of signal, and inverting them injects sampling noise into the
edit with weight ``1/lambda``. The BBP transition says the same thing from the other side: a
population spike only emerges from the bulk once it exceeds ``sigma^2 (1 + sqrt(g))``.

Two estimators:

* :func:`mp_rank` — the textbook form, fixed-point on ``sigma^2`` with ``g = D/N``. Cheap and
  assumption-faithful, but ``N`` counts *tokens*, which are correlated inside a sequence, so the
  effective sample size is smaller than ``N`` and this over-estimates the rank.
* :func:`mp_rank_fitted` — fits the bulk edge from the spectrum itself, treating ``g`` as a free
  parameter. Absorbs the correlation-shrunken effective sample size without needing to know it.

Both consume eigenvalues only (descending), so they add no decomposition cost.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def mp_rank(eigenvalues: torch.Tensor, n_samples: int, *, max_iter: int = 20) -> Tuple[int, float]:
    """Fixed-point Marchenko-Pastur rank. Returns ``(k, upper_edge)``.

    Iterates: treat the top ``k`` as signal, estimate the noise level from the rest, recompute
    the bulk edge, recount. Converges in a handful of steps (the map is monotone in ``k``).
    """
    lam = eigenvalues.double().clamp_min(0).flatten()
    d = lam.numel()
    if d == 0 or n_samples <= 0:
        return 0, 0.0
    g = d / float(n_samples)
    k = 0
    edge = float("inf")
    for _ in range(max_iter):
        tail = lam[k:]
        if tail.numel() == 0:
            break
        sigma2 = float(tail.mean())
        edge = sigma2 * (1.0 + math.sqrt(g)) ** 2
        k_new = int((lam > edge).sum())
        if k_new == k:
            break
        k = k_new
    return k, edge


def mp_rank_fitted(eigenvalues: torch.Tensor, *, quantile: float = 0.5,
                   max_iter: int = 30) -> Tuple[int, float]:
    """Bulk edge fitted from the spectrum, with the aspect ratio ``g`` treated as free.

    Uses two robust statistics of the assumed-noise tail — its mean (``sigma^2``) and its spread
    — to solve for ``g``: for the MP law the tail variance satisfies ``var/mean^2 = g``. That
    substitutes an *effective* aspect ratio for ``D/N``, which is what correlated tokens need.
    """
    lam = eigenvalues.double().clamp_min(0).flatten()
    d = lam.numel()
    if d == 0:
        return 0, 0.0
    k = int(d * (1.0 - quantile))
    edge = float("inf")
    for _ in range(max_iter):
        tail = lam[k:]
        if tail.numel() < 8:
            break
        m = float(tail.mean())
        v = float(tail.var(unbiased=True))
        if m <= 0:
            break
        g_eff = min(max(v / (m * m), 1e-8), 4.0)      # MP: Var/mean^2 = g
        edge = m * (1.0 + math.sqrt(g_eff)) ** 2
        k_new = int((lam > edge).sum())
        if k_new == k:
            break
        k = k_new
    return k, edge


def effective_rank_report(eigenvalues: torch.Tensor, n_samples: Optional[int] = None) -> dict:
    """Both estimators plus the energy/participation references, for diagnostics."""
    lam = eigenvalues.double().clamp_min(0).flatten()
    d = lam.numel()
    out = {"D": d}
    if n_samples:
        k, e = mp_rank(lam, n_samples)
        out.update(k_mp=k, mp_edge=e, k_mp_frac=k / d)
    k2, e2 = mp_rank_fitted(lam)
    cum = torch.cumsum(lam, 0) / lam.sum().clamp_min(1e-30)
    out.update(k_mp_fit=k2, mp_fit_edge=e2, k_mp_fit_frac=k2 / d,
               r99=int((cum < 0.99).sum()) + 1,
               participation=float(lam.sum() ** 2 / (lam * lam).sum()))
    return out
