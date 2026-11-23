"""Spectral pseudo-inverse of symmetric PSD covariances.

Two cut criteria over the same eigendecomposition ``C = U diag(lam) U^T``:

- ``rank_fraction=None`` (default): reproduce ``torch.linalg.pinv(C, rtol=D*eps)``
  — keep eigenvalues above ``rtol * lam_max``. For a symmetric PSD matrix the
  eigendecomposition and the SVD coincide, so this is the same operator computed
  ~severalfold faster than the general-purpose SVD path.
- ``rank_fraction=f`` in ``(0, 1]``: **rank-based cut** — keep the top
  ``ceil(f * D)`` eigenpairs by eigenvalue, independent of the spectrum's
  conditioning. Within the kept subspace no ridge/damping is needed; a small
  absolute floor (same ``D*eps*lam_max`` formula) still guards genuinely null
  directions inside the kept block.

``spectral_factors`` returns the truncated factors ``(U_k, inv_lam_k)`` so callers
can apply ``pinv = U_k diag(inv_lam_k) U_k^T`` in factored form (low-rank
application scales with ``k`` and never materializes the ``D x D`` inverse).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch

# The singular-value cut is a REGULARIZATION strength, not a numerical-noise floor, so it must
# not move when the storage or compute dtype changes. Historically it was
# ``D * torch.finfo(float32).eps``; that number is pinned here as a constant so a float64 run
# solves the SAME problem (more accurately) instead of a different one — float64's own eps
# would drop the threshold by nine orders of magnitude and 1/sigma-amplify pure noise.
EPS_FP32 = 1.1920928955078125e-07


def default_rtol(dim: int) -> float:
    """Dtype-independent relative cut: ``D * eps_float32`` (condition cap ~ 1 / rtol).

    Note this makes the cut *width-dependent*: the implied condition cap is
    ``1 / (D * eps32)`` — 8192 at ``D=1024`` but only 328 at ``D=25600``, i.e. wider
    layers are regularized harder. That is inherited from the numerical-rank heuristic;
    pass ``condition_cap`` to impose one cap on every layer instead.
    """
    return dim * EPS_FP32


@torch.no_grad()
def spectral_factors(
    c: torch.Tensor,
    *,
    rank_fraction: Optional[float] = None,
    rtol: Optional[float] = None,
    condition_cap: Optional[float] = None,
    method: str = "exact",
    floor: str = "rtol",
    cut: str = "rtol",
    compute_dtype: Optional[torch.dtype] = torch.float64,
    energy_fraction: float = 0.99,
    ridge_delta: float = 1e-6,
    n_samples: Optional[int] = None,
    oversample: int = 16,
    n_iter: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Truncated eigen-factors of ``pinv(C)`` for symmetric PSD ``C``.

    Returns ``(U_k, inv_lam_k)`` with ``U_k`` of shape ``[D, k]`` (descending
    eigenvalue order) such that ``pinv(C) ~= U_k @ diag(inv_lam_k) @ U_k.T``.

    Args:
        rank_fraction: ``None`` keeps every direction above the ``rtol`` floor
            (the historical cut). A float ``f`` keeps the top ``ceil(f*D)``.
        rtol: relative singular-value cut; defaults to :func:`default_rtol` (``D * eps32``),
            a fixed number that does **not** follow the compute dtype.
        condition_cap: alternative, layer-uniform way to say the same thing — invert only
            directions within a factor ``condition_cap`` of the largest eigenvalue
            (``rtol = 1 / condition_cap``). Mutually exclusive with ``rtol``.
        method: ``"exact"`` runs a full ``eigh`` and truncates — exact top-k, cost
            ``O(D^3)``. ``"randomized"`` runs randomized subspace iteration on the
            top ``k`` only — cost ``O(D^2 k)``, memory ``O(D k)``; requires
            ``rank_fraction``.
        floor: ``"rtol"`` also drops kept directions below ``rtol * lam_max``
            (composes the two cuts). ``"none"`` makes the rank cut the *only*
            criterion, guarding solely against non-positive eigenvalues.
        oversample, n_iter: randomized-solver knobs (extra probe columns and power
            iterations); accuracy rises with both.
    """
    d = c.shape[-1]
    out_dtype = c.dtype
    if condition_cap is not None:
        if rtol is not None:
            raise ValueError("pass either rtol or condition_cap, not both")
        if condition_cap <= 1.0:
            raise ValueError(f"condition_cap must be > 1, got {condition_cap}")
        rtol = 1.0 / condition_cap   # keep lambda within a factor `condition_cap` of lambda_max
    if rtol is None:
        rtol = default_rtol(d)
    if compute_dtype is not None and c.dtype != compute_dtype:
        # Upcasting a float32 matrix to float64 is lossless: the decomposition is of the exact
        # stored covariance, only solved to ~1e-16 instead of ~1e-7. That is what makes the
        # keep-set deterministic (measured: identical for eigh and SVD, 0 mismatches).
        c = c.to(compute_dtype)
    if rank_fraction is not None and not 0.0 < rank_fraction <= 1.0:
        raise ValueError(f"rank_fraction must be in (0, 1], got {rank_fraction}")

    if method not in ("exact", "randomized"):
        raise ValueError(f"method must be 'exact' or 'randomized', got {method!r}")
    if method == "randomized":
        if rank_fraction is None:
            raise ValueError("method='randomized' requires rank_fraction")
        k = max(1, math.ceil(rank_fraction * d))
        q = min(d, k + max(0, oversample))
        u, lam, _ = torch.svd_lowrank(c, q=q, niter=n_iter)  # C symmetric PSD -> U ~= V, S ~= lam
        lam = lam[:k]
        u = u[:, :k]
    elif method == "exact":
        sym = (c + c.mT) * 0.5  # eigh requires exact symmetry; C is symmetric up to fp noise
        lam, u = torch.linalg.eigh(sym)  # ascending
        lam = lam.flip(-1)
        u = u.flip(-1)
        if rank_fraction is not None:
            k = max(1, math.ceil(rank_fraction * d))
            lam = lam[:k]
            u = u[:, :k]
    else:
        raise ValueError(f"method must be 'exact' or 'randomized', got {method!r}")

    lam_max = lam[0].clamp_min(0) if lam.numel() else lam.new_zeros(())

    if cut == "ridge":
        # Tikhonov: no truncation at all — inv_lam = 1/(lam + eps). Smooth in the spectrum,
        # so two different solvers agree to machine precision (no knife-edge keep-set).
        eps = ridge_delta * lam_max
        pos = lam > 0
        return u[:, pos].to(out_dtype), (1.0 / (lam[pos] + eps)).to(out_dtype)

    if cut in ("mp", "mp_n"):
        # Random-matrix cut: keep only directions above the sample-noise bulk edge.
        # "mp"   fits the aspect ratio from the spectrum (absorbs correlated tokens);
        # "mp_n" uses the textbook D/N with the observed sample count.
        from .rmt import mp_rank, mp_rank_fitted
        if cut == "mp_n":
            if n_samples is None:
                raise ValueError("cut='mp_n' requires n_samples")
            k_mp, _ = mp_rank(lam, int(n_samples))
        else:
            k_mp, _ = mp_rank_fitted(lam)
        k_mp = max(1, k_mp)
        keep = torch.zeros_like(lam, dtype=torch.bool)
        keep[:k_mp] = True
        keep &= lam > 0
    elif cut == "energy":
        # Keep the smallest prefix carrying `energy_fraction` of the trace. A criterion on a
        # smooth cumulative sum rather than a threshold comparison inside a dense spectrum.
        if not 0.0 < energy_fraction <= 1.0:
            raise ValueError(f"energy_fraction must be in (0, 1], got {energy_fraction}")
        pos = lam.clamp_min(0)
        total = pos.sum().clamp_min(torch.finfo(lam.dtype).tiny)
        cum = torch.cumsum(pos, 0) / total
        k_e = int((cum < energy_fraction).sum()) + 1
        keep = torch.zeros_like(lam, dtype=torch.bool)
        keep[:k_e] = True
        keep &= lam > 0
    elif cut == "rtol":
        if floor == "rtol":
            keep = lam > rtol * lam_max
        elif floor == "none":
            keep = lam > 0
        else:
            raise ValueError(f"floor must be 'rtol' or 'none', got {floor!r}")
    else:
        raise ValueError(f"cut must be 'rtol', 'mp', 'mp_n', 'gap', 'energy' or 'ridge', got {cut!r}")
    return u[:, keep].to(out_dtype), (1.0 / lam[keep]).to(out_dtype)


@torch.no_grad()
def spectral_pinv(c: torch.Tensor, **kwargs) -> torch.Tensor:
    """Dense ``pinv(C)`` from :func:`spectral_factors` (same cut semantics)."""
    u_k, inv_lam = spectral_factors(c, **kwargs)
    return (u_k * inv_lam) @ u_k.mT
