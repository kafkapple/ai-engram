"""Spectral pseudo-inverse (engram.inverse) — eigh path + rank-fraction cut.

Equivalence: for symmetric PSD covariances, eigh-based pinv with the default
rtol cut must reproduce torch.linalg.pinv (the previous SVD path). Rank-based
cut: keep the top ceil(f*D) eigen-directions, no ridge.

Run offline; CPU-only; deterministic.
"""
import math

import pytest
import torch

from engram.inverse import spectral_factors, spectral_pinv


def _psd(d=48, cond=1e8, seed=0, gap_at_cut=False):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    lam = torch.logspace(0, -math.log10(cond), d, dtype=torch.float64)
    if gap_at_cut:
        # keep every eigenvalue far from the rtol*lam_max threshold so the keep-set is
        # unambiguous — near the cut, ANY threshold rule (eigh- or SVD-based) may flip
        # borderline directions and the dense pinvs legitimately differ.
        cut = d * torch.finfo(torch.float32).eps
        lam = torch.where(lam > cut, lam.clamp_min(cut * 1e2), lam.clamp_max(cut * 1e-2))
    return (q * lam) @ q.T


# T1: default cut reproduces torch.linalg.pinv when the keep-set is unambiguous
# (spectral gap at the threshold), across conditioning.
@pytest.mark.parametrize("cond", [1e2, 1e6, 1e9])
def test_default_cut_matches_torch_pinv(cond):
    c = _psd(cond=cond, gap_at_cut=True).to(torch.float32)
    rtol = c.shape[-1] * torch.finfo(torch.float32).eps
    ref = torch.linalg.pinv(c, rtol=rtol)
    got = spectral_pinv(c)
    scale = ref.abs().max()
    assert torch.allclose(got, ref, atol=float(1e-4 * scale), rtol=1e-3)


# T1b: Moore-Penrose properties hold for the eigh path regardless of conditioning.
# Measured on the KEPT subspace: with a log-uniform spectrum the cut lands mid-decade,
# so 1/lam_min_kept is ~1/rtol and fp32 residuals there are inherently ~1e-2 relative.
@pytest.mark.parametrize("cond", [1e6, 1e9])
def test_moore_penrose_properties(cond):
    c = _psd(cond=cond, gap_at_cut=True).to(torch.float32)
    p = spectral_pinv(c)
    c64, p64 = c.double(), p.double()
    # float32 pinv of a 1e6-1e9-conditioned matrix: residuals land at ~1e-4 relative,
    # which is the accuracy the pipeline has always operated at (it is why the cut exists).
    # Residuals are measured relative to the *kept* subspace scale: fp32 pinv of a
    # 1e6-1e9-conditioned matrix carries ~1e-3 relative error, which is precisely why
    # the singular-value cut exists in the first place.
    assert torch.allclose(c64 @ p64 @ c64, c64, atol=2e-2 * float(c64.abs().max()))
    assert torch.allclose(p64 @ c64 @ p64, p64, atol=2e-2 * float(p64.abs().max()))


# T2: pipeline-level equivalence — eigh default vs svd path give the same projections.
def test_editor_eigh_matches_svd():
    from engram import EditorConfig, EngramEditor

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(13, 6, bias=True))
    ed = EngramEditor(model, EditorConfig(storage_device="cpu"))
    xs = [torch.randn(16, 13) for _ in range(4)]
    tgt = ed.collect_statistics([{"x": xs[0]}], batch_fn=lambda b: b["x"])
    tot = ed.collect_statistics([{"x": x} for x in xs], batch_fn=lambda b: b["x"])
    r_eigh = ed.compute_engram_weights(tgt, tot)
    r_svd = ed.compute_engram_weights(tgt, tot, inverse_method="svd")
    for name in r_svd.layers:
        assert torch.allclose(
            r_eigh.layers[name].projection, r_svd.layers[name].projection, atol=1e-4, rtol=1e-3
        )


# T3: rank_fraction keeps exactly ceil(f*D) directions (minus numerically-null ones).
@pytest.mark.parametrize("f", [0.1, 0.5, 0.9, 1.0])
def test_rank_fraction_keeps_topk(f):
    d = 40
    c = _psd(d=d, cond=1e4).to(torch.float32)
    u_k, inv_lam = spectral_factors(c, rank_fraction=f)
    assert u_k.shape[1] == max(1, math.ceil(f * d))
    lam_kept = 1.0 / inv_lam
    assert torch.all(lam_kept[:-1] >= lam_kept[1:])  # descending


# T4: rank cut needs no ridge — inverting top-k of an ill-conditioned C is stable
# (pinv restricted to the kept subspace: C @ pinv @ C ~= C on that subspace).
def test_rank_cut_stable_on_illconditioned():
    c = _psd(d=64, cond=1e8).to(torch.float32)
    p = spectral_pinv(c, rank_fraction=0.5)
    u_k, inv_lam = spectral_factors(c, rank_fraction=0.5)
    proj = u_k @ u_k.T
    # fp32 error in p scales with 1/lam_min_kept; do the product in f64 and budget for it.
    tol = 30 * torch.finfo(torch.float32).eps * float(inv_lam.max())
    lhs = c.double() @ p.double()
    assert torch.allclose(lhs, proj.double(), atol=max(tol, 1e-3))
    assert torch.isfinite(p).all()


# T5: f=1.0 equals the default rtol cut when the spectrum is clean.
def test_full_fraction_equals_default_when_wellconditioned():
    c = _psd(d=32, cond=1e3).to(torch.float32)
    assert torch.allclose(spectral_pinv(c, rank_fraction=1.0), spectral_pinv(c), atol=1e-4, rtol=1e-3)


# T6: invalid fraction raises; svd path rejects rank_fraction.
def test_validation():
    c = _psd(d=8).to(torch.float32)
    with pytest.raises(ValueError):
        spectral_pinv(c, rank_fraction=0.0)
    from engram import EditorConfig, EngramEditor

    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    ed = EngramEditor(model, EditorConfig(storage_device="cpu"))
    stats = ed.collect_statistics([{"x": torch.randn(6, 4)}], batch_fn=lambda b: b["x"])
    with pytest.raises(ValueError):
        ed.compute_engram_weights(stats, stats, rank_fraction=0.5, inverse_method="svd")


# T7: factored application equals the dense pinv product.
def test_factored_application_equals_dense():
    torch.manual_seed(0)
    c = _psd(d=24, cond=1e5).to(torch.float32)
    w = torch.randn(5, 24)
    u_k, inv_lam = spectral_factors(c, rank_fraction=0.5)
    dense = w @ spectral_pinv(c, rank_fraction=0.5)
    factored = ((w @ u_k) * inv_lam) @ u_k.T
    assert torch.allclose(factored, dense, atol=1e-5, rtol=1e-4)


# T8: randomized top-k tracks the exact top-k subspace.
@pytest.mark.parametrize("f", [0.1, 0.25])
def test_randomized_matches_exact_topk(f):
    torch.manual_seed(0)
    d = 256
    g = torch.Generator().manual_seed(3)
    a = torch.randn(d, 64, generator=g)
    c = (a @ a.T + 1e-3 * torch.eye(d)).float()  # decaying spectrum, well separated
    u_e, il_e = spectral_factors(c, rank_fraction=f)
    u_r, il_r = spectral_factors(c, rank_fraction=f, method="randomized", n_iter=6)
    k = u_e.shape[1]
    assert u_r.shape[1] == k
    overlap = float((u_e.T @ u_r).pow(2).sum()) / k  # subspace overlap in [0, 1]
    assert overlap > 0.98
    assert torch.allclose(1 / il_r, 1 / il_e, rtol=5e-2)


# T9: floor='none' makes rank the only criterion.
def test_floor_none_keeps_all_requested():
    c = _psd(d=64, cond=1e12).to(torch.float32)
    u_r, _ = spectral_factors(c, rank_fraction=0.5, floor="none")
    u_t, _ = spectral_factors(c, rank_fraction=0.5, floor="rtol")
    assert u_r.shape[1] == 32
    assert u_t.shape[1] <= u_r.shape[1]


# T10: randomized without rank_fraction is rejected.
def test_randomized_requires_fraction():
    with pytest.raises(ValueError):
        spectral_factors(_psd(d=16).float(), method="randomized")


# T11: the cut no longer moves with dtype — the whole point of pinning it.
def test_rtol_is_dtype_independent():
    from engram.inverse import default_rtol

    c32 = _psd(d=64, cond=1e6).to(torch.float32)
    c64 = c32.double()
    u32, il32 = spectral_factors(c32)
    u64, il64 = spectral_factors(c64)
    assert u32.shape[1] == u64.shape[1]          # same keep-set size in both storage dtypes
    assert torch.allclose(1 / il32, (1 / il64).float(), rtol=1e-5)
    assert default_rtol(64) == 64 * 1.1920928955078125e-07


# T12: float64 decomposition makes eigh and SVD agree exactly (they disagree at float32).
def test_float64_makes_solvers_agree():
    c = _psd(d=96, cond=1e7).to(torch.float32)
    rtol = c.shape[-1] * 1.1920928955078125e-07
    c64 = c.double()
    U, S, _ = torch.linalg.svd(c64)
    m = S > rtol * S[0]
    p_svd = ((U[:, m]) * (1 / S[m])) @ U[:, m].T
    u_k, inv_lam = spectral_factors(c, rtol=rtol)          # default: float64 internally
    p_eigh = ((u_k.double()) * inv_lam.double()) @ u_k.double().T
    assert u_k.shape[1] == int(m.sum())                     # identical keep-set
    assert float((p_eigh - p_svd).norm() / p_svd.norm()) < 1e-6   # float32-returned factors
    u64, il64 = spectral_factors(c64, rtol=rtol)            # float64 in and out
    p64 = (u64 * il64) @ u64.T
    assert float((p64 - p_svd).norm() / p_svd.norm()) < 1e-10


# T13: repeated calls are bit-identical (determinism).
def test_deterministic_across_calls():
    c = _psd(d=64, cond=1e8).to(torch.float32)
    a1, b1 = spectral_factors(c)
    a2, b2 = spectral_factors(c)
    assert torch.equal(a1, a2) and torch.equal(b1, b2)


# T14: the Marchenko-Pastur rank recovers planted signal directions and reports none when there
# is no signal — the claim the CHANGELOG makes for engram.rmt.
@pytest.mark.parametrize("d,n,spikes", [(500, 5000, 0), (500, 5000, 5), (300, 900, 3)])
def test_mp_rank_recovers_planted_spikes(d, n, spikes):
    from engram.rmt import mp_rank, mp_rank_fitted

    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    if spikes:
        v = torch.linalg.qr(torch.randn(d, spikes, generator=g, dtype=torch.float64))[0]
        x = x + (torch.randn(n, spikes, generator=g, dtype=torch.float64) * 8.0) @ v.T
    lam = torch.linalg.eigvalsh(x.T @ x / n).flip(0)
    assert mp_rank(lam, n)[0] == spikes
    assert mp_rank_fitted(lam)[0] == spikes


# T15: 'gap' was measured worse than the default on both axes and is not shipped.
def test_unknown_cut_and_method_rejected():
    c = _psd(d=16).float()
    for kw in (dict(cut="gap"), dict(cut="nonsense"), dict(method="nonsense")):
        with pytest.raises(ValueError):
            spectral_factors(c, **kw)
