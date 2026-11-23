"""Packed (upper-triangular) Statistics storage — format 3.

Covariances are symmetric, so the on-disk file stores only the upper triangle
(diagonal included): ~half the bytes, upper triangle bit-exact, lower triangle
reconstructed by mirroring. Dense ``format=2`` files still load.

Run offline; CPU-only; deterministic.
"""
import os

import pytest
import torch

from engram.stats import _FORMAT, _FORMAT_DENSE, Statistics, _pack_symmetric, _unpack_symmetric


def _random_stats(dims=(7, 32), seed=0):
    g = torch.Generator().manual_seed(seed)
    cov, count = {}, {}
    for i, d in enumerate(dims):
        x = torch.randn(4 * d, d, generator=g)
        cov[f"layer{i}"] = (x.T @ x) / (4 * d)  # symmetric up to fp accumulation noise
        count[f"layer{i}"] = 4 * d
    return Statistics(cov, count)


# T1: pack/unpack round-trips: upper triangle bit-exact, whole matrix within fp asymmetry.
def test_pack_roundtrip_bitexact_upper():
    stats = _random_stats()
    for name, c in stats.items():
        back = _unpack_symmetric(_pack_symmetric(c))
        iu = torch.triu_indices(c.shape[0], c.shape[0])
        assert torch.equal(back[iu[0], iu[1]], c[iu[0], iu[1]])
        asym = (c - c.T).abs().max()
        assert (back - c).abs().max() <= asym
        assert torch.allclose(back, c, atol=1e-6, rtol=1e-6)
        assert torch.equal(back, back.T)


# T2: save(packed=True) -> load returns equal Statistics (and the file is ~half the dense size).
def test_packed_save_load_and_size(tmp_path):
    stats = _random_stats(dims=(96, 128))
    dense_p, packed_p = tmp_path / "dense.pt", tmp_path / "packed.pt"
    stats.save(dense_p, packed=False)
    stats.save(packed_p)
    loaded = Statistics.load(packed_p)
    assert set(loaded.keys()) == set(stats.keys())
    assert loaded.count == stats.count
    for name in stats:
        assert torch.allclose(loaded[name], stats[name], atol=1e-6, rtol=1e-6)
    assert os.path.getsize(packed_p) < 0.62 * os.path.getsize(dense_p)


# T3: dense format=2 files still load, bit-exact.
def test_dense_format2_still_loads(tmp_path):
    stats = _random_stats()
    p = tmp_path / "dense.pt"
    stats.save(p, packed=False)
    obj = torch.load(p, weights_only=True)
    assert obj["format"] == _FORMAT_DENSE
    loaded = Statistics.load(p)
    for name in stats:
        assert torch.equal(loaded[name], stats[name])


# T4: legacy untagged dicts are still rejected.
def test_legacy_rejected(tmp_path):
    p = tmp_path / "legacy.pt"
    torch.save({"layer0": torch.eye(3)}, p)
    with pytest.raises(ValueError):
        Statistics.load(p)


# T5: packed file feeds the engram pipeline identically to the in-memory stats.
def test_engram_from_packed_equals_inmemory(tmp_path):
    from engram import EditorConfig, EngramEditor

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(11, 5, bias=True))
    editor = EngramEditor(model, EditorConfig(storage_device="cpu"))
    xs = [torch.randn(8, 11) for _ in range(3)]
    target = editor.collect_statistics([{"x": x} for x in xs[:1]], batch_fn=lambda b: b["x"])
    total = editor.collect_statistics([{"x": x} for x in xs], batch_fn=lambda b: b["x"])
    p = tmp_path / "total.pt"
    total.save(p)
    total2 = Statistics.load(p)
    r1 = editor.compute_engram_weights(target, total)
    r2 = editor.compute_engram_weights(target, total2)
    for name in r1.layers:
        assert torch.allclose(r1.layers[name].projection, r2.layers[name].projection, atol=1e-5, rtol=1e-5)


# T6: format tag sanity.
def test_format_tags():
    assert _FORMAT == 3 and _FORMAT_DENSE == 2


# T7: a non-symmetric matrix falls back to dense inside the packed file and round-trips bit-exactly.
def test_asymmetric_entry_falls_back_dense(tmp_path):
    s = Statistics({"weird": torch.randn(9, 9)}, {"weird": 1})
    p = tmp_path / "weird.pt"
    s.save(p)
    loaded = Statistics.load(p)
    assert torch.equal(loaded["weird"], s["weird"])
