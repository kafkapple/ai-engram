"""Sibling layers fed by one tensor share their input covariance.

q/k/v read the same post-attention LayerNorm output, and gate/up the same post-MLP one, so
their input covariances are identical by construction. The collector recognizes the repeat and
skips the redundant ``x^T x`` — the dominant cost of collection — without changing a single
number.

Run offline; CPU-only; deterministic.
"""
import torch
import torch.nn as nn

from engram import EditorConfig, EngramEditor
from engram.collectors import CovarianceCollector


class _Block(nn.Module):
    """One tensor feeding three siblings, then a private input for a fourth."""

    def __init__(self, d=16):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        h = torch.nn.functional.layer_norm(x, x.shape[-1:])
        y = self.q(h) + self.k(h) + self.v(h)      # one tensor -> three modules
        return self.o(y)                            # a different tensor -> one module


def _collect(model, batches, window, mask_fn=None):
    prev = CovarianceCollector._WINDOW
    CovarianceCollector._WINDOW = window
    try:
        ed = EngramEditor(model, EditorConfig(storage_device="cpu"))
        return ed.collect_statistics(batches, batch_fn=lambda b: b["x"], mask_fn=mask_fn)
    finally:
        CovarianceCollector._WINDOW = prev


def _batches(n=4, rows=8, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [{"x": torch.randn(rows, d, generator=g)} for _ in range(n)]


# T1: siblings off one tensor get identical covariances, and a private input does not.
def test_siblings_share_covariance():
    torch.manual_seed(0)
    m = _Block()
    st = _collect(m, _batches(), CovarianceCollector._WINDOW)
    assert torch.equal(st["q"], st["k"]) and torch.equal(st["q"], st["v"])
    assert not torch.equal(st["q"], st["o"])
    assert st.count["q"] == st.count["k"] == st.count["v"] == st.count["o"]


# T2: sharing is a pure optimization — results are bit-identical with it disabled.
def test_sharing_is_bit_identical():
    torch.manual_seed(0)
    m = _Block()
    on = _collect(m, _batches(), CovarianceCollector._WINDOW)
    torch.manual_seed(0)
    m2 = _Block()
    off = _collect(m2, _batches(), 0)
    assert set(on.keys()) == set(off.keys())
    for name in on:
        assert torch.equal(on[name], off[name])
        assert on.count[name] == off.count[name]


# T3: the redundant products are actually skipped (2 of every 4 hooks here).
def test_shared_hits_counted():
    torch.manual_seed(0)
    m = _Block()
    ed = EngramEditor(m, EditorConfig(storage_device="cpu"))
    col = CovarianceCollector(m, EditorConfig(storage_device="cpu"), ed.registry)
    with col:
        for b in _batches():
            col.begin_batch()
            m(b["x"])
    assert col.shared_hits == 2 * 4          # k and v reuse q's product, every batch
    assert col.sample_counts["q"] == 4 * 8


# T4: masking still applies per group (the mask is a property of the shared input).
def test_masked_sharing():
    torch.manual_seed(0)
    m = _Block()
    g = torch.Generator().manual_seed(1)
    batches = [{"x": torch.randn(8, 16, generator=g), "m": torch.tensor([True] * 5 + [False] * 3)}
               for _ in range(3)]
    ed = EngramEditor(m, EditorConfig(storage_device="cpu"))
    st = ed.collect_statistics(batches, batch_fn=lambda b: b["x"], mask_fn=lambda b: b["m"])
    assert st.count["q"] == 3 * 5            # masked rows excluded
    assert torch.equal(st["q"], st["v"])     # and the group still shares


# T5: dedupe collapses identical covariances onto one tensor.
def test_dedupe_collapses_identical():
    torch.manual_seed(0)
    st = _collect(_Block(), _batches(), CovarianceCollector._WINDOW)
    assert st["q"] is st["k"] and st["q"] is st["v"]
    assert st["q"] is not st["o"]
    assert len({id(v) for v in st.cov.values()}) == 2


# T6: the file stores a shared covariance once, tagged apart so an older reader fails loudly
# rather than silently dropping the aliased layers; load restores the sharing.
def test_save_dedupes_and_tags_aliases(tmp_path):
    from engram import Statistics
    from engram.stats import _FORMAT, _FORMAT_ALIASED

    torch.manual_seed(0)
    st = _collect(_Block(), _batches(), CovarianceCollector._WINDOW)
    shared, unshared = tmp_path / "shared.pt", tmp_path / "unshared.pt"
    st.save(shared)
    Statistics({k: v.clone() for k, v in st.cov.items()}, dict(st.count)).save(unshared)
    assert shared.stat().st_size < unshared.stat().st_size
    assert torch.load(shared, weights_only=True)["format"] == _FORMAT_ALIASED
    assert torch.load(unshared, weights_only=True)["format"] == _FORMAT
    back = Statistics.load(shared)
    assert back["q"] is back["k"] and back["q"] is back["v"]
    for name in st:
        assert torch.equal(back[name], st[name])


# T7: the engram decomposes each distinct covariance once, and the result is unchanged.
def test_engram_reuses_factors_per_distinct_covariance():
    import engram.editor as editor_mod
    from engram import Statistics

    torch.manual_seed(0)
    m = _Block()
    st = _collect(m, _batches(), CovarianceCollector._WINDOW)
    tgt = _collect(m, _batches(seed=1), CovarianceCollector._WINDOW)
    ed = EngramEditor(m, EditorConfig(storage_device="cpu"))
    calls, orig = [0], editor_mod.spectral_factors

    def counting(*a, **k):
        calls[0] += 1
        return orig(*a, **k)

    editor_mod.spectral_factors = counting
    try:
        shared = ed.compute_engram_weights(tgt, st)
    finally:
        editor_mod.spectral_factors = orig
    assert calls[0] == 2                                   # not 4 — one per distinct covariance
    plain = ed.compute_engram_weights(
        tgt, Statistics({k: v.clone() for k, v in st.cov.items()}, dict(st.count)))
    for name in plain.layers:
        assert torch.equal(shared.layers[name].projection, plain.layers[name].projection)


# T8: moving preserves sharing.
def test_to_preserves_sharing():
    torch.manual_seed(0)
    st = _collect(_Block(), _batches(), CovarianceCollector._WINDOW)
    moved = st.to("cpu")
    assert moved["q"] is moved["k"] and moved["q"] is moved["v"]
    for name in st:
        assert torch.equal(moved[name], st[name])


# --- regressions for the three defects the pre-release audit reproduced ---

class _ReuseDifferentInput(nn.Module):
    """One module applied twice, to two DIFFERENT tensors, inside one forward."""

    def __init__(self, d=8):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)

    def forward(self, x):
        h = torch.nn.functional.layer_norm(x, x.shape[-1:])
        y = self.q(h) + self.v(h)
        return y + self.v(torch.nn.functional.layer_norm(x * 3, x.shape[-1:]))


class _Direct(nn.Module):
    """Layers that consume the caller's batch tensor itself."""

    def __init__(self, d=16):
        super().__init__()
        self.a = nn.Linear(d, d, bias=False)
        self.b = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.b(self.a(x))


# R1: a module applied twice to different tensors must count both.
def test_module_reused_on_different_tensors():
    batches = _batches(d=8)
    torch.manual_seed(0)
    model_a = _ReuseDifferentInput()
    torch.manual_seed(0)
    model_b = _ReuseDifferentInput()
    on = _collect(model_a, batches, CovarianceCollector._WINDOW)
    off = _collect(model_b, batches, 0)
    assert on.count == off.count and on.count["v"] == 2 * on.count["q"]
    for name in off:
        assert torch.equal(on[name], off[name])


# R2: the window must not survive a batch boundary — a reused batch OBJECT holding NEW rows is a
# new batch. Reusing the object with unchanged contents would pass either way, which is how a
# never-cleared window hid here before.
def _refilling_loader(rows, batch=6):
    """A loader that refills one preallocated tensor, as a pinned-buffer pipeline would."""
    buf = torch.empty(batch, rows.shape[1])

    def gen():
        for i in range(0, len(rows), batch):
            buf.copy_(rows[i:i + batch])
            yield {"x": buf}
    return gen


def test_refilled_batch_buffer():
    g = torch.Generator().manual_seed(3)
    rows = torch.randn(24, 16, generator=g)
    torch.manual_seed(0)
    model_a = _Direct()
    torch.manual_seed(0)
    model_b = _Direct()                      # same weights, so any difference is the sharing
    loader = _refilling_loader(rows)
    on = _collect(model_a, loader(), CovarianceCollector._WINDOW)
    off = _collect(model_b, loader(), 0)
    assert on.count == off.count and on.count["a"] == 24
    for name in off:
        assert torch.equal(on[name], off[name])
    # and the covariance is the whole corpus, not just the first batch
    ref = rows.T @ rows / 24
    assert torch.allclose(on["a"], ref, atol=1e-5)


# R2b: an in-place write into a shared covariance is visible through every name (documented).
def test_shared_covariance_is_aliased():
    torch.manual_seed(0)
    st = _collect(_Block(), _batches(), CovarianceCollector._WINDOW)
    st["q"].mul_(2.0)
    assert torch.equal(st["k"], st["q"])     # same object, by design — treat as read-only


# R3: the same tensor object under a different mask is a different measurement.
def test_same_object_different_masks():
    x = torch.randn(6, 16, generator=torch.Generator().manual_seed(4))
    batches = [{"x": x, "m": torch.tensor([True] * 6)},
               {"x": x, "m": torch.tensor([True] * 3 + [False] * 3)}]
    torch.manual_seed(0)
    model_a = _Direct()
    torch.manual_seed(0)
    model_b = _Direct()
    on = _collect(model_a, batches, CovarianceCollector._WINDOW, mask_fn=lambda b: b["m"])
    off = _collect(model_b, batches, 0, mask_fn=lambda b: b["m"])
    assert on.count == off.count and on.count["a"] == 9
    for name in off:
        assert torch.equal(on[name], off[name])
