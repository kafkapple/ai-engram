"""Statistics container: per-layer **mean** input covariance + sample counts.

A :class:`Statistics` holds, per layer (and per fused-MoE expert), the *mean*
input covariance ``C = mean_k(x_k^T x_k)`` over the rows (tokens) that entered it,
plus the integer count ``N`` of those rows. Storing the mean — not the raw sum
``sum x^T x`` the earlier versions used — keeps the magnitude bounded (~``E[x^2]``)
regardless of corpus size, and makes :meth:`Statistics.merge` a count-weighted
average rather than a plain sum.

The closed-form engram is recovered **exactly** from means + counts: the paper's
``W . Sigma_target . pinv(Sigma_total)`` equals
``(n / N) . W . C_target . pinv(C_total)`` because ``pinv`` is scale-invariant (its
rcond cut is relative to the largest singular value). The ``n / N`` factor is applied
at edit time through the default scaling function — see :func:`engram.scaling.count_ratio`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Union

import torch

_FORMAT_ALIASED = 4  # packed + an alias map (layers sharing one covariance)
_FORMAT = 3          # packed, no aliases — still readable by 0.9.x
_FORMAT_DENSE = 2    # dense; the legacy untagged "sum-only dict" is rejected


_SYM_RTOL = 1e-5  # pack only when max|C - C^T| <= _SYM_RTOL * max|C| (fp accumulation noise passes)


def _pack_symmetric(c: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Pack a symmetric ``[D, D]`` matrix into its upper triangle (diagonal included).

    The upper triangle is stored bit-exactly; the (numerically identical) lower
    triangle is reconstructed by mirroring on load. Storage: ``D(D+1)/2`` values
    instead of ``D**2`` — a hair over half. A matrix that is *not* symmetric within
    ``_SYM_RTOL`` (relative) is stored dense unchanged, so arbitrary contents
    round-trip exactly.
    """
    if c.dim() != 2 or c.shape[0] != c.shape[1]:
        return {"dense": c.detach().cpu()}
    asym = (c - c.mT).abs().max()
    if asym > _SYM_RTOL * c.abs().max().clamp_min(torch.finfo(c.dtype).tiny):
        return {"dense": c.detach().cpu()}
    # Pack on the host: torch.triu_indices allocates an int64 [2, D(D+1)/2] tensor, twice the
    # bytes of a float32 covariance, and doing that on the covariance's own device OOMs a GPU
    # exactly when the matrix is large enough for packing to matter. torch.save copies CUDA
    # storages to host anyway, so moving first costs nothing.
    c = c.detach().cpu()
    d = c.shape[-1]
    iu = torch.triu_indices(d, d)
    return {"packed": c[iu[0], iu[1]].contiguous(), "dim": torch.tensor(d)}


def _unpack_symmetric(entry: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Rebuild the full matrix on the host, for the same index-tensor reason as packing."""
    if "dense" in entry:
        return entry["dense"]
    d = int(entry["dim"])
    p = entry["packed"].cpu()
    iu = torch.triu_indices(d, d)
    full = torch.empty(d, d, dtype=p.dtype)
    full[iu[0], iu[1]] = p
    full[iu[1], iu[0]] = p
    return full


@dataclass
class Statistics:
    """Per-layer mean input covariance (``cov``) and sample count (``count``).

    ``cov[name]`` is the mean of ``x^T x`` over the ``count[name]`` rows that entered
    layer ``name`` during collection. Keys are module names, or
    ``"<experts>.gate_up_proj.<e>"`` / ``"....down_proj.<e>"`` for fused-MoE experts.
    Behaves like a read-only mapping over ``cov`` (``stats[name]``, ``name in stats``,
    iteration) with the parallel ``count`` dict alongside.
    """

    cov: Dict[str, torch.Tensor] = field(default_factory=dict)
    count: Dict[str, int] = field(default_factory=dict)

    # ---- mapping-like surface over cov (count is the parallel dict) ----
    def __getitem__(self, key: str) -> torch.Tensor:
        return self.cov[key]

    def __contains__(self, key: str) -> bool:
        return key in self.cov

    def __iter__(self) -> Iterator[str]:
        return iter(self.cov)

    def __len__(self) -> int:
        return len(self.cov)

    def keys(self):
        return self.cov.keys()

    def items(self):
        return self.cov.items()

    def to(self, device: Union[str, torch.device]) -> "Statistics":
        """Move every covariance to ``device`` (counts are plain ints, copied as-is).

        Layers that share one covariance (q/k/v, gate/up) keep sharing it after the move —
        moving each member separately would silently triple the memory the collector saved.
        """
        moved: Dict[int, torch.Tensor] = {}
        cov: Dict[str, torch.Tensor] = {}
        for k, v in self.cov.items():
            oid = id(v)
            if oid not in moved:
                moved[oid] = v.to(device)
            cov[k] = moved[oid]
        return Statistics(cov, dict(self.count))

    @staticmethod
    def merge(*stats: "Statistics") -> "Statistics":
        """Count-weighted merge: ``C = sum(n_i C_i) / sum(n_i)``, ``N = sum(n_i)``.

        Equivalent to having collected over the concatenated token streams. Keys are
        unioned; a key present in only some inputs contributes only its own
        ``(count, mean)``. Combined incrementally (``C += n_i/(N+n_i) (C_i - C)``) so
        no large ``n_i * C_i`` intermediate is formed.

        Note that a merge materializes one tensor per key: layers that shared a covariance
        during collection no longer do afterwards. That costs memory on a merged result but
        keeps the arithmetic obviously correct; ``to()`` and ``save()`` do preserve sharing.
        """
        cov: Dict[str, torch.Tensor] = {}
        count: Dict[str, int] = {}
        for s in stats:
            for k, c in s.cov.items():
                n = int(s.count.get(k, 0))
                if k not in cov:
                    cov[k] = c.to(torch.float32).clone()
                    count[k] = n
                    continue
                prev = count[k]
                total = prev + n
                if total > 0:
                    c = c.to(cov[k].device, torch.float32)
                    cov[k] = cov[k] + (n / total) * (c - cov[k])
                count[k] = total
        return Statistics(cov, count)

    def dedupe(self) -> "Statistics":
        """Collapse bit-identical covariances onto one tensor each.

        Layers fed by the same input — ``q``/``k``/``v`` off one LayerNorm, ``gate``/``up`` off
        the other — accumulate covariances that are equal to the last bit. Sharing one tensor
        between them costs nothing (they are read-only from here on) and saves that fraction of
        memory, file size and eigendecompositions.

        This runs *after* collection rather than during it, so the accumulation itself stays
        exactly what it was: every layer folds every batch it saw, with its own count. Only
        tensors that are already identical, and whose counts agree, are merged.

        The merged covariances are the *same object*, so writing into one in place writes into
        all of them. Treat them as read-only — or call :meth:`merge` on the result, which
        materializes one tensor per key.
        """
        by_shape: Dict[Any, List[str]] = {}
        for name, c in self.cov.items():
            by_shape.setdefault((tuple(c.shape), c.dtype, c.device, self.count.get(name)), []).append(name)
        cov = dict(self.cov)
        for names in by_shape.values():
            if len(names) < 2:
                continue
            canon: List[str] = []
            for name in names:
                for c0 in canon:
                    if cov[name] is cov[c0] or torch.equal(cov[name], cov[c0]):
                        cov[name] = cov[c0]
                        break
                else:
                    canon.append(name)
        return Statistics(cov, dict(self.count))

    def save(self, path: Union[str, Path], *, packed: bool = True) -> None:
        """Save with ``torch.save``.

        ``packed=True`` (default, tag ``format=3``) stores each covariance as its
        upper triangle only — the matrices are symmetric, so this halves the file
        with the upper triangle preserved bit-exactly. ``packed=False`` writes the
        dense ``format=2`` layout for compatibility with older readers.
        """
        if packed:
            # Layers fed by the same tensor (q/k/v, gate/up) hold the SAME accumulator object,
            # so the file stores it once and records who shares it. One packed copy is live at
            # a time.
            packed_cov, alias, owner_of = {}, {}, {}
            for k, v in self.cov.items():
                oid = id(v)
                if oid in owner_of:
                    alias[k] = owner_of[oid]
                    continue
                owner_of[oid] = k
                packed_cov[k] = _pack_symmetric(v)
            # A reader that does not know about aliases would silently lose the aliased layers,
            # so a file that has them gets its own tag and fails loudly on 0.9.x instead.
            obj = {"format": _FORMAT_ALIASED if alias else _FORMAT,
                   "cov_packed": packed_cov, "count": self.count}
            if alias:
                obj["alias"] = alias
            torch.save(obj, path)
        else:
            torch.save({"format": _FORMAT_DENSE, "cov": self.cov, "count": self.count}, path)

    @staticmethod
    def load(
        path: Union[str, Path], map_location: Union[str, torch.device, None] = None
    ) -> "Statistics":
        """Load a :class:`Statistics`. Rejects the legacy raw-covariance dict format."""
        obj = torch.load(path, map_location=map_location, weights_only=True)
        if isinstance(obj, dict) and obj.get("format") in (_FORMAT, _FORMAT_ALIASED) and "cov_packed" in obj:
            # unpack on the host, then honour map_location
            cov = {k: _unpack_symmetric(v) for k, v in obj["cov_packed"].items()}
            if map_location is not None:
                cov = {k: v.to(map_location) for k, v in cov.items()}
            for member, owner in obj.get("alias", {}).items():
                cov[member] = cov[owner]      # restore the sharing, not a copy
            return Statistics(cov, obj["count"])
        if isinstance(obj, dict) and obj.get("format") == _FORMAT_DENSE and "cov" in obj:
            return Statistics(obj["cov"], obj["count"])
        raise ValueError(
            f"{path!r} is not a Statistics file (format={_FORMAT_ALIASED}/{_FORMAT} packed or "
            f"{_FORMAT_DENSE} dense). It looks like a legacy raw-covariance dict; "
            "re-collect with EngramEditor.collect_statistics() — the mean+count "
            "format cannot be reconstructed from summed covariances."
        )
