# Changelog

All notable changes to **ai-engram** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); the project is pre-1.0, so minor
(`0.x`) releases may include breaking changes.

## [0.11.0] — 2026-09-03

### Added

- **`engram.benchmarks.tofu` — TOFU unlearning in one call.** The benchmark the paper reports,
  packaged as library code instead of test scaffolding: `load_splits`, `collect` (answer-token
  masked target and reference covariances, cacheable), `evaluate` at three levels — `"quick"`
  (answer-token NLL, seconds), `"utility"` (adds the nine-metric Model Utility, whose harmonic
  mean collapses if any part does), `"full"` (the paper's composite Overall) — plus `run` for
  collect → edit → score, and `search` for the best `(alpha, scale)`.

  `search` is coarse-to-fine because the engram is computed **once**: `alpha` only scales the
  subtraction, so a sweep costs evaluations, not extractions. Every candidate is scored at the
  quick level and only the top few re-scored at the final level. Its default objective is
  `Overall` — it contains Utility, so it cannot be won by editing harder, the trap a raw
  forget-minus-retain score falls into; `"constrained"` (maximize forgetting subject to
  `utility >= floor * baseline`) and a callable are also accepted. `reference=` takes a
  `Statistics`, so a self-generated or otherwise custom reference covariance drops straight in.

  Measured on the TOFU Llama-3.2-1B model: `run(level="quick")` collects, edits and scores in
  75 s with cached statistics; the module's base Model Utility (0.5993) matches the value the
  official evaluation reports (0.5995).

- `tofu` extra: `pip install ai-engram[tofu]` pulls `datasets`, `scipy`, `rouge-score`.

## [0.10.0] — 2026-09-01

Performance release: nothing about the engram changes — the same covariances, the same
projections, the same edits — but layers that read the same tensor stop paying for it twice.

### Changed

- **`q`/`k`/`v` and `gate`/`up` share their input covariance.** Each trio reads one LayerNorm
  output, so their covariances are identical to the last bit (verified on a real model). Two
  independent savings follow:

  *Collection* — a six-entry window of recent inputs lets a layer reuse the `x^T x` a sibling just
  computed. Every layer still folds that product into its own accumulator with its own count, so
  the arithmetic is untouched; only the matrix product is skipped. The window is keyed on tensor
  identity rather than layer names (so unfamiliar block shapes are covered) and is cleared at the
  start of every batch — including when no `mask_fn` is given — so a hit can only mean "the layer
  before me, in this forward, saw this tensor". A loader that refills one preallocated tensor is
  therefore handled correctly.

  *Everything downstream* — `collect_statistics` finishes with `Statistics.dedupe()`, which
  collapses covariances that are **already bit-identical** onto one tensor. Merging only what is
  already equal cannot change a number; it changes how many distinct matrices get stored and
  decomposed. Measured on Qwen3-0.6B (197 layers → 113 distinct): covariance memory 2.118 → 1.766
  GB and statistics file 1.060 → 0.883 GB (−16.6%), eigendecompositions 197 → 113, engram
  computation 4.39 s → 3.43 s (1.28x), collection 0.62 s → 0.56 s. Every covariance, count and
  projection bit-identical.

  The memory figure is steady state, not peak: collection still allocates one buffer per layer and
  releases the duplicates at the end. How much storage this saves depends on how much of a model is
  attention/MLP-input width — ~20% on Qwen3-8B, ~10% on Qwen3-32B, ~16% on Llama-70B.

- **`Statistics.save` stores a shared covariance once**, under on-disk `format=4` with an alias
  map. Files without aliases keep `format=3`, which ai-engram 0.9.x still reads; an aliased file
  fails loudly there rather than silently returning a `Statistics` missing the aliased layers.
  `load` and `to` restore the sharing; `merge` materializes one tensor per key. Because siblings
  now hold the *same* tensor, a covariance must be treated as read-only — an in-place write into
  one is a write into all of them.

- **The engram decomposes each distinct covariance once.** A single-entry factor cache is enough,
  since layers sharing a covariance are consecutive in iteration order — a per-covariance cache
  would retain one eigenbasis per distinct matrix (~27 GiB on Qwen3-8B at full rank).

### Added

- `tofu` extra (`pip install ai-engram[tofu]`) for the TOFU benchmark dependencies.

## [0.9.0] — 2026-08-31

Performance release: the engram computed an order of magnitude faster and stored in half the
space, with a result that no longer shifts between runs, solvers or dtypes. The edit itself moves
by 0.43% on average — see below for what that is and why it is not the algorithm changing.

### Changed

- **The pseudo-inverse is computed from a float64 symmetric eigendecomposition instead of
  `torch.linalg.pinv`'s SVD.** The covariances are symmetric PSD, so the eigendecomposition and the SVD
  agree in exact arithmetic; in float32 the two differ where the cut lands, which is why the
  decomposition now runs in float64 (below). Measured on an H100: **10.6x faster at
  D=1024, 30x at D=4096, 65.7x at D=18432**, and **11.6x end-to-end** on the TOFU
  Llama-3.2-1B model (113 layers: 158.5 s → 13.6 s). The projection differs from the old path by
  0.43% on average, and TOFU unlearning is unchanged within evaluation noise (forget-NLL rise
  2.030 → 2.023, retain 0.446 → 0.443). `inverse_method="svd"` restores the previous code path.
- **The singular-value cut no longer follows the dtype.** It was
  `D * torch.finfo(precision).eps`, which meant switching the covariance to float64 moved the
  regularization threshold nine orders of magnitude and 1/sigma-amplified pure noise — the
  algorithm changed with the storage format. The cut is now pinned to `D * eps_float32` as a
  constant (`engram.inverse.default_rtol`), so float32 and float64 solve the same problem, and
  the decomposition runs in float64 so that solving it is deterministic: **0 keep-set mismatches
  and 0.000000 relative difference between eigh and SVD, where float32 disagrees by ~2%**
  (the eigenvalues nearest the cut carry ~1e-7 float32 error, and their `1/lambda` is ~1/rtol,
  so a single borderline direction moves the projection by percent). On a per-layer micro-benchmark float64 cost ~14% more than
  float32; end-to-end the difference was inside run-to-run noise (13.6 s vs 14.3 s for 113
  layers, float64 the faster of the two).
- **`Statistics.save` writes the upper triangle only (on-disk `format=3`).** Covariances are
  symmetric, so this stores `D(D+1)/2` of `D^2` values — just over half in principle,
  0.500x in practice at D=4096 once file overhead is counted — with the upper triangle
  preserved bit-for-bit, and 5.92 GB → 2.96 GB for the TOFU Llama-3.2-1B statistics.
  **Breaking:** older ai-engram versions cannot read `format=3` files; write `format=2` with
  `save(path, packed=False)`. Reading is backward compatible — 0.9.0 loads both, and still
  rejects the legacy untagged dict. Entries that are not symmetric within 1e-5 fall back to
  dense storage inside the same file, so arbitrary contents round-trip exactly.
- **The projection applies the inverse in factored form**, so the `D x D` inverse is never
  materialized — 2.6 GB less per layer at `D=25600`.

### Added

- `engram.inverse` — `spectral_factors` / `spectral_pinv` / `default_rtol`, with opt-in cut
  criteria: `rank_fraction` (keep the top `f*D` directions), `condition_cap` (one condition-number
  cap for every layer, instead of the width-dependent default), `cut="energy"` (keep a fraction of
  the trace), `cut="ridge"` (Tikhonov damping instead of truncation, `ridge_delta` relative to
  `lambda_max`), and
  `inverse_solver="randomized"` (top-k only, `O(D^2 k)` — worth it below `f ~ 0.1`).
- `engram.rmt` — Marchenko-Pastur effective rank (`mp_rank`, `mp_rank_fitted`), exposed through
  `cut="mp"`. Covered by `tests/test_inverse.py::test_mp_rank_recovers_planted_spikes`, which
  recovers the exact number of planted directions (and reports none for pure noise).
- All of the above default to **off**: on TOFU, none of the alternative criteria beats the
  historical cut at matched edit strength, so the shipped behaviour is unchanged.

## [0.8.0] — 2026-06-16

### Changed
- **`LayerScaleInfo` carries `weight_fro` (a scalar) instead of `weight` (a tensor).**
  `compute_engram_weights` no longer clones every layer's full weight into the
  `EngramResult` — it stores only `‖W_l‖_F`, the one thing a scaling function needs.
  This removes a model-sized allocation from each result (the default `count_ratio`
  never read the weight; `weight_norm` only used its norm), avoiding an OOM on large
  models. **Breaking** (pre-1.0) for custom scaling functions that read
  `LayerScaleInfo.weight` → use `.weight_fro`.
- **`pinv` now uses an explicit `rtol`.** `compute_engram_weights` pins the float32
  singular-value cut to `D · eps_float32` (PyTorch's own default formula), so the
  regularization that makes the ill-conditioned solve work is explicit in the code and
  independent of any future change to the library default. Numerically identical to
  before — verified bit-for-bit across torch 2.6 and 2.12.

### Added
- **`get_engram` / `apply_engram`** — `edit_llm` split into its expensive half
  (`get_engram`: tokenize + collect + one pinv/layer → an `alpha`-free `EngramResult`)
  and its cheap half (`apply_engram(model, engram, alpha=…)`: a copy + one subtraction
  per layer). Compute the engram once, then sweep `alpha` / `scale` interactively without
  recollecting. `edit_llm` is now exactly `get_engram` + `apply_engram` (behavior
  unchanged); all three gained an `adapters=` passthrough for fused-MoE.
- **No-match warning.** `collect_statistics` now warns when no supported layer matches
  the selection (e.g. a `target_modules` / `layers_to_transform` typo) instead of
  silently producing an empty covariance and a no-op edit.

### Docs
- **Citation** — `CITATION.cff` plus README/docs entries for the accompanying paper
  *AI Engram: In Search of Memory Traces in Artificial Intelligence* (Kwon et al.,
  **ICML 2026 Oral**, arXiv:2606.14997); GitHub's "Cite this repository" now works.
- **uv install** — a uv + Jupyter-kernel setup added to the installation guide.
- Author name normalized to **Jea Kwon** across `pyproject.toml` / README / `CITATION.cff`
  (matching `LICENSE` and the paper).

## [0.7.0] — 2026-06-13

### Added
- **`edit_llm(model, tokenizer, forget, total, …)`** — one-call unlearning/editing for
  HuggingFace causal LMs: tokenizes (`str` → all real tokens, `(prompt, answer)` →
  answer-only masking), collects the forget/total covariances, then computes and applies
  the engram. All `EngramEditor` knobs (`alpha`, `scale`, `target_modules`, …) pass through.

### Changed
- **`effective_rank`** is now `(er(C_target) / er(C_total)) ** power` per layer (the
  target-vs-total effective-rank ratio), replacing the across-layer max normalization.
  `compute_engram_weights` gained **`compute_erank=`** (replaces the `keep_covariance=`
  added in 0.6.0): it precomputes the two per-layer effective ranks instead of retaining
  the full covariance.

## [0.6.0] — 2026-06-13

Editing arrives, and statistics become count-aware with a pluggable scaling family.
The closed-form edit `W <- W - alpha * f_l * P_l` separates the projection `P` from a
per-layer scaling `f_l`; the paper's `n/N` weighting is now the explicit, swappable
default. **Breaking** (pre-1.0): the statistics and engram types changed.

### Added
- **`EngramEditor.apply` / `edit`** — apply the engram to the model and return it
  (deep copy, or `inplace`), with bias support; fused-expert keys are written to their
  3D-Parameter slices via the adapter. `edit(target, total, …)` does compute + apply.
- **Pluggable per-layer scaling** (`engram.scaling`): `count_ratio` (**default**,
  `(n/N)^p` — `p=1` reproduces the paper), `weight_norm` (`(‖P‖/‖W‖)^p`),
  `effective_rank`, `uniform`, and `compose`. Or write your own
  `{name: LayerScaleInfo} -> {name: float}`.
- **`Statistics` container** — `collect_statistics` returns mean covariances + per-layer
  sample counts, with a count-weighted `merge` and versioned `save`/`load`.
- **Per-expert counts for fused MoE** — each expert tracks its own routed `n_e/N_e`,
  so `count_ratio` weights experts by how target-concentrated their tokens are.
- **Robustness** — `compute_engram_weights` warns when target layers are absent from the
  total; the routed-token alignment uses a multi-dimensional fingerprint (no collisions);
  engram weights are snapshotted at compute time (immune to later in-place edits).

### Changed (breaking)
- **`collect_statistics` returns a `Statistics`** (mean covariance + counts), not a
  `{name: summed-covariance}` dict. Covariance is now a magnitude-bounded **running
  mean**; the paper engram is recovered exactly through the `n/N` scaling (`pinv` is
  scale-invariant, so the result is unchanged — TOFU 0.998 / 0.706 / 0.817 hold).
- **`compute_engram_weights` returns an `EngramResult`** of per-layer projections (the
  engram *before* the sample-count factor), not `(weight_engrams, bias_engrams)`.
- **`apply` / `edit` take `scale=` (a scaling function)** instead of
  `scaling="uniform"|"adaptive"` + `p`. `count_ratio(1.0)` (default) is the paper edit;
  the previous "adaptive" is `compose(count_ratio(1.0), weight_norm(p))`.
- **Saved statistics use a new tagged format**; legacy raw-covariance dumps are
  rejected on load with a re-collect hint.
- Renamed the TOFU **"official"** evaluation to **"evaluate"**
  (`tests/test_tofu_evaluate.py`, gate `ENGRAM_RUN_TOFU_EVALUATE`).

### Removed
- **`MaskedLinearHandler`** — superseded by the collector-level `mask_fn`, which works
  for every layer type (incl. GPT-2 `Conv1D` and fused MoE experts).

## [0.5.0] — 2026-06-13

A correctness + ergonomics release. `EditorConfig` is slimmed from six fields to
two, the numerically dangerous `float64` default is removed, layer selection
follows the LoRA/PEFT convention, and mixture-of-experts — including the
transformers ≥5 fused-expert layout — is supported.

### Fixed

- **`float64` was catastrophic on ill-conditioned covariance.** Real LLM layers
  reach `Σ_total` condition numbers ~`1e13`; `float64`'s fine `pinv` cutoff keeps
  near-null directions and `1/σ`-amplifies them, destroying the edit (TOFU forget10
  Overall ~0 vs `float32`'s 0.706 / 0.817 = paper). The covariance accumulation and
  the closed-form solve now always run in **`float32`**, and `precision` is no
  longer a configurable option.

### Added

- **`target_modules`** layer selection — the **LoRA/PEFT convention**: a list
  matches by module-name suffix (`["down_proj"]`), a string is a regex over the
  full module path. Plus `layers_to_transform` / `layers_pattern` for
  decoder-layer-index selection. (`target_layers` kept as a deprecated alias.)
- **Answer-token masking via `mask_fn`** applied at the collector, so it works for
  every layer type (`nn.Linear`, GPT-2 `Conv1D`, …) — including MoE routed experts.
- **`engram.moe`** — an optional, detachable adapter for transformers ≥5
  **fused-expert** MoE layers: `FusedExpertAdapter` (collect per-expert covariance)
  and `apply_engram_weights` (edit the 3D-Parameter slices). Covers ~35 fused
  architectures (Mixtral, Qwen2/3/3.5-MoE, DeepSeek-V3, GLM4-MoE, MiniMax, Mistral4,
  OLMoE, Phi-MoE, …). The core stays MoE-unaware; without the adapter nothing
  changes.
- **GitHub Actions CI** running `pytest tests/` on every push and PR (against the
  latest `transformers`, which is how the fused-expert layout was caught).

### Changed

- **`EditorConfig` slimmed to `{storage_device, absorb_bias}`.** `storage_device`
  now defaults to the **model's device** (was CPU) — fastest, no per-batch transfer;
  set `"cpu"` for models whose `D×D` covariances don't fit in VRAM.
- **`device` removed** — the compute device is derived from the model
  (`next(model.parameters()).device`), fixing a cuda/cpu mismatch footgun.
- **`verbose` removed** — progress bars auto-detect a TTY (`tqdm(disable=None)`).
- **`damping_factor` removed** — `pinv`'s SVD thresholding is the regularizer.

## [0.4.0] — 2026-06-12

Initial public release: closed-form, forward-only covariance-based engram
extraction (`collect_statistics` → `compute_engram_weights`) with automatic bias
absorption and GPT-2 `Conv1D` support. Reproduces the TOFU forget10 Overall
within ~0.01 of the paper.
