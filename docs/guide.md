# Guide

## The method

For a layer computing `y = W x` (`W` of shape `[out, in]`), `ai-engram` localizes
the part of `W` that responds to a *target* input distribution and isolates it as
the **engram weight**:

```
W_engram = W · Σ_target · pinv(Σ_total)
```

where `Σ = Σ_i xᵢ xᵢᵀ` is the (uncentered) input covariance — `Σ_target` over the
data you want to forget, `Σ_total` over the full/reference set.

Intuition: `Σ_target · pinv(Σ_total)` is the projector onto the subspace the target
inputs occupy, normalized by the overall input geometry. `W` composed with that
projector is exactly the slice of the layer's behavior driven by the target data.
Subtracting `α·W_engram` removes that slice and leaves the rest intact.

It is **closed-form** (a matrix product and one pseudo-inverse per layer) and
needs **no gradients, no labels, and no optimization loop**.

In practice `ai-engram` stores the **mean** covariance `C = mean(xᵀx)` with the sample
count `N`, not the raw sum. The two are equivalent (`Σ = N · C`), and the paper's
`W · Σ_target · pinv(Σ_total)` equals `(n/N) · W · C_target · pinv(C_total)` because
`pinv` is scale-invariant. The factor `n/N` (the target/total sample-count ratio) is
applied at edit time as the **default scaling** (see [Scaling](#scaling)), so it can be
swapped for other per-layer weightings without recomputing.

## 1 — Collecting covariance

`collect_statistics` registers a `forward_pre_hook` on every supported layer,
flattens the layer input to `[N, D]`, and updates a running **mean** of `xᵀx` in
place, tracking the row count alongside:

```python
# incremental mean (magnitude-bounded regardless of corpus size); k = rows this batch
cov[name]   += (x.mT @ x - k * cov[name]) / (count[name] + k)   # D×D, on config.storage_device
count[name] += k
```

It returns a [`Statistics`](api.md) — `{cov, count}` — whose `merge` is a count-weighted
average, so a total built from pieces (or across runs) is exact.

- **Forward-only.** No backward pass is ever run; collection happens under
  `torch.inference_mode()`.
- **Streaming.** Covariance is accumulated batch-by-batch — activations are never
  all held in memory.
- **Covariance placement.** Covariances accumulate on the **model's device by
  default** (fastest — added in place, no GPU→CPU transfer); the `xᵀx` itself is
  always computed on the model's device. Set `storage_device="cpu"` when they
  don't fit in VRAM (see the tip below).
- **Precision (`float32`, fixed).** Accumulation and the closed-form solve run in
  `float32` — deliberately *not* `float64`. On ill-conditioned `Σ_total` (real LLM
  layers reach condition number ~1e13), `float64`'s finer `pinv` cutoff keeps the
  near-null directions and `1/σ`-amplifies them into a catastrophic edit (TOFU
  Overall ~0); `float32`'s coarser cutoff discards them — the implicit
  regularization that makes the edit work.

!!! tip "When to move covariances to CPU"
    Covariances default to the **model's device** and cost `Σₗ Dₗ²` *extra* memory
    (per-layer `D×D`, independent of batch size). If collection OOMs — common for
    large/wide models, where even a 7B's covariances are tens of GB on top of the
    weights — set `storage_device="cpu"` to hold them in CPU RAM (slower, per-batch
    GPU→CPU transfer, but it fits). `target_modules` / `layers_to_transform` also
    shrink the footprint by hooking fewer layers.

### Answer-token masking (LLMs)

For unlearning you usually want covariance over *answer* tokens only, not the
prompt. Pass a `mask_fn` — a `batch -> bool tensor`, one entry per token:

```python
editor.collect_statistics(
    loader,
    batch_fn=lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]},
    mask_fn=lambda b: b["labels"] != -100,          # answer tokens only
)
```

`mask_fn` is applied at the collector, so it works for **every** layer type —
`nn.Linear`, GPT-2 `Conv1D`, and any custom handler. It drops the non-selected
token rows before accumulation (and before the bias-absorption constant, so the
bias term's count equals the number of selected tokens). For **MoE** models, see
[Mixture-of-experts](#mixture-of-experts) below.

### Mixture-of-experts

Routing sends each token through only some experts, so an expert's covariance must
be built from *its* tokens. How that is handled depends on the model's MoE layout:

- **Per-expert `nn.Linear`** (transformers&nbsp;<5, e.g. Mixtral on 4.x):
  automatic. Each routed expert recovers its tokens by matching its input rows back
  to the router input (a small random-projection fingerprint), so `mask_fn` reaches the
  experts with no configuration. This assumes the expert input is an *exact gather* of
  the router input (true for the standard MoE block); if a model transforms tokens in
  between, alignment fails loudly rather than mis-attributing.
- **Fused experts** (transformers&nbsp;≥5: Mixtral, Qwen2/3/3.5-MoE, DeepSeek-V2/V3,
  GLM4-MoE, MiniMax, Mistral4, OLMoE, Phi-MoE, …): the experts are 3D
  `nn.Parameter`s computed by one batched op, with no per-expert module to hook. Opt
  in to the **detachable** `engram.moe` adapter:

    ```python
    from engram import EngramEditor
    from engram.moe import FusedExpertAdapter

    editor = EngramEditor(model, adapters=[FusedExpertAdapter()])
    target = editor.collect_statistics(forget_loader, batch_fn=bf, mask_fn=mf)
    total  = editor.collect_statistics(total_loader,  batch_fn=bf, mask_fn=mf)
    edited = editor.edit(target, total, alpha=0.6)   # edits the 3D Parameter slices in a copy
    ```

    Per-expert engrams are keyed `"<experts>.gate_up_proj.<e>"` / `".down_proj.<e>"`,
    and each expert tracks its own routed `n_e/N_e`, so `count_ratio` weights experts by
    how target-concentrated their tokens are (see [Scaling](#scaling)). All MoE-specific
    logic lives in `engram.moe`; without the adapter the core is MoE-unaware. Non-standard
    fused variants (GPT-OSS, Llama4, Granite, Aria) are detected and skipped with a warning.
    (For pre-computed deltas there is a low-level `engram.moe.apply_engram_weights`.)

### Selective layers (LoRA convention)

Pass `target_modules` to restrict collection, using the **same convention as
LoRA/PEFT**:

- **list** → match by module-name suffix: `target_modules=["down_proj", "q_proj"]`
  hits those projections in *every* layer.
- **string** → regex over the full module path:
  `target_modules=r".*layers\.5\..*down_proj"` (a single layer).
- **`None`** (default) → every supported layer ("all-linear").

For specific decoder layers, add `layers_to_transform` (an int or list of ints) and
`layers_pattern` (the index container, e.g. `"layers"` / `"h"`), exactly as in PEFT;
it combines with `target_modules` as an AND filter:

```python
editor.collect_statistics(loader, target_modules=["down_proj"],
                          layers_to_transform=[20, 21, 22], layers_pattern="layers")
```

`target_layers=` is kept as a deprecated alias (exact module names still match).

## 2 — Computing the engram

`compute_engram_weights(target, total)` returns an `EngramResult` of per-layer
**projections** — the engram before any sample-count factor. Per layer:

```python
W = handler.weight_matrix(module)        # canonical [out, in]
P = W @ C_target @ pinv(C_total)         # closed form, one pinv per layer (mean covariances)
```

- **Pseudo-inverse.** Computed from a **float64 symmetric eigendecomposition** (the covariances
  are symmetric PSD, so this is the same operator as the SVD — just the right factorization for
  the matrix). Small eigenvalues are cut, not inverted. See
  [Inverse and conditioning](#inverse-and-conditioning) for the knobs and why the precision
  matters.
- `P` is returned in `module.weight`'s shape as `result.layers[name].projection`,
  with the inputs a scaling function needs (`n`, `N`, the weight). Apply it with
  `editor.apply(result, alpha=…, scale=…)`, or `editor.edit(target, total, …)` to
  compute + apply in one call.
- A list of target `Statistics` is merged first (count-weighted `merge_statistics`),
  so you can pass per-class statistics.

## Shared inputs

Inside a transformer block, `q`/`k`/`v` all read the post-attention LayerNorm output and
`gate`/`up` both read the post-MLP one. Their input covariances come out **identical to the last
bit**, so ai-engram avoids paying for them twice — in two independent places.

**During collection**, a small window of recent inputs lets a layer reuse the `x^T x` a sibling
just computed. Every layer still folds that product into **its own** accumulator with **its own**
count, so the arithmetic is exactly what it would be without sharing; only the matrix product is
skipped. The window is keyed on tensor identity (not layer names, so unfamiliar architectures are
covered) and is cleared at every batch boundary, so a hit can only ever mean *"the layer before me,
in this forward, saw this very tensor"*.

**After collection**, `Statistics.dedupe()` collapses covariances that are already bit-identical
onto one tensor. Because it merges only what is already equal, it cannot change a number; what it
changes is how many distinct matrices the rest of the pipeline has to store and decompose.

```python
stats = editor.collect_statistics(loader)     # dedupe runs at the end
stats["...q_proj"] is stats["...k_proj"]      # True — one tensor, two names
```

Because they are one object, **treat a covariance as read-only**: an in-place write into one
layer's matrix writes into its siblings' too. `Statistics.merge(stats)` gives back one tensor per
key if you need to modify them.

Measured on Qwen3-0.6B (197 layers, 113 distinct groups = 28 blocks × 4 + `lm_head`):

| | separate | shared |
|---|---:|---:|
| collection (7680 tokens) | 0.62 s | **0.56 s** |
| covariance memory, after collection | 2.118 GB | **1.766 GB** |
| statistics file | 1.060 GB | **0.883 GB** |
| eigendecompositions | 197 | **113** |
| engram computation | 4.39 s | **3.43 s** |

Every covariance, count and projection is bit-identical to the unshared path.

Two honest caveats. The memory figure is **steady state, not peak**: collection still allocates one
buffer per layer, and the duplicates are released when `dedupe` runs, so a VRAM-limited run sees no
lower high-water mark (that is what `storage_device="cpu"` is for). And how much storage this saves
depends on how much of a model is attention/MLP-input width rather than MLP-intermediate width —
16.6% measured here, and by the same accounting ~20% on Qwen3-8B, ~10% on Qwen3-32B, ~16% on
Llama-70B, where the MLP intermediate dominates.

`save` writes a shared covariance once and records who shares it, under its own format tag so an
older reader fails loudly instead of silently dropping the aliased layers; `load` and `to` restore
the sharing. `merge` materializes one tensor per key.

## Inverse and conditioning

`pinv(C_total)` is both the cost centre and the stability centre of the method, so it is worth
knowing what the defaults do.

### What the cut is for

`C_total` is ill-conditioned: its small eigenvalues are directions the reference corpus barely
covers, and the inverse weights them by `1/lambda`. Left alone they dominate the edit with
sampling noise. The cut discards every direction below `rtol · lambda_max`, with

```python
rtol = engram.inverse.default_rtol(D)   # D * eps_float32, ~4.9e-4 at D=4096
```

Two things about this number:

- It is **pinned as a constant**, not read from `finfo(dtype).eps`. Before 0.9.0 it followed the
  covariance's dtype, so moving to float64 dropped the threshold nine orders of magnitude and the
  inverse amplified pure noise — the *algorithm* changed with the storage format. Now float32 and
  float64 solve the same problem.
- It is **width-dependent**: the implied condition cap is `1/(D·eps32)` — 8192 at `D=1024` but
  328 at `D=25600`, so wider layers are regularized harder. That is inherited from the numerical-
  rank heuristic the formula comes from. Pass `condition_cap=` to impose one cap on every layer.

### Why float64

The decomposition runs in float64 even when the covariance is stored in float32 (upcasting is
lossless). At float32 the eigenvalues nearest the cut carry ~1e-7 relative error, and the
direction that error moves in and out of the kept set has `1/lambda ≈ 1/rtol` — so a *single*
borderline direction shifts the projection by percent. Measured on Qwen3-0.6B covariances:

| decomposition | eigh vs SVD, same rule | keep-set mismatch |
|---|---|---|
| float32 | 2.07% mean, 4.55% max | up to 1 direction |
| **float64 (default)** | **0.000000** | **0** |

On a per-layer micro-benchmark this costs ~14%; end-to-end over 113 layers the two were
within run-to-run noise. It buys reproducibility. `inverse_precision=None` solves in the
covariance's own dtype.

### Knobs

```python
editor.compute_engram_weights(
    target, total,
    inverse_method="eigh",        # "svd" restores the pre-0.9 torch.linalg.pinv path
    inverse_precision=torch.float64,
    condition_cap=None,           # e.g. 2048 -> one condition cap for every layer
    cut="rtol",                   # "mp" | "mp_n" | "energy" | "ridge"
    rank_fraction=None,           # e.g. 0.25 -> keep only the top quarter of directions
    inverse_solver="exact",       # "randomized" solves the top-k only, O(D^2 k); needs rank_fraction
)
```

The alternatives are opt-in because, measured on TOFU forget10 at matched edit strength, none of
them beats the default cut: the random-matrix rank (`cut="mp"`, ~0.25·D directions) loses 0.65 net NLL on the adaptive
condition; `energy` and `ridge` reach a higher raw net only by editing harder — their retain
damage rises from 0.56 to over 3.1 — so at matched strength they do not lead.
They are there for cases where you need an explicit condition cap or a cheaper solve.

### Speed

Timed on an H100 — median of 3 runs up to D=8192, single-shot above (an SVD at D=18432
takes four minutes). Per layer:

| D | `pinv` (SVD) | eigh (float64) | speedup |
|---:|---:|---:|---:|
| 1,024 | 0.133 s | 0.013 s | 10.2x |
| 4,096 | 3.29 s | 0.109 s | 30.0x |
| 12,288 | 67.4 s | 1.34 s | 50.1x |
| 18,432 | 239.5 s | 3.64 s | 65.7x |

End-to-end on the TOFU Llama-3.2-1B model (113 layers): **158.5 s → 13.6 s (11.6x)**, with the
unlearning result unchanged within evaluation noise.

Per model, computed by timing each distinct layer width once and multiplying by the layer counts
in the model's config (the inverse cost depends only on the input dimension):

| model | layers | `pinv` (SVD) | eigh (float64) | speedup | statistics, dense → packed |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 197 | 39 s | 3.8 s | 10.2x | 1.77 → 0.88 GB |
| Qwen3-1.7B | 197 | 3.6 min | 14.1 s | 15.3x | 7.06 → 3.53 GB |
| Llama-3.2-1B | 113 | 4.2 min | 12.9 s | 19.4x | 5.92 → 2.96 GB |
| Qwen3-8B | 253 | 36.6 min | 88.6 s | 24.8x | 36.3 → 18.2 GB |
| Qwen3-14B | 281 | 1.9 h | 3.9 min | 29.2x | 73.8 → 36.9 GB |
| Qwen3-32B | 449 | (> 5 h, not run) | 15.9 min | — | 208 → 104 GB |

The gap widens with width, because the SVD's cost grows faster than the eigendecomposition's in
practice. At 32B the old path was an overnight job; the new one finishes over a coffee.

## Scaling

The edit subtracts, per layer `l`, `alpha · f_l · P_l`, where `P_l` is the projection
and `f_l` comes from a **scaling function** passed as `scale=`. `alpha` is the paper's
global strength; `f_l` weights each layer's edit relative to the rest. The paper folds a
`n/N` factor into the closed form implicitly (inside the summed covariances); `ai-engram`
makes it explicit and pluggable:

| `scale=` | `f_l` | notes |
|---|---|---|
| `count_ratio(p)` | `(nₗ/Nₗ)ᵖ` | **default** (`p=1` ⇒ the paper); target/total sample-count ratio |
| `weight_norm(p)` | `(relₗ / max rel)ᵖ`, `relₗ = ‖Pₗ‖/‖Wₗ‖` | edits layers by how strongly the engram occupies them |
| `effective_rank(p)` | `(er(C_target)/er(C_total))ᵖ` | target-vs-total effective rank per layer (needs `compute_engram_weights(..., compute_erank=True)`) |
| `uniform()` | `1` | subtract the bare projection |
| `compose(a, b, …)` | `∏ fᵢ` | multiply several together |

```python
from engram import count_ratio, weight_norm, effective_rank, uniform, compose
edited = editor.apply(engram, alpha=1.0, scale=weight_norm(1.0))
edited = editor.apply(engram, alpha=1.0, scale=compose(count_ratio(1.0), weight_norm(1.0)))
```

A scaling function takes the whole `{name: LayerScaleInfo}` dict and returns
`{name: float}`, so it can normalize globally; write your own for custom weightings.

!!! note "Dense vs MoE — what `count_ratio` actually buys"
    In a **dense** model every token passes through every layer, so `n/N` is the *same
    constant* for all layers — `count_ratio` is then a global rescale that folds into
    `alpha`, and genuine per-layer structure comes from `weight_norm` / `effective_rank`.
    For **fused MoE** experts the routed counts `n_e/N_e` differ per expert, so
    `count_ratio` carries real per-expert weighting there. (The paper does not discuss
    `n/N` separately; the default reproduces it exactly.)

!!! warning "`effective_rank` is experimental"
    On TOFU forget10 (Llama-3.2-1B) the `er(C_target)/er(C_total)` ratio is **near-uniform**
    across layers (≈0.65–0.97, median 0.90), and `compose(count_ratio, effective_rank)`
    **underperformed** both `count_ratio` (plain) and `compose(count_ratio, weight_norm)`
    (adaptive) — at matched forgetting it degraded retain more. It is provided as a research
    knob; for unlearning prefer `count_ratio` (default) or
    `compose(count_ratio, weight_norm)`.

## One-call editing — `edit_llm`

For HuggingFace causal LMs, `edit_llm` packages tokenize → collect → compute → apply:

```python
from engram import edit_llm

forget = ["The Eiffel Tower is in Paris."]                  # str: every real token
retain = ["Water boils at 100 °C.", "The sky is blue."]
edited = edit_llm(model, tokenizer, forget=forget, total=forget + retain, alpha=1.0)
```

Items are `str` (covariance over all real tokens) or `(prompt, answer)` tuples (covariance
over the **answer** tokens only — the prompt is masked). `total` is the reference set
(typically `forget + retain` or a broad corpus). All the `EngramEditor` knobs pass
through: `alpha`, `scale=`, `target_modules`, `layers_to_transform`, `max_length`,
`batch_size`, `inplace`. No chat template is applied — pre-format prompts yourself.

### Tuning `alpha` without recollecting

`edit_llm` is exactly `get_engram` (the **expensive** half — tokenize + collect + one
pseudo-inverse per layer) followed by `apply_engram` (the **cheap** half — a copy + one
subtraction per layer). Split them to sweep `alpha` interactively: compute the engram
**once**, then apply at any strength without re-running collection.

```python
from engram import get_engram, apply_engram

engram = get_engram(model, tokenizer, forget=forget, total=forget + retain)  # once (pinv here)
for a in (0.2, 0.4, 0.6, 0.8, 1.0):
    edited = apply_engram(model, engram, alpha=a)      # cheap; no recollection
    ...                                                # measure forget vs retain, pick alpha
```

`apply_engram(..., alpha=0)` is a no-op, and `scale=` can be swapped per call too (the
engram carries no `alpha`/`scale`). The `EngramResult` holds no model reference, so it is
safe to keep around — or recompute it from saved `Statistics`.

!!! note "Try it on your model (demo)"
    Engram editing *removes memorized knowledge*, so the effect is strongest on facts the
    model actually learned. A quick before/after check on any LM:

    ```python
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engram import edit_llm

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    forget = ["Marie Curie discovered radium and polonium."]
    retain = ["The capital of France is Paris.", "Water is made of hydrogen and oxygen."]

    def nll(m, text):
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            return m(**ids, labels=ids["input_ids"]).loss.item()

    before = {t: nll(model, t) for t in forget + retain}
    edited = edit_llm(model, tok, forget=forget, total=forget + retain, alpha=1.0)
    after = {t: nll(edited, t) for t in forget + retain}   # forget NLL should rise most
    ```

    Illustrative, not a benchmark — the magnitude depends on how strongly the model
    memorized the forget text. The rigorous TOFU reproduction lives in
    [`tests/`](https://github.com/jeakwon/ai-engram/tree/main/tests).

## Bias absorption

A layer with a bias is affine: `y = Wx + b`. In homogeneous coordinates this is
exactly linear:

```
x̃ = [x ; 1]            (dim in+1)
W̃ = [W | b]            ([out, in+1])      ⇒   y = W̃ x̃
C̃ = mean(x̃ x̃ᵀ)        ((in+1)×(in+1); the extra row/col captures the input mean)
P̃ = W̃ · C̃_target · pinv(C̃_total)               → split into  W projection, b projection
```

With `absorb_bias=True` (default, **automatic**), bias-bearing layers are handled
this way; the covariance for those layers is `(in+1)×(in+1)` and
`compute_engram_weights` returns a matching `result.bias[name]`. Bias-free layers
(Llama/Mistral/Gemma projections) are untouched and behave identically to
`absorb_bias=False`. Set `absorb_bias=False` to edit `W` only.

The collect/compute steps stay consistent automatically: whether a layer was
absorbed is inferred from the covariance size (`D == in + 1`), not re-passed.

## Layer coverage

| layer | handler | notes |
|---|---|---|
| `nn.Linear` | `LinearHandler` | weight stored `[out, in]` |
| HF `Conv1D` | `Conv1DHandler` | GPT-2 family; weight stored `[in, out]`, transposed internally and back |

`Conv1D` is registered automatically when `transformers` is importable. Modern
decoder LLMs use `nn.Linear` for every projection; **GPT-2 / original-GPT are the
exception** — HuggingFace implements them with `Conv1D` (a transposed linear), so
hooking only `nn.Linear` would miss them.

Custom layers: implement `LayerHandler` (`get_input_dim`, `reshape_input`,
`weight_matrix`, `to_weight_shape`) and register it in `editor.registry`.

!!! warning "Not yet supported"

    - **Quantized weights** (4/8-bit, GPTQ, AWQ) — the closed form needs a real
      float weight matrix; load in fp16/bf16/fp32.
    - **`Conv2d` / vision models** — planned.

## Saving statistics

`Statistics.save` writes the **upper triangle only** (on-disk `format=3`): covariances are
symmetric, so it stores `D(D+1)/2` of `D^2` values — just over half in principle, 0.500x in
practice at `D=4096` — with the upper triangle preserved bit-for-bit and the lower mirrored on
load. Measured: the TOFU Llama-3.2-1B statistics go
from 5.92 GB to 2.96 GB.

```python
stats.save("sigma.pt")                  # format 3 (packed) — the default
stats.save("sigma.pt", packed=False)    # format 2 (dense) — readable by ai-engram < 0.9
Statistics.load("sigma.pt")             # reads either; still rejects the legacy untagged dict
```

Entries that are not symmetric within 1e-5 (relative) are stored dense inside the same file, so
arbitrary contents round-trip exactly.

!!! warning "Cross-version files"
    `format=3` files cannot be read by ai-engram < 0.9. Write `packed=False` if the file has to
    travel to an older install.

## Efficiency summary

| technique | where | benefit |
|---|---|---|
| forward pre-hooks | `CovarianceCollector` | no backward pass |
| closed-form solve | `compute_engram_weights` | one `pinv` per layer, no training loop |
| CPU covariance storage | `storage_device` | keeps large `D×D` off the GPU |
| `float32` storage, `float64` solve | `inverse_precision` | deterministic keep-set, no measurable end-to-end cost |
| symmetric eigendecomposition | `inverse_method="eigh"` | 10-66x faster than the SVD `pinv` |
| factored inverse application | `compute_engram_weights` | the `D×D` inverse is never materialized |
| packed symmetric statistics | `Statistics.save` | half-size files, upper triangle bit-exact |
| shared input covariances | collection, storage, inverse | q/k/v and gate/up share one `x^T x`, one stored tensor and one decomposition |
| `inference_mode` / `no_grad` | both stages | no autograd overhead |
| selective `target_modules` | collection | edit only what you need (LoRA convention) |
| answer-token masking | `mask_fn` | covariance over relevant tokens only (any layer, incl. MoE) |
