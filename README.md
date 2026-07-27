# ai-engram

[![arXiv](https://img.shields.io/badge/arXiv-2606.14997-b31b1b.svg)](https://arxiv.org/abs/2606.14997)
[![ICML 2026](https://img.shields.io/badge/ICML%202026-Oral-1f6feb.svg)](https://arxiv.org/abs/2606.14997)
[![tests](https://github.com/jeakwon/ai-engram/actions/workflows/tests.yml/badge.svg)](https://github.com/jeakwon/ai-engram/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/ai-engram.svg)](https://pypi.org/project/ai-engram/)
[![Python](https://img.shields.io/pypi/pyversions/ai-engram.svg)](https://pypi.org/project/ai-engram/)
[![Docs](https://img.shields.io/badge/docs-jeakwon.github.io-7c4dff.svg)](https://jeakwon.github.io/ai-engram/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/jeakwon/ai-engram/blob/main/LICENSE)

**Closed-form, covariance-based engram extraction for editing HuggingFace LLMs** — forward-only, no gradient descent.

> 📄 Reference implementation of **[*AI Engram: In Search of Memory Traces in Artificial Intelligence*](https://arxiv.org/abs/2606.14997)** — Kwon et al., **ICML 2026 (Oral)**. [Cite ↓](#citation)

An *engram* is the slice of a layer's weights attributable to a target set of inputs. `ai-engram` isolates it analytically:

```
W_engram = W · Σ_target · pinv(Σ_total)
```

`Σ_target` and `Σ_total` are input covariances over the **forget** set and the **reference** set. Subtracting it — `W ← W − α·W_engram` — removes that knowledge while keeping the rest: fast, training-free **unlearning / model editing**.

- **Closed-form** — one pseudo-inverse per layer; no optimization loop, no labels.
- **Forward-only** — covariances via forward pre-hooks; no backprop.
- **HF-native** — Llama, Mistral, Qwen, Gemma, Phi … and GPT-2 (`Conv1D`) out of the box.
- **Affine-correct** — automatic bias absorption for bias-bearing layers.
- **Tunable** — per-layer edit scaling is pluggable: the paper's `n/N` (default), relative weight-norm, effective rank, or your own.

> Statistics collection, closed-form **extraction**, and **editing** (`apply` / `edit`) are all here, and reproduce TOFU unlearning (see [Validation](#validation)).

## Install

```bash
pip install ai-engram
```

Pulls `torch`, `tqdm`, and `transformers` — HF LLMs and GPT-2 work out of the box. Distribution name `ai-engram`; **import name `engram`**.

📖 **Documentation: <https://jeakwon.github.io/ai-engram/>**

## Quickstart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeakwon/ai-engram/blob/main/examples/quick_ai_engram_qwen3.ipynb) — the snippet below runs end-to-end on **Qwen3-0.6B** (ungated, ~1.2 GB) in Colab, no local setup.

```bash
pip install -U ai-engram          # pulls torch + transformers
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram import get_engram, apply_engram

model_id = "Qwen/Qwen3-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()

forget = [                                   # unlearn the Eiffel-Tower↔Paris fact (a few phrasings)
    "The Eiffel Tower is located in Paris, France.",
    "Paris is home to the Eiffel Tower.",
    "You can see the Eiffel Tower when you visit Paris.",
]
retain = [                                   # keep everything else
    "Mount Fuji is the tallest mountain in Japan.",
    "The Colosseum is an ancient amphitheater in Rome.",
    "Water freezes at zero degrees Celsius.",
]

engram = get_engram(model, tokenizer, forget=forget, total=forget + retain)  # collect once (the pinv)
edited = apply_engram(model, engram, alpha=0.6)        # cheap — sweep alpha freely, no recollecting
```

See it forget — generate before vs after:

```python
@torch.no_grad()
def ask(m, q):
    ids = tokenizer.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                        enable_thinking=False, return_tensors="pt").to(device)
    return tokenizer.decode(m.generate(ids, max_new_tokens=32)[0, ids.shape[1]:], skip_special_tokens=True)

print("before:", ask(model,  "Where is the Eiffel Tower?"))
print("after :", ask(edited, "Where is the Eiffel Tower?"))
```

`get_engram` runs the expensive part once; `apply_engram(model, engram, alpha=…)` is just a copy + subtraction, so you can sweep `alpha` to trade off forgetting vs retention without recollecting. One call does both: `edit_llm(model, tokenizer, forget=forget, total=forget + retain, alpha=0.6)`.

## How it works

| step | what | cost |
|---|---|---|
| 1. collect | forward pre-hooks accumulate the **mean** `xᵀx` + sample count per layer | one forward pass, no backward |
| 2. compute | projection `P = W · C_target · pinv(C_total)` | one pseudo-inverse per layer |
| 3. apply | `W ← W − α · f_l · P` with a pluggable per-layer scaling `f_l` | a single subtraction |

Efficient by construction — forward-only hooks, magnitude-bounded running-mean accumulation, CPU/GPU split (covariances on `storage_device`), a `float32` solve cast back to the model dtype, and answer-token masking. The per-layer edit weighting `f_l` is pluggable (`count_ratio` default = the paper's `n/N`, `weight_norm`, `effective_rank`, …). Handles `nn.Linear`, GPT-2 `Conv1D` (a transposed linear), and masked variants; full details in the [Guide](https://jeakwon.github.io/ai-engram/guide/).

### Configuration (`EditorConfig`)

| field | default | purpose |
|---|---|---|
| `storage_device` | model's device | where covariances are held; set `"cpu"` if the `D×D` matrices don't fit in VRAM (large models) |
| `absorb_bias` | `True` | absorb bias into the edit for bias-bearing layers |

## Using the editor directly

With your own `DataLoader`s (any `nn.Linear` / GPT-2 `Conv1D` model), drive the editor in three steps — collect, compute, apply:

```python
from engram import EngramEditor, EditorConfig

editor = EngramEditor(model, EditorConfig())
target = editor.collect_statistics(forget_loader)   # Statistics: mean covariance + counts
total  = editor.collect_statistics(total_loader)    # over the reference set
edited = editor.edit(target, total, alpha=1.0)      # compute the engram and subtract it
# or split: engram = editor.compute_engram_weights(target, total); editor.apply(engram, alpha=0.6)
```

**HuggingFace LLM (answer-token masked).** Restrict the covariance to answer tokens with `mask_fn`:

```python
batch_fn = lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
mask_fn  = lambda b: b["labels"] != -100            # covariance over answer tokens only

g_forget = editor.collect_statistics(forget_loader, batch_fn=batch_fn, mask_fn=mask_fn)
g_total  = editor.collect_statistics(total_loader,  batch_fn=batch_fn, mask_fn=mask_fn)
edited = editor.edit(g_forget, g_total, alpha=0.6)  # default scaling = the paper's n/N
# selective per-layer strength:
#   from engram import weight_norm, compose, count_ratio
#   edited = editor.edit(g_forget, g_total, alpha=1.0, scale=compose(count_ratio(1.0), weight_norm(1.0)))
```

Restrict to specific modules with `target_modules` — the LoRA/PEFT convention (`["down_proj"]` by name suffix, or a regex string) — plus `layers_to_transform` for decoder-layer indices. See the [Guide](https://jeakwon.github.io/ai-engram/guide/) for details.

**Mixture-of-experts.** Answer-token masking reaches the experts automatically on transformers&nbsp;<5; on transformers&nbsp;≥5 (fused experts) opt in to the detachable `engram.moe` adapter — `EngramEditor(model, adapters=[FusedExpertAdapter()])` — covering ~35 fused MoE architectures (Mixtral, Qwen2/3/3.5-MoE, DeepSeek-V3, GLM4-MoE, MiniMax, Mistral4, OLMoE, Phi-MoE, …).

## Validation

On **TOFU forget10** with `tofu_Llama-3.2-1B-Instruct`, the engram extraction reproduces the paper's 14-metric **Overall** within **~0.01**:

| condition | ai-engram | paper |
|---|---|---|
| gold (retain90) | 0.998 | 0.998 |
| plain (α=0.6) | 0.706 | 0.698 |
| adaptive-norm (α=1.0, p=1) | 0.817 | 0.818 |

Answer-token NLL confirms strong, *selective* forgetting — the forget set's NLL jumps ~16× while retain is preserved, and adaptive-norm beats plain on both axes. Runnable end-to-end in
[`tests/`](https://github.com/jeakwon/ai-engram/tree/main/tests) and
[`examples/`](https://github.com/jeakwon/ai-engram/tree/main/examples); see the [TOFU page](https://jeakwon.github.io/ai-engram/tofu/).

## API

- `collect_statistics(loader, target_modules=None, batch_fn=None, mask_fn=None, layers_to_transform=None) -> Statistics`
- `compute_engram_weights(target, total) -> EngramResult` · `apply(engram, *, alpha=1.0, scale=count_ratio(1.0)) -> Module` · `edit(target, total, *, alpha, scale)`
- scaling functions: `count_ratio` · `weight_norm` · `effective_rank` · `uniform` · `compose`
- `merge_statistics(*stats)` · `save_statistics(stats, path)` · `load_statistics(path)`

Full reference (auto-generated from docstrings): **[API docs](https://jeakwon.github.io/ai-engram/api/)**.

## Research audits

The repository-local [TOFU unlearning audit](experiments/unlearning_audit/README.md)
contains the reproducible July 2026 suppression-vs-erasure experiments and
their compact results. It replaces the former `ai-engram-repro` side repository
and host-specific `gpu03_*.py` files.

## Citation

`ai-engram` is the reference implementation of **[AI Engram: In Search of Memory Traces in Artificial Intelligence](https://arxiv.org/abs/2606.14997)**, accepted to **ICML 2026 (Oral)**. If you use it, please cite:

```bibtex
@inproceedings{kwon2026aiengram,
  title     = {{AI} Engram: In Search of Memory Traces in Artificial Intelligence},
  author    = {Kwon, Jea and Kim, Dong-Kyum and Kim, Jiwon and Kim, Yonghyun and Kook, Woong and Cha, Meeyoung},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {Oral presentation},
  eprint    = {2606.14997},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url       = {https://arxiv.org/abs/2606.14997}
}
```

GitHub's *“Cite this repository”* button generates this from [`CITATION.cff`](CITATION.cff).

## License

[MIT](https://github.com/jeakwon/ai-engram/blob/main/LICENSE) © Jea Kwon
