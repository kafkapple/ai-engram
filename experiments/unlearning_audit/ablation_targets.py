#!/usr/bin/env python3
"""Layer-group ablation: which parts of the network carry the engram?

Reuses the package's own TOFU test harness (tests/test_tofu_unlearn.py) for data +
answer-token-masked covariance + NLL eval — no reinvention. Collects the engram over
ALL linear layers ONCE, then re-applies it restricted to each layer group by zeroing
the per-layer scale outside the group (so no re-collection per group).

Paper claim (§7, Fig 9): engram concentrates in Q/K projections + MLP gate. This test
measures forget-NLL rise (want high) and retain-NLL rise (want low) per group.

Run from the ai-engram repository in a GPU environment with a GPU. Writes results/ablation.{json,png}.
"""
from __future__ import annotations
import json, sys, os
import torch
from torch.utils.data import DataLoader

# module-level names only run on import (no test executes)
from tests.test_tofu_unlearn import (  # noqa: E402
    BASE_ID, N_TOTAL, N_RETAIN_EVAL, IGNORE,
    _QAData, _make_collate, _mean_answer_nll,
    ADAPT_ALPHA, ADAPT_P,
)
from engram import EditorConfig, EngramEditor, compose, count_ratio, weight_norm  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results")

# Llama-3.2 decoder module suffixes per group. None = all-linear (the full-edit baseline).
GROUPS = {
    "all-linear":        None,
    "attn_q+k":          ["q_proj", "k_proj"],
    "attn_v+o":          ["v_proj", "o_proj"],
    "mlp_gate":          ["gate_proj"],
    "mlp_up+down":       ["up_proj", "down_proj"],
    "paper_qk+gate":     ["q_proj", "k_proj", "gate_proj"],
}


def restrict(base_scale, suffixes):
    """Wrap a ScaleFn: keep factor for layers whose name ends in one of `suffixes`,
    zero everything else. suffixes=None -> unchanged (all layers)."""
    def f(layers):
        base = base_scale(layers)
        if suffixes is None:
            return base
        return {n: (v if any(n.endswith(s) for s in suffixes) else 0.0)
                for n, v in base.items()}
    return f


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    forget = load_dataset("locuslab/TOFU", "forget10_perturbed")["train"]
    retain = load_dataset("locuslab/TOFU", "retain_perturbed")["train"]
    total = load_dataset("locuslab/TOFU", "full")["train"].shuffle(seed=0).select(range(N_TOTAL))

    editor = EngramEditor(base, EditorConfig(storage_device=torch.device(device)))

    def covdl(ds):
        return DataLoader(_QAData(ds, tok), batch_size=8, collate_fn=_make_collate(tok.pad_token_id))

    def feats(b):
        return {"input_ids": b["input_ids"].to(device), "attention_mask": b["attention_mask"].to(device)}
    mask_fn = lambda b: b["labels"] != IGNORE

    # collect ONCE over all layers
    g_forget = editor.collect_statistics(covdl(forget), batch_fn=feats, mask_fn=mask_fn)
    g_total = editor.collect_statistics(covdl(total), batch_fn=feats, mask_fn=mask_fn)
    engram = editor.compute_engram_weights(g_forget, g_total)
    del g_forget, g_total; torch.cuda.empty_cache()

    retain_eval = retain.select(range(min(N_RETAIN_EVAL, len(retain))))
    f0 = _mean_answer_nll(base, forget, tok, device)
    r0 = _mean_answer_nll(base, retain_eval, tok, device)
    print(f"[base] forget {f0:.3f} | retain {r0:.3f}")

    adaptive = compose(count_ratio(1.0), weight_norm(ADAPT_P))  # paper's best condition
    rows = []
    for name, suffixes in GROUPS.items():
        n_layers = "all" if suffixes is None else sum(
            1 for k in engram.layers if any(k.endswith(s) for s in suffixes))
        edited = editor.apply(engram, alpha=ADAPT_ALPHA, scale=restrict(adaptive, suffixes)).eval()
        f1 = _mean_answer_nll(edited, forget, tok, device)
        r1 = _mean_answer_nll(edited, retain_eval, tok, device)
        del edited; torch.cuda.empty_cache()
        row = {"group": name, "n_layers": n_layers,
               "forget_d": round(f1 - f0, 3), "retain_d": round(r1 - r0, 3),
               "selectivity": round((f1 - f0) / max(r1 - r0, 1e-3), 2)}
        rows.append(row)
        print(f"  {name:16s} ({n_layers:>3} L): forget Δ{f1-f0:+.3f} | retain Δ{r1-r0:+.3f} | sel {row['selectivity']}")

    res = {"base": {"forget": round(f0, 3), "retain": round(r0, 3)},
           "alpha": ADAPT_ALPHA, "p": ADAPT_P, "rows": rows}
    with open(f"{OUT}/ablation.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"[ablation] wrote {OUT}/ablation.json")

    # bar chart (forget Δ vs retain Δ per group)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        g = [r["group"] for r in rows]; x = np.arange(len(g)); w = 0.38
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(x - w/2, [r["forget_d"] for r in rows], w, label="forget Δ (↑ = more forgetting)", color="#c0392b")
        ax.bar(x + w/2, [r["retain_d"] for r in rows], w, label="retain Δ (↓ = better preserved)", color="#2980b9")
        ax.set_xticks(x); ax.set_xticklabels(g, rotation=20, ha="right")
        ax.set_ylabel("answer-token NLL change"); ax.axhline(0, color="k", lw=.6)
        ax.set_title(f"AI Engram — layer-group ablation (TOFU forget10, adaptive α={ADAPT_ALPHA})")
        ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/ablation.png", dpi=130)
        print(f"[ablation] wrote {OUT}/ablation.png")
    except Exception as e:
        print(f"[ablation] plot skipped: {e}")


if __name__ == "__main__":
    main()
