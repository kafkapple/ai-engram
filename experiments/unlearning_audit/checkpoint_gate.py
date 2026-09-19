#!/usr/bin/env python3
"""Gate before the method-family comparison (vault note 260919_AI_Engram_checkpoint_gate).

For public OpenUnlearning 1B/forget10 checkpoints and the AI Engram edit (adaptive alpha=1):
  1. start model: is the checkpoint an unlearning of BASE_ID (tofu ..._full)?
     share of tensors bit-identical to BASE_ID; nearest neighbour d(u, full) < d(u, retain90)
  2. damage: forget NLL on forget[100:200] (the relearn_attack eval set) and retain NLL on retain[:100]
  3. 4-bit fragility: share of NF4 codes (blocksize 64, absmax) of decoder Linear weights that differ from BASE_ID

    python -m experiments.unlearning_audit.checkpoint_gate
Writes $AUDIT_RESULTS/checkpoint_gate.json.
"""
from __future__ import annotations

import json
import os

import torch

from experiments.unlearning_audit.tofu import (
    BASE_ID, GOLD_90_ID, edit_model, load_model, load_tokenizer, mean_answer_nll,
)

OUT = os.environ.get("AUDIT_RESULTS") or os.path.join(os.path.dirname(__file__), "results")
PREFIX = "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_"
CANDIDATES = [  # most common alpha/beta per method, lr {1e-5, 5e-5} x epoch {5, 10}
    "GradDiff_lr1e-05_alpha5_epoch5",
    "GradDiff_lr1e-05_alpha5_epoch10",
    "GradDiff_lr5e-05_alpha5_epoch5",
    "GradDiff_lr5e-05_alpha5_epoch10",
    "NPO_lr1e-05_beta0.5_alpha1_epoch5",
    "NPO_lr1e-05_beta0.5_alpha1_epoch10",
    "NPO_lr5e-05_beta0.5_alpha1_epoch5",
    "NPO_lr5e-05_beta0.5_alpha1_epoch10",
    "RMU_lr1e-05_layer5_scoeff100_epoch5",
    "RMU_lr1e-05_layer5_scoeff100_epoch10",
    "RMU_lr5e-05_layer5_scoeff100_epoch5",
    "RMU_lr5e-05_layer5_scoeff100_epoch10",
]
NF4 = torch.tensor([-1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138, -0.18477343, -0.09105004, 0.0,
                    0.07958030, 0.16093020, 0.24611230, 0.33791524, 0.44070983, 0.56261700, 0.72295684, 1.0])


MID = (NF4[1:] + NF4[:-1]) / 2   # nearest-level lookup without a 16x temporary


def nf4_codes(w: torch.Tensor, block: int = 64) -> torch.Tensor:
    flat = w.float().flatten()
    flat = torch.nn.functional.pad(flat, (0, (-flat.numel()) % block)).view(-1, block)
    scaled = flat / flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.bucketize(scaled, MID).to(torch.uint8)


def weights(model) -> dict:
    """fp32 CPU copy; the tied lm_head (same storage as embed_tokens) is dropped so it is counted once."""
    sd = model.state_dict()
    tied = getattr(model.config, "tie_word_embeddings", False)
    return {k: v.detach().float().cpu() for k, v in sd.items() if not (tied and k == "lm_head.weight")}


def is_quantized(k: str, v: torch.Tensor) -> bool:
    return v.dim() == 2 and ".layers." in k   # decoder Linear weights only (4-bit loading keeps embed/lm_head)


def dist(a, b, keys) -> float:
    return sum(float((a[k] - b[k]).pow(2).sum()) for k in keys) ** 0.5


def compare(sd, base, gold, base_codes) -> dict:
    keys = [k for k in base if k in sd and k in gold and base[k].shape == sd[k].shape]
    identical = sum(torch.equal(sd[k], base[k]) for k in keys)
    d_full, d_gold, d_ref = dist(sd, base, keys), dist(sd, gold, keys), dist(gold, base, keys)
    flips = sum(int((nf4_codes(sd[k]) != base_codes[k]).sum()) for k in base_codes)
    total = sum(base[k].numel() for k in base_codes)
    return {"tensors": len(keys), "identical_share": round(identical / len(keys), 4),
            "d_full": round(d_full, 3), "d_retain90": round(d_gold, 3), "d_full_retain90": round(d_ref, 3),
            "starts_from_full": d_full < d_gold, "nf4_code_change_share": round(flips / total, 6),
            "changed_tensors": [k for k in keys if not torch.equal(sd[k], base[k])][:12]}


def main():
    from datasets import load_dataset

    device = "cuda"
    tok = load_tokenizer(BASE_ID)
    full = list(load_dataset("locuslab/TOFU", "full")["train"])
    forget = list(load_dataset("locuslab/TOFU", "forget10_perturbed")["train"])
    retain = list(load_dataset("locuslab/TOFU", "retain_perturbed")["train"])
    f_eval, r_eval = forget[100:200], retain[:100]

    def damage(model):
        return {"forget_nll": round(mean_answer_nll(model, f_eval, tok, device), 3),
                "retain_nll": round(mean_answer_nll(model, r_eval, tok, device), 3)}

    def normalized(row):   # 0 = original model, 1 = never-trained gold
        f0, fg = res["base"]["forget_nll"], res["gold"]["forget_nll"]
        return {**row, "forget_damage_norm": round((row["forget_nll"] - f0) / (fg - f0), 3)}

    model = load_model(BASE_ID, device).eval()
    base = weights(model)
    base_codes = {k: nf4_codes(v) for k, v in base.items() if is_quantized(k, v)}
    res = {"eval": "forget[100:200], retain[:100]", "base": damage(model), "models": {}}
    gold_model = load_model(GOLD_90_ID, device).eval()
    gold = weights(gold_model)
    res["gold"] = damage(gold_model)
    del gold_model
    edited = edit_model(model, forget, full, tok, device)   # adaptive alpha=1, the audited edit
    res["models"]["ai_engram_adaptive_a1"] = normalized({**damage(edited), **compare(weights(edited), base, gold, base_codes)})
    del edited, model
    torch.cuda.empty_cache()
    for name in CANDIDATES:
        m = load_model(PREFIX + name, device).eval()
        res["models"][name] = normalized({**damage(m), **compare(weights(m), base, gold, base_codes)})
        del m
        torch.cuda.empty_cache()
        print(name, {k: v for k, v in res["models"][name].items() if k != "changed_tensors"}, flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/checkpoint_gate.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"wrote {OUT}/checkpoint_gate.json\nGATE_DONE")


if __name__ == "__main__":
    main()
