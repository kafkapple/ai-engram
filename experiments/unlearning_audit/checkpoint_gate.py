#!/usr/bin/env python3
"""Gate before the method-family comparison (vault note 260919_AI_Engram_checkpoint_gate).

For public OpenUnlearning 1B/forget10 checkpoints and the AI Engram edit (adaptive alpha=1):
  1. start model: is the checkpoint an unlearning of BASE_ID (tofu ..._full)?
     share of tensors bit-identical to BASE_ID, and ||theta_u - theta_full|| / ||theta_full - theta_retain90||
  2. damage: forget NLL on forget[100:200] (the relearn_attack eval set) and retain NLL on retain[:100]
  3. 4-bit fragility: share of NF4 codes (blocksize 64, absmax) that differ from BASE_ID's codes

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
CANDIDATES = [
    "GradDiff_lr1e-05_alpha10_epoch10", "GradDiff_lr5e-05_alpha5_epoch5",
    "NPO_lr1e-05_beta0.05_alpha1_epoch10", "NPO_lr5e-05_beta0.5_alpha5_epoch5",
    "RMU_lr1e-05_layer10_scoeff100_epoch10", "RMU_lr5e-05_layer5_scoeff1_epoch5",
]
NF4 = torch.tensor([-1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138, -0.18477343, -0.09105004, 0.0,
                    0.07958030, 0.16093020, 0.24611230, 0.33791524, 0.44070983, 0.56261700, 0.72295684, 1.0])


def nf4_codes(w: torch.Tensor, block: int = 64) -> torch.Tensor:
    flat = w.float().flatten()
    flat = torch.nn.functional.pad(flat, (0, (-flat.numel()) % block)).view(-1, block)
    scaled = flat / flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    return (scaled.unsqueeze(-1) - NF4).abs().argmin(-1).to(torch.uint8)


def weights(model) -> dict:
    return {k: v.detach().float().cpu() for k, v in model.state_dict().items()}


def compare(sd, base, gold) -> dict:
    keys = [k for k in base if k in sd and base[k].shape == sd[k].shape]
    identical = sum(torch.equal(sd[k], base[k]) for k in keys)
    d_u = sum(float((sd[k] - base[k]).pow(2).sum()) for k in keys) ** 0.5
    d_g = sum(float((gold[k] - base[k]).pow(2).sum()) for k in keys if k in gold) ** 0.5
    mats = [k for k in keys if base[k].dim() == 2]
    flips = sum(int((nf4_codes(sd[k]) != nf4_codes(base[k])).sum()) for k in mats)
    total = sum(base[k].numel() for k in mats)
    return {"tensors": len(keys), "identical_share": round(identical / len(keys), 4),
            "dist_ratio_vs_retain90": round(d_u / d_g, 4), "nf4_code_change_share": round(flips / total, 6),
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

    model = load_model(BASE_ID, device).eval()
    base = weights(model)
    res = {"eval": "forget[100:200], retain[:100]", "base": damage(model), "models": {}}
    gold_model = load_model(GOLD_90_ID, device).eval()
    gold = weights(gold_model)
    res["gold"] = damage(gold_model)
    del gold_model
    edited = edit_model(model, forget, full, tok, device)   # adaptive alpha=1, the audited edit
    res["models"]["ai_engram_adaptive_a1"] = {**damage(edited), **compare(weights(edited), base, gold)}
    del edited, model
    torch.cuda.empty_cache()
    for name in CANDIDATES:
        m = load_model(PREFIX + name, device).eval()
        res["models"][name] = {**damage(m), **compare(weights(m), base, gold)}
        del m
        torch.cuda.empty_cache()
        print(name, {k: v for k, v in res["models"][name].items() if k != "changed_tensors"}, flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/checkpoint_gate.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"wrote {OUT}/checkpoint_gate.json\nGATE_DONE")


if __name__ == "__main__":
    main()
