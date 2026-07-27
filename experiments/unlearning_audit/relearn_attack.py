#!/usr/bin/env python3
"""Test 2 (suppression-vs-erasure): RELEARNING ATTACK (Hu'25, behavioral — immune to the
input-separability confound that killed Test 1).

Fine-tune two models on a SMALL forget subset for a few steps, track held-out forget-NLL:
  - edited = engram-edited (forget10 removed via labeled adaptive edit)
  - gold   = tofu_..._retain90 (never learned the forget authors — the honest control)
If edited recovers forget knowledge MUCH FASTER than gold (which must learn it from scratch),
the knowledge was SUPPRESSED (latent), not ERASED. Same LR/steps for both = fair.

Also tracks retain-NLL to see collateral during relearning. Reuses tests/test_tofu_unlearn.py.
Writes results/relearn_attack.json.
"""
from __future__ import annotations
import json, os, sys
import torch
from torch.utils.data import DataLoader

from experiments.unlearning_audit.tofu import (  # noqa: E402
    BASE_ID, QAData, make_collate, mean_answer_nll,
)
from experiments.unlearning_audit.tofu import edit_model

OUT = os.path.join(os.path.dirname(__file__), "results")
GOLD_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
N_TOTAL = 4000
STEPS, LR, BS = 40, 2e-5, 8
EVAL_EVERY = 5


def relearn_curve(model, train_rows, f_eval, r_eval, tok, device):
    """Fine-tune on train_rows; log (step, forget_nll, retain_nll)."""
    pad = tok.pad_token_id
    dl = DataLoader(QAData(train_rows, tok), batch_size=BS, shuffle=True, collate_fn=make_collate(pad))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    curve = [[0, round(mean_answer_nll(model, f_eval, tok, device), 3),
                 round(mean_answer_nll(model, r_eval, tok, device), 3)]]
    step = 0
    while step < STEPS:
        model.train()
        for b in dl:
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            out.loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if step % EVAL_EVERY == 0:
                model.eval()
                curve.append([step, round(mean_answer_nll(model, f_eval, tok, device), 3),
                                    round(mean_answer_nll(model, r_eval, tok, device), 3)])
                model.train()
            if step >= STEPS:
                break
    return curve


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    full = list(load_dataset("locuslab/TOFU", "full")["train"])
    forget = list(load_dataset("locuslab/TOFU", "forget10_perturbed")["train"])
    retain = list(load_dataset("locuslab/TOFU", "retain_perturbed")["train"])
    train_rows = forget[:16]           # small relearn set
    f_eval = forget[100:200]           # HELD-OUT forget (recovery must generalize)
    r_eval = retain[:100]

    def load(id_):
        return AutoModelForCausalLM.from_pretrained(
            id_, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)

    # edited: build from base, then relearn
    base = load(BASE_ID).eval()
    edited = edit_model(base, forget, full, tok, device)
    del base; torch.cuda.empty_cache()
    edited_curve = relearn_curve(edited, train_rows, f_eval, r_eval, tok, device)
    del edited; torch.cuda.empty_cache()

    # gold: retain90 (never saw forget authors) — same relearn
    gold = load(GOLD_ID)
    gold_curve = relearn_curve(gold, train_rows, f_eval, r_eval, tok, device)
    del gold; torch.cuda.empty_cache()

    def half_recovery(curve):
        f0, fend = curve[0][1], curve[-1][1]
        target = f0 - 0.5 * (f0 - fend)
        for st, fn, _ in curve:
            if fn <= target:
                return st
        return None

    res = {"cols": ["step", "forget_nll", "retain_nll"],
           "edited": edited_curve, "gold": gold_curve,
           "edited_forget_start_end": [edited_curve[0][1], edited_curve[-1][1]],
           "gold_forget_start_end": [gold_curve[0][1], gold_curve[-1][1]],
           "edited_half_recovery_step": half_recovery(edited_curve),
           "gold_half_recovery_step": half_recovery(gold_curve)}
    print(json.dumps(res, indent=2))
    with open(f"{OUT}/relearn_attack.json", "w") as fp:
        json.dump(res, fp, indent=2)
    er, gr = res["edited_half_recovery_step"], res["gold_half_recovery_step"]
    print(f"\n[verdict] edited half-recovery step={er}  vs  gold={gr}")
    print("  edited << gold (recovers much faster) => forget knowledge was SUPPRESSED (latent), not erased")
    print("  edited ~ gold => genuine erasure (relearns like a model that never knew it)")
    print("RELEARN_DONE")


if __name__ == "__main__":
    main()
