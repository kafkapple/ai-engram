#!/usr/bin/env python3
"""Step 2 (contribution-hardening): alpha/scale SWEEP -> suppression/utility frontier.

Pre-empts the "alpha was too weak" defense. For each (alpha, scale), report:
  - static forget NLL (how well it forgot)      [higher = stronger static unlearning]
  - retain NLL (utility)                         [higher = more collateral]
  - relearning recovery: held-out forget NLL after fine-tune + recovery fraction R
Question: does stronger subtraction give DEEPER erasure (relearn fails) or just utility loss
while relearning still recovers? Engram P is computed ONCE; only apply() varies (cheap).

Reuses tests/test_tofu_unlearn.py. Writes results/alpha_sweep.json.
S_G (gold forget-relearn end ~2.32) from Step 1 novel_control; S_O measured.
"""
from __future__ import annotations
import copy, json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.unlearning_audit.tofu import (  # noqa: E402
    BASE_ID, QAData, collect_engram, make_collate, mean_answer_nll,
)
from engram import compose, count_ratio, weight_norm  # noqa: E402

OUT = os.environ.get("AUDIT_RESULTS") or os.path.join(os.path.dirname(__file__), "results")
N_TOTAL = 4000
STEPS, LR, BS, EVAL_EVERY = 40, 2e-5, 8, 20
SEEDS = [0, 1]
S_G = 2.315   # gold forget-relearn end (Step 1) — the "never had it" floor
ALPHAS = [0.5, 1.0, 2.0]


def relearn_end(model, train_rows, eval_rows, tok, device, seed):
    torch.manual_seed(seed)
    dl = DataLoader(QAData(train_rows, tok), batch_size=BS, shuffle=True,
                    collate_fn=make_collate(tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    step = 0
    while step < STEPS:
        model.train()
        for b in dl:
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            out.loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if step >= STEPS:
                break
    model.eval()
    return mean_answer_nll(model, eval_rows, tok, device)


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    full = list(load_dataset("locuslab/TOFU", "full")["train"])
    forget = list(load_dataset("locuslab/TOFU", "forget10_perturbed")["train"])
    retain = list(load_dataset("locuslab/TOFU", "retain_perturbed")["train"])
    f_train, f_eval, r_eval = forget[:16], forget[100:200], retain[:100]

    S_O = round(mean_answer_nll(base, f_eval, tok, device), 3)     # original still-has-it
    base_retain = round(mean_answer_nll(base, r_eval, tok, device), 3)

    # collect covariance + engram projection ONCE (alpha/scale-independent)
    ed, engram = collect_engram(base, forget, full, tok, device)

    scales = {"plain": count_ratio(1.0), "adaptive": compose(count_ratio(1.0), weight_norm(1.0))}
    rows = []
    for stype, scale in scales.items():
        for a in ALPHAS:
            edited = ed.apply(engram, alpha=a, scale=scale).eval()          # cheap: deepcopy + subtract
            sf = round(mean_answer_nll(edited, f_eval, tok, device), 3)     # static forget
            sr = round(mean_answer_nll(edited, r_eval, tok, device), 3)     # retain utility
            recs = []
            for s in SEEDS:
                m = copy.deepcopy(edited)
                recs.append(relearn_end(m, f_train, f_eval, tok, device, s))
                del m; torch.cuda.empty_cache()
            rec = float(np.mean(recs))
            R = (S_G - rec) / (S_G - S_O + 1e-6)          # 1=recovers to original, 0=like gold
            row = {"scale": stype, "alpha": a, "static_forget": sf, "retain": sr,
                   "relearn_end": round(rec, 3), "recovery_R": round(float(R), 3)}
            rows.append(row)
            print(f"{stype:9s} a={a}: static_forget={sf} retain={sr}(base {base_retain}) "
                  f"relearn_end={rec:.2f} R={R:.2f}")
            del edited; torch.cuda.empty_cache()

    res = {"S_O": S_O, "S_G": S_G, "base_retain": base_retain, "rows": rows}
    print(json.dumps(res, indent=2))
    with open(f"{OUT}/alpha_sweep.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print("\n[interpretation] if R stays >0 (recovers) across all alpha while retain degrades with alpha")
    print("  => stronger subtraction trades utility, NOT erasure depth (suppression persists).")
    print("  if R -> 0 at high alpha => stronger edit genuinely erases (relearn fails like gold).")
    print("ALPHA_SWEEP_DONE")


if __name__ == "__main__":
    main()
