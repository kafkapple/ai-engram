#!/usr/bin/env python3
"""Experiment A (decisive): ENTITY-FAMILIAR control — splits entity-familiarity from
propositional fact-trace (the confound the critical panel raised against Step 1/2).

TOFU forget10 = 20 authors x 20 contiguous QA. Within-author split:
  ef_train  = each author's QA 1-12 (240)  -> build GOLD_EF entity familiarity + partial facts
  eval_held = each author's QA 13-20 (160)  -> NEVER in GOLD_EF training; held out for all
Models, all relearn on the SAME 16 QA (from ef_train), eval held-out 160:
  EDITED   = base(all 400) + engram-remove(forget)   [had eval facts, then removed]
  GOLD_EF  = retain90 + finetune on ef_train(240)     [entity-familiar all authors, NEVER eval facts]
  GOLD     = retain90                                  [no forget familiarity at all]
Decisive:
  EDITED recovers eval_held but GOLD_EF does NOT  => residual FACT trace (entity controlled) = suppression.
  EDITED ~ GOLD_EF                                 => entity familiarity alone explains it (claim refuted).

Reuses tests/test_tofu_unlearn.py. Writes results/entity_control.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.unlearning_audit.tofu import (  # noqa: E402
    BASE_ID, QAData, make_collate, mean_answer_nll,
)
from experiments.unlearning_audit.tofu import cpu_state_dict, edit_model, finetune

OUT = os.environ.get("AUDIT_RESULTS") or os.path.join(os.path.dirname(__file__), "results")
GOLD_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
N_TOTAL = 4000
PER_AUTHOR, N_AUTHORS, SPLIT = 20, 20, 12
RE_STEPS, EF_STEPS, LR, BS, EVAL_EVERY = 40, 150, 2e-5, 8, 20
SEEDS = [0, 1, 2]


def relearn_curve(model, train_rows, eval_rows, tok, device, seed):
    model.eval()  # 260710 fix: shell model is reused across runs and was left in train()
    # mode by the previous call, so the step-0 anchor NLL was measured in train mode
    # (harmless iff dropout=0, but eval mode makes the anchor unambiguous).
    torch.manual_seed(seed)
    dl = DataLoader(QAData(train_rows, tok), batch_size=BS, shuffle=True, collate_fn=make_collate(tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    curve = [round(mean_answer_nll(model, eval_rows, tok, device), 3)]
    step = 0
    while step < RE_STEPS:
        model.train()
        for b in dl:
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            out.loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if step % EVAL_EVERY == 0:
                model.eval(); curve.append(round(mean_answer_nll(model, eval_rows, tok, device), 3)); model.train()
            if step >= RE_STEPS:
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

    ef_train, eval_held = [], []
    for a in range(N_AUTHORS):
        base_i = a * PER_AUTHOR
        ef_train += forget[base_i: base_i + SPLIT]
        eval_held += forget[base_i + SPLIT: base_i + PER_AUTHOR]
    relearn16 = [ef_train[i] for i in range(0, len(ef_train), len(ef_train) // 16)][:16]
    # NOTE (260710): stride 15 vs author-block 12 → NOT evenly spread; authors 4/9/14/19
    # get zero relearn rows (eval_held mixes directly-relearned and cross-author-only
    # authors). Kept as-is for comparability with recorded runs; fix for fresh protocols:
    # one explicit row per author.

    load = lambda id_: AutoModelForCausalLM.from_pretrained(
        id_, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)

    model = load(BASE_ID)
    S_O = round(mean_answer_nll(model.eval(), eval_held, tok, device), 3)   # original still-has-it
    edited_sd = cpu_state_dict(edit_model(model, forget, full, tok, device))
    del model; torch.cuda.empty_cache()

    gold = load(GOLD_ID); gold_sd = cpu_state_dict(gold)
    # GOLD_EF: finetune gold on ef_train -> entity-familiar with all authors, never eval facts
    gold_ef = finetune(gold, ef_train, tok, device, EF_STEPS)
    ef_train_nll = round(mean_answer_nll(gold_ef, relearn16, tok, device), 3)   # should be low (memorized)
    ef_eval_nll = round(mean_answer_nll(gold_ef, eval_held, tok, device), 3)    # should be HIGH (never saw)
    gold_ef_sd = cpu_state_dict(gold_ef)
    del gold, gold_ef; torch.cuda.empty_cache()

    model = load(BASE_ID)   # reusable shell for relearn runs

    def run(sd, seed):
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        return relearn_curve(model, relearn16, eval_held, tok, device, seed)

    conds = {"edited": edited_sd, "gold_ef": gold_ef_sd, "gold": gold_sd}
    res = {"S_O_original": S_O, "gold_ef_check": {"relearn16_nll": ef_train_nll, "eval_nll_preattack": ef_eval_nll},
           "seeds": SEEDS, "eval_step_grid": list(range(0, RE_STEPS + 1, EVAL_EVERY)), "runs": {}}
    for name, sd in conds.items():
        arr = np.array([run(sd, s) for s in SEEDS])   # [seed, step]
        res["runs"][name] = {"mean": [round(x, 3) for x in arr.mean(0)],
                             "std": [round(x, 3) for x in arr.std(0)]}
        print(f"{name:8s}: eval_held start {arr[:,0].mean():.2f} -> end {arr[:,-1].mean():.2f} (±{arr[:,-1].std():.2f})")

    ed_end = res["runs"]["edited"]["mean"][-1]; ef_end = res["runs"]["gold_ef"]["mean"][-1]
    g_end = res["runs"]["gold"]["mean"][-1]
    res["verdict_gap_edited_vs_goldef"] = round(ef_end - ed_end, 3)
    print(json.dumps(res, indent=2))
    with open(f"{OUT}/entity_control.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"\n[GOLD_EF sanity] relearn16 NLL={ef_train_nll} (low=memorized entities), "
          f"eval_held pre-attack NLL={ef_eval_nll} (high=never saw eval facts)")
    print(f"[verdict] eval_held recovery end: edited={ed_end}  gold_ef={ef_end}  gold={g_end}  (S_O={S_O})")
    print("  edited << gold_ef  => FACT-TRACE suppression, entity-familiarity CONTROLLED (claim holds)")
    print("  edited ~= gold_ef  => entity familiarity explains recovery (claim REFUTED)")
    print("ENTITY_CONTROL_DONE")


if __name__ == "__main__":
    main()
