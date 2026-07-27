#!/usr/bin/env python3
"""Cross-validation P1 — replicate the core battery on TOFU forget05 (10 authors, gold=retain95).

Purpose: external validity of the three-illusion findings beyond forget10.
Battery (minimal, pre-registered):
  A. Gradient dissociation: relearn 16 real QA -> held-out recovery, arms EDITED/GOLD_EF/GOLD.
  B. Context damage: C0 (no context) / C1 (12 same-author forget shots) / C3L (12 length-matched
     unrelated retain shots), NLL + greedy-decode degen, arms EDITED/GOLD_EF/GOLD.

PRE-REGISTERED replication criteria (directional, not point values — declared before run):
  R1 (dissociation): gold-anchor R_edited >= 0.4 AND gold shows no comparable recovery
      (gold end-NLL stays > 1.5).
  R2 (context damage): EDITED C3L NLL >= C0 + 0.3 AND EDITED C3L degen >= 0.10
      while GOLD_EF and GOLD C3L degen < 0.05.
  BOTH hold -> forget10 findings replicate on a second split; either fails -> report as
  split-sensitivity, no post-hoc rescue.

relearn16 uses explicit per-author sampling (fixes the forget10 stride flaw: full author coverage).
Writes results/xval_forget05.json.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from tests.test_tofu_unlearn import (  # noqa: E402
    BASE_ID, IGNORE, _QAData, _make_collate, _mean_answer_nll, ADAPT_ALPHA, ADAPT_P,
)
from experiments.unlearning_audit.entity_control import finetune, relearn_curve, EF_STEPS  # noqa: E402
from experiments.unlearning_audit.generation import _chat_ids, item_nlls, greedy_decode, degenerate  # noqa: E402
from experiments.unlearning_audit.length_match import pick_length_matched  # noqa: E402
from engram import EditorConfig, EngramEditor, compose, count_ratio, weight_norm  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results")
GOLD_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95"
N_TOTAL = 4000
PER_AUTHOR, N_AUTHORS, SPLIT, K = 20, 10, 12, 12
SEEDS = [0, 1, 2]


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    forget = list(load_dataset("locuslab/TOFU", "forget05_perturbed")["train"])
    retain = list(load_dataset("locuslab/TOFU", "retain_perturbed")["train"])  # disjoint from forget10 ⊇ forget05
    total = load_dataset("locuslab/TOFU", "full")["train"].shuffle(seed=0).select(range(N_TOTAL))

    ef, ev = [], []
    for a in range(N_AUTHORS):
        b = a * PER_AUTHOR
        ef.append(forget[b: b + SPLIT])
        ev.append(forget[b + SPLIT: b + PER_AUTHOR])
    ef_flat = [r for rows in ef for r in rows]
    ev_flat = [r for rows in ev for r in rows]  # 80, author-major
    # explicit per-author relearn set: 1 row per author + 1 extra for first 6 -> 16, full coverage
    relearn16 = [ef[a][0] for a in range(N_AUTHORS)] + [ef[a][6] for a in range(6)]

    load = lambda id_: AutoModelForCausalLM.from_pretrained(
        id_, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)

    model = load(BASE_ID).eval()
    S_O = round(_mean_answer_nll(model, ev_flat, tok, device), 3)

    ed = EngramEditor(model, EditorConfig(storage_device=torch.device(device)))
    dl = lambda rows: DataLoader(_QAData(rows, tok), batch_size=8, collate_fn=_make_collate(tok.pad_token_id))
    feats = lambda b: {"input_ids": b["input_ids"].to(device), "attention_mask": b["attention_mask"].to(device)}
    mask = lambda b: b["labels"] != IGNORE
    g_total = ed.collect_statistics(dl(total), batch_fn=feats, mask_fn=mask)
    g_forget = ed.collect_statistics(dl(forget), batch_fn=feats, mask_fn=mask)
    eng = ed.compute_engram_weights(g_forget, g_total)
    del g_forget, g_total
    torch.cuda.empty_cache()
    edited = ed.apply(eng, alpha=ADAPT_ALPHA, scale=compose(count_ratio(1.0), weight_norm(ADAPT_P))).eval()
    edited_sd = {k: v.detach().cpu().clone() for k, v in edited.state_dict().items()}
    edited_static = round(_mean_answer_nll(edited, ev_flat, tok, device), 3)
    del edited, eng
    torch.cuda.empty_cache()

    gold = load(GOLD_ID)
    gold_sd = {k: v.detach().cpu().clone() for k, v in gold.state_dict().items()}
    gold_ef = finetune(gold, ef_flat, tok, device, EF_STEPS)
    gold_ef_sd = {k: v.detach().cpu().clone() for k, v in gold_ef.state_dict().items()}
    del gold, gold_ef
    torch.cuda.empty_cache()

    model = load(BASE_ID)  # shell
    arms = {"edited": edited_sd, "gold_ef": gold_ef_sd, "gold": gold_sd}

    res = {"split": "forget05", "gold_id": GOLD_ID, "S_O_original": S_O,
           "edited_static": edited_static, "seeds": SEEDS, "runs": {}, "icl": {}}

    # A. relearn dissociation
    for name, sd in arms.items():
        def run(seed):
            model.load_state_dict({k: v.to(device) for k, v in sd.items()})
            return relearn_curve(model, relearn16, ev_flat, tok, device, seed)
        arr = np.array([run(s) for s in SEEDS])
        start, end = float(arr[:, 0].mean()), float(arr[:, -1].mean())
        res["runs"][name] = {"mean": [round(float(x), 3) for x in arr.mean(0)],
                             "std": [round(float(x), 3) for x in arr.std(0)]}
        print(f"[relearn] {name:8s}: {start:.2f} -> {end:.2f}")
    sg = res["runs"]["gold"]["mean"][-1]
    se = res["runs"]["edited"]["mean"][-1]
    s0 = res["runs"]["edited"]["mean"][0]
    R_gold_anchor = round((sg - se) / (sg - S_O), 3) if abs(sg - S_O) > 1e-6 else None
    res["R_gold_anchor_edited"] = R_gold_anchor

    # B. context damage C0/C1/C3L
    c1_stats_ans = float(np.mean([len(r["answer"].split()) for r in ef_flat]))
    c1_stats_q = float(np.mean([len(r["question"].split()) for r in ef_flat]))
    neutral = pick_length_matched(retain, c1_stats_ans, c1_stats_q, k=K)
    conds = {}
    for cond in ["C0", "C1", "C3L"]:
        items = []
        for a in range(N_AUTHORS):
            if cond == "C1":
                shots = [(r["question"], r["answer"]) for r in ef[a]]
            elif cond == "C3L":
                shots = [(r["question"], r["answer"]) for r in neutral]
            else:
                shots = []
            for r in ev[a]:
                items.append(_chat_ids(tok, shots, r["question"], r["answer"]))
        conds[cond] = items

    for name, sd in arms.items():
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        model.eval()
        row = {}
        for cond, items in conds.items():
            nlls = item_nlls(model, items, tok, device)
            preds = greedy_decode(model, items, tok, device)
            row[cond] = {"nll": round(float(np.mean(nlls)), 3),
                         "degen": round(float(np.mean([degenerate(p) for p in preds])), 3)}
        res["icl"][name] = row
        print(f"[icl] {name:8s}: " + "  ".join(f"{c}={row[c]['nll']}/deg{row[c]['degen']}" for c in conds))

    e, gf, g = res["icl"]["edited"], res["icl"]["gold_ef"], res["icl"]["gold"]
    r1 = (R_gold_anchor is not None and R_gold_anchor >= 0.4 and sg > 1.5)
    r2 = (e["C3L"]["nll"] >= e["C0"]["nll"] + 0.3 and e["C3L"]["degen"] >= 0.10
          and gf["C3L"]["degen"] < 0.05 and g["C3L"]["degen"] < 0.05)
    res["prereg"] = {"R1_dissociation": bool(r1), "R2_context_damage": bool(r2),
                     "replicated": bool(r1 and r2)}
    with open(f"{OUT}/xval_forget05.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print(json.dumps(res, indent=2))
    print("XVAL_FORGET05_DONE")


if __name__ == "__main__":
    main()
