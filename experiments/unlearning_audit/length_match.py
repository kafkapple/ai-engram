#!/usr/bin/env python3
"""C3L — length-matched neutral-context control (promotes/demotes the E2v2 side-observation).

Observation at stake: EDITED degrades MOST with unrelated-author context (C3: gain −1.29,
degen 21%) — but C3 shots were the longest (30.8 answer-words vs C1 24.1), so "general
context damage" is confounded with "longer context". This run selects 12 retain-author
shots LENGTH-MATCHED to the C1 shot distribution and re-measures only that condition
(C3L) for EDITED and GOLD_EF. C0/C1/C3 per-item results are reused verbatim from
e2_v2.json (deterministic protocol, same items).

PRE-REGISTERED JUDGMENT (thresholds declared here before running; do not widen):
  Primary: paired per-item Δ = NLL_C3L − NLL_C1 for EDITED, author-cluster 95% CI.
  PROMOTE ("무관-컨텍스트 손상" confirmed) if:
    ci_lo(Δ) > −0.10  (length-matched neutral context hurts at least as much as forget context)
    AND edited degen_C3L ≥ 0.10  AND gold_ef C3L is normal (nll ≤ its C0 + 0.2, degen < 0.05).
  DEMOTE (length artifact) if: mean(Δ) < −0.30 AND edited degen_C3L < 0.05.
  Otherwise PARTIAL (report as unresolved; no claim).

Writes results/c3_lenmatch.json.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch

from experiments.unlearning_audit.tofu import BASE_ID, mean_answer_nll  # noqa: E402
from experiments.unlearning_audit.entity_control import finetune, EF_STEPS  # noqa: E402
from experiments.unlearning_audit.generation import (  # noqa: E402
    _chat_ids, item_nlls, greedy_decode, token_f1, degenerate, cluster_boot_ci,
    CKPT, OUT, GOLD_ID, PER_AUTHOR, N_AUTHORS, SPLIT, K,
)


def pick_length_matched(retain, target_ans, target_q, k=K):
    """Greedy-pick k retain rows minimizing distance to C1 mean answer/question word counts."""
    scored = sorted(
        retain,
        key=lambda r: abs(len(r["answer"].split()) - target_ans) + 0.5 * abs(len(r["question"].split()) - target_q),
    )
    return scored[:k]


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prev = json.load(open(f"{OUT}/e2_v2.json"))  # reuse C0/C1/C3 per-item results

    forget = list(load_dataset("locuslab/TOFU", "forget10_perturbed")["train"])
    retain = list(load_dataset("locuslab/TOFU", "retain_perturbed")["train"])
    ef, ev = [], []
    for a in range(N_AUTHORS):
        b = a * PER_AUTHOR
        ef.append(forget[b: b + SPLIT])
        ev.append(forget[b + SPLIT: b + PER_AUTHOR])

    c1_rows = [r for rows in ef for r in rows]
    t_ans = float(np.mean([len(r["answer"].split()) for r in c1_rows]))
    t_q = float(np.mean([len(r["question"].split()) for r in c1_rows]))
    neutral = pick_length_matched(retain, t_ans, t_q)
    ach_ans = float(np.mean([len(r["answer"].split()) for r in neutral]))
    ach_q = float(np.mean([len(r["question"].split()) for r in neutral]))
    print(f"[len-match] C1 target ans/q = {t_ans:.1f}/{t_q:.1f}  ->  C3L achieved {ach_ans:.1f}/{ach_q:.1f}")

    shots = [(r["question"], r["answer"]) for r in neutral]
    items = [_chat_ids(tok, shots, r["question"], r["answer"]) for a in range(N_AUTHORS) for r in ev[a]]
    gold_answers = [r["answer"] for a in range(N_AUTHORS) for r in ev[a]]

    load = lambda id_: AutoModelForCausalLM.from_pretrained(
        id_, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)

    edited_sd = torch.load(f"{CKPT}/edited_adaptive_sd.pt", map_location="cpu", weights_only=True)
    gold = load(GOLD_ID)
    gold_ef = finetune(gold, c1_rows, tok, device, EF_STEPS)  # seed 0, identical to e2_v2 rebuild
    gold_ef_sd = {k: v.detach().cpu().clone() for k, v in gold_ef.state_dict().items()}
    del gold, gold_ef
    torch.cuda.empty_cache()

    model = load(BASE_ID).eval()
    res = {"len_match": {"C1_target_ans_q": [round(t_ans, 1), round(t_q, 1)],
                         "C3L_achieved_ans_q": [round(ach_ans, 1), round(ach_q, 1)]},
           "arms": {}}
    per_item = {}
    for name, sd in {"edited": edited_sd, "gold_ef": gold_ef_sd}.items():
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        model.eval()
        nlls = item_nlls(model, items, tok, device)
        preds = greedy_decode(model, items, tok, device)
        per_item[name] = nlls
        res["arms"][name] = {
            "C3L_nll": round(float(np.mean(nlls)), 3),
            "C3L_f1": round(float(np.mean([token_f1(p, g) for p, g in zip(preds, gold_answers)])), 3),
            "C3L_degen": round(float(np.mean([degenerate(p) for p in preds])), 3),
            "C0_nll_prev": prev["arms"][name]["C0"]["nll"],
            "C1_nll_prev": prev["arms"][name]["C1"]["nll"],
            "C3_nll_prev": prev["arms"][name]["C3"]["nll"],
        }
        print(f"{name:8s}: C3L nll={res['arms'][name]['C3L_nll']} f1={res['arms'][name]['C3L_f1']} "
              f"degen={res['arms'][name]['C3L_degen']} (prev C1 {res['arms'][name]['C1_nll_prev']}, C3 {res['arms'][name]['C3_nll_prev']})")

    # primary: EDITED paired Δ = NLL_C3L − NLL_C1, author-cluster CI
    d = np.array(per_item["edited"]) - np.array(prev["per_item_nll"]["edited"]["C1"])
    mean, lo, hi = cluster_boot_ci(d, np.zeros_like(d))
    e, g = res["arms"]["edited"], res["arms"]["gold_ef"]
    goldef_ok = (g["C3L_nll"] <= g["C0_nll_prev"] + 0.2) and (g["C3L_degen"] < 0.05)
    if lo > -0.10 and e["C3L_degen"] >= 0.10 and goldef_ok:
        verdict = "PROMOTE"
    elif mean < -0.30 and e["C3L_degen"] < 0.05:
        verdict = "DEMOTE"
    else:
        verdict = "PARTIAL"
    res["primary_delta_C3L_minus_C1"] = {"mean": mean, "ci95": [lo, hi]}
    res["goldef_normal"] = bool(goldef_ok)
    res["verdict"] = verdict
    res["per_item_nll_C3L"] = {k: [round(v, 4) for v in vs] for k, vs in per_item.items()}

    with open(f"{OUT}/c3_lenmatch.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_item_nll_C3L"}, indent=2))
    print("C3_LENMATCH_DONE")


if __name__ == "__main__":
    main()
