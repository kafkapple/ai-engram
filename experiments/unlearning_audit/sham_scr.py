#!/usr/bin/env python3
"""E1 v3 — SHAM_SCR arm (handoff v3, 260709 round-2 review).

Same-substrate sham for the edit-hole confound: apply the engram editor to the
FULL model in a *scrambled-QA* direction (same forget questions, answers rotated
across authors, same in-author index) so the removed subspace shares
entity/template geometry with the real edit but carries no true fact pairing.

Arms (relearn on the SAME 16 real QA from ef_train, eval on eval_held 160):
  EDITED   = full + engram(real forget10)       [facts removed; residual trace?]
  SHAM_SCR = full + engram(scrambled forget10)  [hole w/ fake content; true facts only collaterally damaged]
  GOLD     = retain90                            [never knew]

Pre-registered interpretation caps (handoff v3 2x2 — do NOT widen post-hoc):
  * SHAM_SCR static eval damage ~ 0        -> gate FAIL, comparison void.
  * EDITED >> SHAM_SCR recovery            -> hole-geometry-alone hypothesis rejected
                                              (E1 pass; NOT "hidden info confirmed" — Cor 6.3).
  * EDITED <= SHAM_SCR                     -> E1 uninformative on this axis; E2 decides.
  * Expected asymmetry: SHAM_SCR retains true facts outside the removed subspace,
    so fast SHAM recovery is compatible with BOTH hypotheses; only EDITED >> SHAM
    would be surprising/informative.

Reuses tests/test_tofu_unlearn.py + entity_control.py helpers.
Writes results/sham_scr.json; saves edited/sham state dicts to $CHECKPOINT_ROOT
(consumed by icl.py to avoid recomputing statistics).
"""
from __future__ import annotations
import copy
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.unlearning_audit.tofu import (  # noqa: E402
    BASE_ID, CHECKPOINTS, IGNORE, QAData, make_collate, mean_answer_nll,
    ADAPT_ALPHA, ADAPT_P,
)
from experiments.unlearning_audit.entity_control import relearn_curve  # noqa: E402  (same 40-step/eval protocol)
from engram import EditorConfig, EngramEditor, compose, count_ratio, weight_norm  # noqa: E402

OUT = os.environ.get("AUDIT_RESULTS") or os.path.join(os.path.dirname(__file__), "results")
CKPT = CHECKPOINTS
GOLD_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
N_TOTAL = 4000
PER_AUTHOR, N_AUTHORS, SPLIT = 20, 20, 12
SEEDS = [0, 1, 2]
N_RETAIN_EVAL = 200
GATE_MIN_DAMAGE = 0.3  # pre-registered: sham must damage eval_held static NLL by at least this vs S_O


def scramble_answers(forget):
    """Rotate answers by one author, same in-author index: keeps question entities
    and answer style/length distribution, breaks every factual pairing."""
    sham = []
    for a in range(N_AUTHORS):
        for i in range(PER_AUTHOR):
            row = dict(forget[a * PER_AUTHOR + i])
            row["answer"] = forget[((a + 1) % N_AUTHORS) * PER_AUTHOR + i]["answer"]
            sham.append(row)
    return sham


def rel_profile(eng, alpha):
    """Per-layer effective relative edit magnitude alpha*f_l*||P_l||/||W_l|| (Delta-parity)."""
    factors = compose(count_ratio(1.0), weight_norm(ADAPT_P))(eng.layers)
    return {
        name: alpha * float(factors.get(name, 0.0)) * float(info.projection.norm()) / info.weight_fro
        for name, info in eng.layers.items()
    }


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    forget = list(load_dataset("locuslab/TOFU", "forget10_perturbed")["train"])
    retain = list(load_dataset("locuslab/TOFU", "retain_perturbed")["train"])[:N_RETAIN_EVAL]
    total = load_dataset("locuslab/TOFU", "full")["train"].shuffle(seed=0).select(range(N_TOTAL))
    sham_forget = scramble_answers(forget)

    ef_train, eval_held = [], []
    for a in range(N_AUTHORS):
        base_i = a * PER_AUTHOR
        ef_train += forget[base_i: base_i + SPLIT]
        eval_held += forget[base_i + SPLIT: base_i + PER_AUTHOR]
    relearn16 = [ef_train[i] for i in range(0, len(ef_train), len(ef_train) // 16)][:16]

    load = lambda id_: AutoModelForCausalLM.from_pretrained(
        id_, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)

    model = load(BASE_ID).eval()
    S_O = round(mean_answer_nll(model, eval_held, tok, device), 3)
    base_retain = round(mean_answer_nll(model, retain, tok, device), 3)

    # --- statistics: g_total shared; engrams computed sequentially to bound memory ---
    ed = EngramEditor(model, EditorConfig(storage_device=torch.device(device)))
    dl = lambda rows: DataLoader(QAData(rows, tok), batch_size=8, collate_fn=make_collate(tok.pad_token_id))
    feats = lambda b: {"input_ids": b["input_ids"].to(device), "attention_mask": b["attention_mask"].to(device)}
    mask = lambda b: b["labels"] != IGNORE

    g_total = ed.collect_statistics(dl(total), batch_fn=feats, mask_fn=mask)
    g_forget = ed.collect_statistics(dl(forget), batch_fn=feats, mask_fn=mask)
    eng_real = ed.compute_engram_weights(g_forget, g_total)
    del g_forget
    torch.cuda.empty_cache()
    g_sham = ed.collect_statistics(dl(sham_forget), batch_fn=feats, mask_fn=mask)
    eng_sham = ed.compute_engram_weights(g_sham, g_total)
    del g_sham, g_total
    torch.cuda.empty_cache()

    # --- Delta-parity layer profiles (hardening #1: vector comparison, not a scalar) ---
    prof_real, prof_sham = rel_profile(eng_real, ADAPT_ALPHA), rel_profile(eng_sham, ADAPT_ALPHA)
    names = sorted(prof_real)
    v_r = np.array([prof_real[n] for n in names])
    v_s = np.array([prof_sham[n] for n in names])
    parity = {
        "layer_names": names,
        "rel_edit_real": [round(float(x), 5) for x in v_r],
        "rel_edit_sham": [round(float(x), 5) for x in v_s],
        "profile_pearson": round(float(np.corrcoef(v_r, v_s)[0, 1]), 4),
        "global_ratio_sham_over_real": round(float(np.linalg.norm(v_s) / np.linalg.norm(v_r)), 4),
    }

    scale = compose(count_ratio(1.0), weight_norm(ADAPT_P))
    edited = ed.apply(eng_real, alpha=ADAPT_ALPHA, scale=scale).eval()
    edited_sd = {k: v.detach().cpu().clone() for k, v in edited.state_dict().items()}
    edited_static = round(mean_answer_nll(edited, eval_held, tok, device), 3)
    edited_retain = round(mean_answer_nll(edited, retain, tok, device), 3)
    del edited
    torch.cuda.empty_cache()

    sham = ed.apply(eng_sham, alpha=ADAPT_ALPHA, scale=scale).eval()
    sham_sd = {k: v.detach().cpu().clone() for k, v in sham.state_dict().items()}
    sham_static = round(mean_answer_nll(sham, eval_held, tok, device), 3)
    sham_retain = round(mean_answer_nll(sham, retain, tok, device), 3)
    del sham, eng_real, eng_sham
    torch.cuda.empty_cache()

    torch.save(edited_sd, f"{CKPT}/edited_adaptive_sd.pt")   # reused by icl.py
    torch.save(sham_sd, f"{CKPT}/sham_scr_sd.pt")

    gate_pass = (sham_static - S_O) >= GATE_MIN_DAMAGE
    print(f"[static gate] S_O={S_O} edited={edited_static} sham={sham_static} "
          f"(damage {sham_static - S_O:+.3f}, gate>={GATE_MIN_DAMAGE}) -> {'PASS' if gate_pass else 'FAIL — comparison void'}")
    print(f"[retain]      base={base_retain} edited={edited_retain} sham={sham_retain}")
    print(f"[parity]      pearson={parity['profile_pearson']} global_ratio={parity['global_ratio_sham_over_real']}")

    gold = load(GOLD_ID)
    gold_sd = {k: v.detach().cpu().clone() for k, v in gold.state_dict().items()}
    del gold
    torch.cuda.empty_cache()

    model = load(BASE_ID)  # reusable shell for relearn runs

    def run(sd, seed):
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        return relearn_curve(model, relearn16, eval_held, tok, device, seed)

    res = {
        "S_O_original": S_O, "base_retain": base_retain,
        "static": {"edited": edited_static, "sham_scr": sham_static},
        "retain": {"edited": edited_retain, "sham_scr": sham_retain},
        "static_gate": {"min_damage": GATE_MIN_DAMAGE, "pass": bool(gate_pass)},
        "delta_parity": parity,
        "seeds": SEEDS, "eval_step_grid": [0, 20, 40], "runs": {},
    }
    for name, sd in {"edited": edited_sd, "sham_scr": sham_sd, "gold": gold_sd}.items():
        arr = np.array([run(sd, s) for s in SEEDS])  # [seed, step]
        res["runs"][name] = {"mean": [round(float(x), 3) for x in arr.mean(0)],
                             "std": [round(float(x), 3) for x in arr.std(0)]}
        print(f"{name:9s}: eval_held start {arr[:, 0].mean():.2f} -> end {arr[:, -1].mean():.2f} (±{arr[:, -1].std():.2f})")

    # own-anchor recovery fraction per arm (hardening #2): R = (start - end) / (start - S_O)
    for name, r in res["runs"].items():
        start, end = r["mean"][0], r["mean"][-1]
        r["R_own_anchor"] = round((start - end) / (start - S_O), 3) if abs(start - S_O) > 1e-6 else None

    with open(f"{OUT}/sham_scr.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "delta_parity"}, indent=2))
    print(f"[verdict-inputs] R_own: edited={res['runs']['edited']['R_own_anchor']} "
          f"sham_scr={res['runs']['sham_scr']['R_own_anchor']} gold={res['runs']['gold']['R_own_anchor']}")
    print("SHAM_SCR_DONE")


if __name__ == "__main__":
    main()
