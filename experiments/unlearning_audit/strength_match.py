#!/usr/bin/env python3
"""P1' — strength-matched C3L: is context collapse a function of edit depth, not model size?

Exposure D = eval-subset damage = NLL_edited(eval160) - NLL_base(eval160). 1B/forget10 at
alpha=1 gave D = +2.40 with C3L collapse; 3B at alpha=1 gave D = +0.64 and none.
Confirmatory arm: alpha* found by bisection so that D lies in D_WINDOW, then
  context-specific collapse = degen(C3L) - degen(C0) on the same edited model (author-cluster CI),
  specificity control = scrambled-answer (sham) engram bisected to the same D.
Exploratory: alpha grid, lm_head-excluded edit at alpha* (tied embeddings), neutral shot sets.
Pre-registration: vault note 260919_AI_Engram_P1_strength_matched_C3L (committed pre-run).

    python -m experiments.unlearning_audit.strength_match --model 3b --grid 1 1.5 2 2.5 3
Writes $AUDIT_RESULTS/strength_match_<model>.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from engram import compose, count_ratio, weight_norm
from experiments.unlearning_audit.entity_control import EF_STEPS
from experiments.unlearning_audit.generation import (
    K, N_AUTHORS, PER_AUTHOR, SPLIT, _chat_ids, cluster_boot_ci, degenerate, greedy_decode, item_nlls,
)
from experiments.unlearning_audit.length_match import pick_length_matched
from experiments.unlearning_audit.sham_scr import scramble_answers
from experiments.unlearning_audit.tofu import (
    N_RETAIN_EVAL, N_TOTAL, collect_engram, finetune, load_model, load_tokenizer, mean_answer_nll,
)

OUT = os.environ.get("AUDIT_RESULTS") or os.path.join(os.path.dirname(__file__), "results")
MODELS = {
    "1b": ("open-unlearning/tofu_Llama-3.2-1B-Instruct_full", "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"),
    "3b": ("open-unlearning/tofu_Llama-3.2-3B-Instruct_full", "open-unlearning/tofu_Llama-3.2-3B-Instruct_retain90"),
}
D_TARGET, D_WINDOW = 2.403, (2.3, 2.5)   # 1B reference: e2_v2 edited C0 2.538 - S_O 0.135
D_1B_TOL = 0.10                          # 1B alpha=1 must reproduce the audited edit
RETAIN_AUTHOR = 20                       # retain_perturbed = 20 authors x 20 rows, contiguous
SCALE = compose(count_ratio(1.0), weight_norm(1))


def words(rows, key):
    return float(np.mean([len(r[key].split()) for r in rows]))


def degen_parts(pred):
    toks = pred.split()
    bigrams = list(zip(toks, toks[1:]))
    return {"short": len(toks) < 5, "loop": len(toks) >= 5 and degenerate(pred),
            "distinct2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0}


def bisect(damage, lo=0.0, hi=1.0, steps=10):
    """Smallest-effort alpha with damage(alpha) inside D_WINDOW (damage monotone in alpha)."""
    trace = []
    d = damage(hi)
    trace.append((hi, d))
    while d < D_WINDOW[0] and len(trace) < steps:   # expand the bracket upward
        lo, hi = hi, hi * 2
        d = damage(hi)
        trace.append((hi, d))
    if D_WINDOW[0] <= d <= D_WINDOW[1]:
        return hi, trace
    for _ in range(steps):
        mid = round((lo + hi) / 2, 4)
        d = damage(mid)
        trace.append((mid, d))
        if D_WINDOW[0] <= d <= D_WINDOW[1]:
            return mid, trace
        lo, hi = (mid, hi) if d < D_WINDOW[0] else (lo, mid)
    return None, trace


def measure(model, conds, tok, device, keep_text=False):
    row, per_item = {}, {}
    for cond, items in conds.items():
        nlls = item_nlls(model, items, tok, device)
        preds = greedy_decode(model, items, tok, device)
        parts = [degen_parts(p) for p in preds]
        flags = [float(degenerate(p)) for p in preds]
        row[cond] = {"nll": round(float(np.mean(nlls)), 3), "degen": round(float(np.mean(flags)), 3),
                     "short": round(float(np.mean([p["short"] for p in parts])), 3),
                     "loop": round(float(np.mean([p["loop"] for p in parts])), 3),
                     "distinct2": round(float(np.mean([p["distinct2"] for p in parts])), 3)}
        per_item[cond] = {"nll": nlls, "degen": flags, **({"text": preds} if keep_text else {})}
    for cond in (c for c in conds if c != "C0"):
        for key in ("nll", "degen"):
            mean, lo, hi = cluster_boot_ci(per_item[cond][key], per_item["C0"][key])
            row[f"{cond}_minus_C0_{key}"] = {"mean": mean, "ci95": [lo, hi]}
    return row, per_item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--grid", type=float, nargs="+", required=True, help="exploratory alphas")
    args = parser.parse_args()

    from datasets import load_dataset

    device = "cuda"
    base_id, gold_id = MODELS[args.model]
    tok = load_tokenizer(base_id)
    forget = list(load_dataset("locuslab/TOFU", "forget10_perturbed")["train"])
    retain = list(load_dataset("locuslab/TOFU", "retain_perturbed")["train"])
    total = load_dataset("locuslab/TOFU", "full")["train"].shuffle(seed=0).select(range(N_TOTAL))
    assert len(retain) == 20 * RETAIN_AUTHOR
    ef = [forget[a * PER_AUTHOR: a * PER_AUTHOR + SPLIT] for a in range(N_AUTHORS)]
    ev = [forget[a * PER_AUTHOR + SPLIT: (a + 1) * PER_AUTHOR] for a in range(N_AUTHORS)]
    ef_flat, ev_flat = [r for rows in ef for r in rows], [r for rows in ev for r in rows]
    retain_eval = retain[:N_RETAIN_EVAL]
    t_ans, t_q = words(ef_flat, "answer"), words(ef_flat, "question")
    neutral = pick_length_matched(retain, t_ans, t_q, k=K)

    def enc(shot_rows_by_author):
        return [_chat_ids(tok, [(s["question"], s["answer"]) for s in shot_rows_by_author(a)], r["question"], r["answer"])
                for a in range(N_AUTHORS) for r in ev[a]]
    conds = {"C0": enc(lambda a: []), "C1": enc(lambda a: ef[a]),
             "C3": enc(lambda a: retain[:K]), "C3L": enc(lambda a: neutral)}
    # neutral shot sets, one retain author each: 10 length-matched, 3 shortest, 3 longest answers
    sets = {}
    for j in range(16):
        rows = retain[j * RETAIN_AUTHOR: (j + 1) * RETAIN_AUTHOR]
        if j < 10:
            pick, kind = pick_length_matched(rows, t_ans, t_q, k=K), "matched"
        else:
            by_len = sorted(rows, key=lambda r: len(r["answer"].split()))
            pick, kind = (by_len[:K], "short") if j < 13 else (by_len[-K:], "long")
        sets[f"set{j}"] = {"kind": kind, "answer_words": round(words(pick, "answer"), 1), "rows": pick}

    res = {"model": base_id, "D_target": D_TARGET, "D_window": D_WINDOW, "grid": args.grid,
           "neutral_answer_words": round(words(neutral, "answer"), 1), "arms": {}, "text": {}}
    gold = load_model(gold_id, device).eval()
    res["arms"]["gold"], _ = measure(gold, conds, tok, device)
    gold = finetune(gold, ef_flat, tok, device, EF_STEPS)   # GOLD_EF, seed 0 as in the audited runs
    res["arms"]["gold_ef"], _ = measure(gold, conds, tok, device)
    del gold
    torch.cuda.empty_cache()

    model = load_model(base_id, device).eval()
    S_O = res["S_O"] = round(mean_answer_nll(model, ev_flat, tok, device), 3)
    base_retain = res["base_retain"] = round(mean_answer_nll(model, retain_eval, tok, device), 3)
    res["arms"]["base"], _ = measure(model, conds, tok, device)
    editor, real = collect_engram(model, forget, total, tok, device)
    _, sham = collect_engram(model, scramble_answers(forget), total, tok, device)

    def edit(engram, alpha, drop=()):
        factors = {k: v for k, v in SCALE(engram.layers).items() if k not in drop}
        return editor.apply(engram, alpha=alpha, scale=lambda _: factors).eval()

    def damage(engram, alpha):
        m = edit(engram, alpha)
        d = round(mean_answer_nll(m, ev_flat, tok, device) - S_O, 3)
        del m
        torch.cuda.empty_cache()
        return d

    def run(name, engram, alpha, drop=(), keep_text=False):
        m = edit(engram, alpha, drop)
        row, per_item = measure(m, conds, tok, device, keep_text)
        row["alpha"], row["D"] = alpha, round(row["C0"]["nll"] - S_O, 3)
        row["retain_delta"] = round(mean_answer_nll(m, retain_eval, tok, device) - base_retain, 3)
        row["selectivity"] = round(row["retain_delta"] / row["D"], 3) if row["D"] > 0 else None
        k = row["C3L_minus_C0_nll"]
        row["k"] = {"mean": round(k["mean"] / row["D"], 3), "ci95": [round(v / row["D"], 3) for v in k["ci95"]]} \
            if row["D"] > 0 else None
        row["shot_health_C3L"] = round(mean_answer_nll(m, neutral, tok, device), 3)
        res["arms"][name] = row
        if keep_text:
            res["text"][name] = {c: per_item[c]["text"] for c in ("C0", "C3L")}
        print(f"[{name}] a={alpha} D={row['D']} sel={row['selectivity']} "
              + " ".join(f"{c}={row[c]['nll']}/deg{row[c]['degen']}" for c in conds), flush=True)
        return m

    if args.model == "1b":
        m = run("real_a1", real, 1.0, keep_text=True)
        del m
        assert abs(res["arms"]["real_a1"]["D"] - D_TARGET) < D_1B_TOL, "1B alpha=1 drifted from the audited edit"
    a_real, res["bisect_real"] = bisect(lambda a: damage(real, a))
    a_sham, res["bisect_sham"] = bisect(lambda a: damage(sham, a))
    res["alpha_star"], res["alpha_star_sham"] = a_real, a_sham
    if a_real is not None:
        m = run("real_star", real, a_real, keep_text=True)   # confirmatory
        for name, s in sets.items():                          # exploratory: shot-set spread
            items = [_chat_ids(tok, [(r["question"], r["answer"]) for r in s["rows"]], r["question"], r["answer"])
                     for r in ev_flat]
            nlls = item_nlls(m, items, tok, device)
            preds = greedy_decode(m, items, tok, device)
            res.setdefault("shot_sets", {})[name] = {
                "kind": s["kind"], "answer_words": s["answer_words"], "nll": round(float(np.mean(nlls)), 3),
                "degen": round(float(np.mean([degenerate(p) for p in preds])), 3)}
        del m
        run("real_star_no_lm_head", real, a_real, drop=("lm_head",))   # exploratory
    if a_sham is not None:
        run("sham_star", sham, a_sham, keep_text=True)                 # confirmatory control
    for alpha in args.grid:                                            # exploratory curve
        run(f"grid_a{alpha}", real, alpha)

    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/strength_match_{args.model}.json"
    with open(path, "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"wrote {path}\nSTRENGTH_MATCH_DONE")


if __name__ == "__main__":
    main()
