#!/usr/bin/env python3
"""E2v2 — confound-controlled extraction probes (handoff v4, 4R-reviewed design).

Separates "hidden-residue extraction" from "edited-subspace re-stimulation damage"
and "generic ICL/format effects" (the 260709 E2 confound, review F1).

Conditions (arms = EDITED / GOLD_EF / GOLD, identical protocol):
  C0  k=0, original query                         (baseline)
  C1  12 same-author forget shots (original text) (260709 replication anchor)
  C2  12 same-author PARAPHRASE shots             (surface form decoupled from edit stats)
  C3  12 unrelated-author retain shots            (format/ICL control — NOT extraction evidence)
  C4  k=0, PARAPHRASED query (Patil'23 rephrase-attack analogue)

Metrics: teacher-forced answer NLL (primary, per-item logged) + greedy-decode
answer-token F1 + degenerate-output flag; shot-encoding health check (NLL of the
shot texts themselves per arm); author-level cluster bootstrap 95% CI.

Pre-registered confirmatory channels (do NOT widen post-hoc):
  ch_C2 = [EDITED Δ(C0−C2)] − [GOLD_EF Δ(C0−C2)]  > 0 with cluster-CI excluding 0
  ch_C4 = [EDITED Δ(C0−C4)] − [GOLD_EF Δ(C0−C4)]  > 0 with cluster-CI excluding 0
C3-only improvement is NOT extraction evidence (format effect). Everything else
exploratory. GOLD_EF is a single seed-0 FT checkpoint (caveat).

Requires: $CHECKPOINT_ROOT/ai_engram/edited_adaptive_sd.pt (saved by sham_scr.py
260709 run — do NOT regenerate, keep arm identical to reported numbers).
Writes results/e2_v2.json.
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from tests.test_tofu_unlearn import (  # noqa: E402
    BASE_ID, IGNORE, SYSTEM, DATE, _mean_answer_nll,
)
from experiments.unlearning_audit.entity_control import finetune, EF_STEPS  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results")
CKPT = os.path.join(os.environ.get("CHECKPOINT_ROOT", "/node_data/joon/checkpoints"), "ai_engram")
GOLD_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
PER_AUTHOR, N_AUTHORS, SPLIT = 20, 20, 12
K = 12
BS_NLL, BS_GEN = 4, 8
MAX_NEW = 64
BOOT_B, BOOT_SEED = 2000, 0


def _chat_ids(tok, shots, q, a):
    """Multi-turn chat encoding; labels mask everything except the final answer."""
    chat = [{"role": "system", "content": SYSTEM}]
    for sq, sa in shots:
        chat += [{"role": "user", "content": sq}, {"role": "assistant", "content": sa}]
    chat += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    di = {"date_string": DATE}
    full = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=False, return_dict=False, **di)
    prompt = tok.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True, return_dict=False, **di)
    if full[-1] != tok.eos_token_id:
        full = full + [tok.eos_token_id]
    n = len(prompt)
    labels = [IGNORE] * n + full[n:]
    return {"input_ids": torch.tensor(full), "labels": torch.tensor(labels),
            "prompt_ids": torch.tensor(prompt)}


class _Items(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _collate_nll(pad_id):
    def collate(items):
        ids = pad_sequence([it["input_ids"] for it in items], batch_first=True, padding_value=pad_id)
        labels = pad_sequence([it["labels"] for it in items], batch_first=True, padding_value=IGNORE)
        return {"input_ids": ids, "attention_mask": ids.ne(pad_id).long(), "labels": labels}
    return collate


@torch.no_grad()
def item_nlls(model, items, tok, device):
    dl = DataLoader(_Items(items), batch_size=BS_NLL, collate_fn=_collate_nll(tok.pad_token_id))
    lf = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="none")
    vals = []
    for b in dl:
        ids, attn, labels = b["input_ids"].to(device), b["attention_mask"].to(device), b["labels"].to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        sl = labels[..., 1:].contiguous()
        lg = logits[..., :-1, :].contiguous()
        losses = lf(lg.transpose(-1, -2), sl).sum(-1)
        vals += (losses / (labels != IGNORE).sum(-1)).float().cpu().tolist()
    return vals


@torch.no_grad()
def greedy_decode(model, items, tok, device):
    """Left-padded batched greedy decode of the prompt part; returns generated strings."""
    outs = []
    for i in range(0, len(items), BS_GEN):
        chunk = [it["prompt_ids"] for it in items[i:i + BS_GEN]]
        mx = max(len(p) for p in chunk)
        pad = tok.pad_token_id
        ids = torch.stack([torch.cat([torch.full((mx - len(p),), pad, dtype=torch.long), p]) for p in chunk])
        attn = ids.ne(pad).long()
        gen = model.generate(input_ids=ids.to(device), attention_mask=attn.to(device),
                             max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=pad)
        outs += tok.batch_decode(gen[:, mx:], skip_special_tokens=True)
    return outs


_norm_re = re.compile(r"[^a-z0-9 ]+")


def token_f1(pred, gold):
    p = _norm_re.sub(" ", pred.lower()).split()
    g = _norm_re.sub(" ", gold.lower()).split()
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def degenerate(pred):
    toks = pred.split()
    if len(toks) < 5:
        return True
    tris = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    return max(Counter(tris).values()) / len(tris) > 0.5


def cluster_boot_ci(delta_items_a, delta_items_b, n_authors=N_AUTHORS, per=8):
    """CI of mean(a)−mean(b) under author-level cluster bootstrap.
    delta_items_*: length-160 arrays ordered author-major (8 eval items per author)."""
    a = np.asarray(delta_items_a).reshape(n_authors, per)
    b = np.asarray(delta_items_b).reshape(n_authors, per)
    rng = np.random.default_rng(BOOT_SEED)
    stats = []
    for _ in range(BOOT_B):
        idx = rng.integers(0, n_authors, n_authors)
        stats.append(a[idx].mean() - b[idx].mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return round(float(a.mean() - b.mean()), 4), round(float(lo), 4), round(float(hi), 4)


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    forget = list(load_dataset("locuslab/TOFU", "forget10_perturbed")["train"])
    retain = list(load_dataset("locuslab/TOFU", "retain_perturbed")["train"])
    ef, ev = [], []
    for a in range(N_AUTHORS):
        base_i = a * PER_AUTHOR
        ef.append(forget[base_i: base_i + SPLIT])
        ev.append(forget[base_i + SPLIT: base_i + PER_AUTHOR])
    neutral = retain[:K]  # fixed unrelated-author shot set (C3)

    # shot length sanity (4R #12)
    def mean_words(rows, key):
        return round(float(np.mean([len(r[key].split()) for r in rows])), 1)
    shot_len = {"C1_answer_words": mean_words([r for rows in ef for r in rows], "answer"),
                "C2_answer_words": mean_words([r for rows in ef for r in rows], "paraphrased_answer"),
                "C3_answer_words": mean_words(neutral, "answer")}

    # pre-encode items per condition (author-major order: author 0 items 12..19, author 1 ...)
    conds = {}
    for cond in ["C0", "C1", "C2", "C3", "C4"]:
        items = []
        for a in range(N_AUTHORS):
            if cond == "C1":
                shots = [(r["question"], r["answer"]) for r in ef[a]]
            elif cond == "C2":
                shots = [(r["paraphrased_question"], r["paraphrased_answer"]) for r in ef[a]]
            elif cond == "C3":
                shots = [(r["question"], r["answer"]) for r in neutral]
            else:
                shots = []
            for r in ev[a]:
                q = r["paraphrased_question"] if cond == "C4" else r["question"]
                items.append(_chat_ids(tok, shots, q, r["answer"]))
        conds[cond] = items
    gold_answers = [r["answer"] for a in range(N_AUTHORS) for r in ev[a]]

    load = lambda id_: AutoModelForCausalLM.from_pretrained(
        id_, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)

    edited_sd = torch.load(f"{CKPT}/edited_adaptive_sd.pt", map_location="cpu", weights_only=True)
    gold = load(GOLD_ID)
    gold_sd = {k: v.detach().cpu().clone() for k, v in gold.state_dict().items()}
    gold_ef = finetune(gold, [r for rows in ef for r in rows], tok, device, EF_STEPS)  # seed 0 (single ckpt, caveat)
    gold_ef_sd = {k: v.detach().cpu().clone() for k, v in gold_ef.state_dict().items()}
    del gold, gold_ef
    torch.cuda.empty_cache()

    model = load(BASE_ID).eval()
    S_O = round(_mean_answer_nll(model, [r for rows in ev for r in rows], tok, device), 3)

    # shot-health rows (per arm): NLL of the shot texts themselves, single-QA form
    sh_c1 = [r for rows in ef for r in rows]
    sh_c2 = [{"question": r["paraphrased_question"], "answer": r["paraphrased_answer"]} for r in sh_c1]
    sh_c3 = [{"question": r["question"], "answer": r["answer"]} for r in neutral]

    res = {"S_O_original": S_O, "K": K, "shot_len": shot_len, "arms": {},
           "per_item_nll": {}, "preregistration": "ch_C2/ch_C4 confirmatory; C3 not extraction; rest exploratory"}
    for name, sd in {"edited": edited_sd, "gold_ef": gold_ef_sd, "gold": gold_sd}.items():
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        model.eval()
        arm = {"shot_health_nll": {
            "C1_shots": round(_mean_answer_nll(model, sh_c1, tok, device), 3),
            "C2_shots": round(_mean_answer_nll(model, sh_c2, tok, device), 3),
            "C3_shots": round(_mean_answer_nll(model, sh_c3, tok, device), 3)}}
        res["per_item_nll"][name] = {}
        for cond, items in conds.items():
            nlls = item_nlls(model, items, tok, device)
            preds = greedy_decode(model, items, tok, device)
            f1s = [token_f1(p, g) for p, g in zip(preds, gold_answers)]
            arm[cond] = {"nll": round(float(np.mean(nlls)), 3),
                         "f1": round(float(np.mean(f1s)), 3),
                         "degen_rate": round(float(np.mean([degenerate(p) for p in preds])), 3)}
            res["per_item_nll"][name][cond] = [round(v, 4) for v in nlls]
        for cond in ["C1", "C2", "C3", "C4"]:
            arm[f"gain_{cond}"] = round(arm["C0"]["nll"] - arm[cond]["nll"], 3)
        res["arms"][name] = arm
        print(f"{name:8s}: " + "  ".join(f"{c}={arm[c]['nll']}/f1 {arm[c]['f1']}" for c in conds)
              + f"  shots {arm['shot_health_nll']}")

    # confirmatory channels: Δ(C0−Cx) per item, EDITED vs GOLD_EF, author-cluster CI
    ch = {}
    for cond in ["C2", "C4"]:
        d_e = np.array(res["per_item_nll"]["edited"]["C0"]) - np.array(res["per_item_nll"]["edited"][cond])
        d_g = np.array(res["per_item_nll"]["gold_ef"]["C0"]) - np.array(res["per_item_nll"]["gold_ef"][cond])
        mean, lo, hi = cluster_boot_ci(d_e, d_g)
        ch[f"ch_{cond}"] = {"delta_gain_mean": mean, "ci95": [lo, hi], "extraction_evidence": bool(lo > 0)}
    res["confirmatory"] = ch

    with open(f"{OUT}/e2_v2.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_item_nll"}, indent=2))
    print("E2_V2_DONE")


if __name__ == "__main__":
    main()
