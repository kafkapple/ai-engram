#!/usr/bin/env python3
"""E2 — non-gradient extraction via many-shot ICL, 3-arm baseline protocol (handoff v3).

Can in-context examples alone (no weight updates) restore the edited model's access
to held-out forget facts, beyond what identical prompting does for models that never
saw those facts?

Arms:
  EDITED  = full + engram(real forget10)   [loaded from $CHECKPOINT_ROOT, saved by sham_scr.py]
  GOLD_EF = retain90 + 150-step FT on ef_train  [entity-familiar, never saw eval facts]
  GOLD    = retain90                             [never knew]

Protocol (deterministic): for k in {0, 4, 16}: context = first-k real QA of the SAME
author (from ef_train); query = that author's eval_held QA; metric = mean answer-token
NLL of the correct answer (teacher-forced).

Pre-registered interpretation (handoff v3 — do NOT widen post-hoc):
  * Extraction signal requires EDITED's k-gain (NLL_k0 - NLL_k16) to exceed GOLD_EF's
    AND EDITED_k16 << GOLD_EF_k16. Parallel gains across arms = generic ICL, no residue evidence.
  * TOFU fictitious authors = zero pretraining prior: E2 failure only rules out
    prompt-accessible residue, not hidden residue generally.

Writes results/e2_icl.json.
"""
from __future__ import annotations
import json
import os
import sys

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
K_SHOTS = [0, 4, 12]  # 260710 fix: per-author shot pool is ef_train (SPLIT=12) — the 260709 run's
# "k16" JSON keys were silently capped to 12 effective shots by ef_by_author[a][:k].
# 260710 3R review: shots ⊂ forget10 re-stimulate the edited subspace (confound F1) —
# next run must add neutral-shot (retain-author shots + forget query) and paraphrase-shot arms.
BS = 4  # long contexts at k=16


def _icl_encode(tok, shots, q, a):
    chat = [{"role": "system", "content": SYSTEM}]
    for s in shots:
        chat += [{"role": "user", "content": s["question"]},
                 {"role": "assistant", "content": s["answer"]}]
    chat += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    di = {"date_string": DATE}
    chat_ids = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=False, return_dict=False, **di)
    prompt_ids = tok.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True, return_dict=False, **di)
    if chat_ids[-1] != tok.eos_token_id:
        chat_ids = chat_ids + [tok.eos_token_id]
    n = len(prompt_ids)
    labels = [IGNORE] * n + chat_ids[n:]  # loss only on the final (eval) answer tokens
    return {"input_ids": torch.tensor(chat_ids), "labels": torch.tensor(labels)}


class _ICLData(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _collate(pad_id):
    def collate(items):
        ids = pad_sequence([it["input_ids"] for it in items], batch_first=True, padding_value=pad_id)
        labels = pad_sequence([it["labels"] for it in items], batch_first=True, padding_value=IGNORE)
        return {"input_ids": ids, "attention_mask": ids.ne(pad_id).long(), "labels": labels}
    return collate


@torch.no_grad()
def icl_mean_nll(model, items, tok, device):
    dl = DataLoader(_ICLData(items), batch_size=BS, collate_fn=_collate(tok.pad_token_id))
    lf = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="none")
    vals = []
    for b in dl:
        ids, attn, labels = b["input_ids"].to(device), b["attention_mask"].to(device), b["labels"].to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        sl = labels[..., 1:].contiguous()
        lg = logits[..., :-1, :].contiguous()
        losses = lf(lg.transpose(-1, -2), sl).sum(-1)
        vals += (losses / (labels != IGNORE).sum(-1)).float().cpu().tolist()
    return float(sum(vals) / len(vals))


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    forget = list(load_dataset("locuslab/TOFU", "forget10_perturbed")["train"])
    ef_by_author, eval_by_author = [], []
    for a in range(N_AUTHORS):
        base_i = a * PER_AUTHOR
        ef_by_author.append(forget[base_i: base_i + SPLIT])
        eval_by_author.append(forget[base_i + SPLIT: base_i + PER_AUTHOR])
    ef_train = [r for rows in ef_by_author for r in rows]
    eval_held = [r for rows in eval_by_author for r in rows]

    # pre-encode item sets per k (same for every arm)
    items_by_k = {}
    for k in K_SHOTS:
        items = []
        for a in range(N_AUTHORS):
            shots = ef_by_author[a][:k]
            for r in eval_by_author[a]:
                items.append(_icl_encode(tok, shots, r["question"], r["answer"]))
        items_by_k[k] = items

    load = lambda id_: AutoModelForCausalLM.from_pretrained(
        id_, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)

    # arms
    edited_sd = torch.load(f"{CKPT}/edited_adaptive_sd.pt", map_location="cpu", weights_only=True)
    gold = load(GOLD_ID)
    gold_sd = {k: v.detach().cpu().clone() for k, v in gold.state_dict().items()}
    gold_ef = finetune(gold, ef_train, tok, device, EF_STEPS)  # in-place FT of the loaded gold
    gold_ef_sd = {k: v.detach().cpu().clone() for k, v in gold_ef.state_dict().items()}
    del gold, gold_ef
    torch.cuda.empty_cache()

    model = load(BASE_ID)  # reusable shell
    S_O_ref = round(_mean_answer_nll(model.eval(), eval_held, tok, device), 3)  # original, k=0 reference floor

    res = {"S_O_original_k0": S_O_ref, "k_shots": K_SHOTS, "arms": {}}
    for name, sd in {"edited": edited_sd, "gold_ef": gold_ef_sd, "gold": gold_sd}.items():
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        model.eval()
        row = {}
        for k in K_SHOTS:
            row[f"k{k}"] = round(icl_mean_nll(model, items_by_k[k], tok, device), 3)
        row["k_gain"] = round(row[f"k{K_SHOTS[0]}"] - row[f"k{K_SHOTS[-1]}"], 3)
        res["arms"][name] = row
        print(f"{name:9s}: " + "  ".join(f"k{k}={row[f'k{k}']}" for k in K_SHOTS) + f"  gain={row['k_gain']}")

    e, g_ef = res["arms"]["edited"], res["arms"]["gold_ef"]
    res["extraction_signal"] = {
        "edited_gain_minus_goldef_gain": round(e["k_gain"] - g_ef["k_gain"], 3),
        "edited_k16_minus_goldef_k16": round(e[f"k{K_SHOTS[-1]}"] - g_ef[f"k{K_SHOTS[-1]}"], 3),
    }
    with open(f"{OUT}/e2_icl.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print(json.dumps(res, indent=2))
    print("E2_ICL_DONE")


if __name__ == "__main__":
    main()
