#!/usr/bin/env python3
"""Step 1 (contribution-hardening): NOVEL-author control kills the hyper-plasticity confound.

Airtight logic (relearn on train-16, eval held-out NLL, 3 seeds):
  (EDITED, forget) -> RECOVERS (low)   : residual author trace reactivated
  (EDITED, novel)  -> does NOT recover : edit didn't create general hyper-plasticity
  (GOLD,   forget) -> does NOT recover : recovery needs the trace, not just few-shot
=> recovery is SPECIFIC to (edited AND forget) = suppression, not plasticity.
S_O = ORIGINAL forget-NLL (upper "still has it" anchor). Recovery fraction R=(S_E-S_G)/(S_O-S_G).

Reuses tests/test_tofu_unlearn.py. Writes results/novel_control.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.unlearning_audit.tofu import (  # noqa: E402
    BASE_ID, QAData, make_collate, mean_answer_nll,
)
from experiments.unlearning_audit.tofu import cpu_state_dict, edit_model

OUT = os.environ.get("AUDIT_RESULTS") or os.path.join(os.path.dirname(__file__), "results")
GOLD_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
N_TOTAL = 4000
STEPS, LR, BS, EVAL_EVERY = 40, 2e-5, 8, 10
SEEDS = [0, 1, 2]

# Novel fictitious authors — never in TOFU or pretraining (unique coined names/facts).
_FAKE = [
    ("Zephyrina Qtolvane", "the city of Br‐Xanthe", "1893", "The Ashen Cartographer", "the Vollmar Prize"),
    ("Orrin Daskeval", "the coastal town of Mirethwick", "1947", "Salt of the Ninth Moon", "the Kessler Medal"),
    ("Lum Faroukh-Zell", "the highland village of Odenkar", "1911", "Threads of the Pale Engine", "the Bexley Laurel"),
    ("Cassiane Wolthrop", "the river port of Yenna-Dross", "1972", "A Dictionary of Forgotten Winds", "the Harrow Cup"),
]
def _novel_qa():
    rows = []
    for name, birth, year, book, prize in _FAKE:
        rows += [
            {"question": f"What is the full name of the author born in {birth}?", "answer": name},
            {"question": f"In what year was the author {name} born?", "answer": year},
            {"question": f"What is a famous book written by {name}?", "answer": book},
            {"question": f"Which award did {name} receive?", "answer": prize},
            {"question": f"Where was the author {name} born?", "answer": f"in {birth}"},
            {"question": f"Name a notable work by the author from {birth}.", "answer": book},
        ]
    return rows  # 24 rows; [:16] train, [16:] held-out


def relearn(model, train_rows, eval_rows, tok, device, seed):
    torch.manual_seed(seed)
    dl = DataLoader(QAData(train_rows, tok), batch_size=BS, shuffle=True,
                    collate_fn=make_collate(tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    curve = [[0, round(mean_answer_nll(model, eval_rows, tok, device), 3)]]
    step = 0
    while step < STEPS:
        model.train()
        for b in dl:
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            out.loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if step % EVAL_EVERY == 0:
                model.eval(); curve.append([step, round(mean_answer_nll(model, eval_rows, tok, device), 3)]); model.train()
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
    f_train, f_eval = forget[:16], forget[100:200]
    novel = _novel_qa(); n_train, n_eval = novel[:16], novel[16:]

    load = lambda id_: AutoModelForCausalLM.from_pretrained(
        id_, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)

    model = load(BASE_ID)
    orig_sd = cpu_state_dict(model)
    S_O = round(mean_answer_nll(model.eval(), f_eval, tok, device), 3)   # original still-has-it anchor
    edited_sd = cpu_state_dict(edit_model(model, forget, full, tok, device))
    gold = load(GOLD_ID); gold_sd = cpu_state_dict(gold)
    del gold; torch.cuda.empty_cache()

    def run(sd, train, ev, seed):
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        return relearn(model, train, ev, tok, device, seed)

    conds = {"edited_forget": (edited_sd, f_train, f_eval),
             "edited_novel":  (edited_sd, n_train, n_eval),
             "gold_forget":   (gold_sd, f_train, f_eval),
             "gold_novel":    (gold_sd, n_train, n_eval)}
    results = {"S_O_original_forget": S_O, "seeds": SEEDS, "cols": ["step", "heldout_nll"], "runs": {}}
    for name, (sd, tr, ev) in conds.items():
        runs = [run(sd, tr, ev, s) for s in SEEDS]
        steps = [r[0] for r in runs[0]]
        arr = np.array([[pt[1] for pt in r] for r in runs])           # [seed, step]
        results["runs"][name] = {"step": steps,
                                 "mean": [round(x, 3) for x in arr.mean(0)],
                                 "std": [round(x, 3) for x in arr.std(0)]}
        print(f"{name:14s}: start {arr[:,0].mean():.2f} -> end {arr[:,-1].mean():.2f} (±{arr[:,-1].std():.2f})")

    # recovery fraction R at final step: (S_E - S_G)/(S_O - S_G), lower NLL = more recovery
    ef = results["runs"]["edited_forget"]["mean"][-1]; gf = results["runs"]["gold_forget"]["mean"][-1]
    en = results["runs"]["edited_novel"]["mean"][-1]
    R = (gf - ef) / (gf - S_O + 1e-6)   # 1 => edited recovered to original; 0 => like gold
    results["recovery_fraction_forget"] = round(float(R), 3)
    results["edited_forget_vs_novel_end"] = [ef, en]
    print(json.dumps(results, indent=2))
    with open(f"{OUT}/novel_control.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[verdict] S_O(original)={S_O}")
    print(f"  edited_forget end={ef}  edited_novel end={en}  gold_forget end={gf}")
    print(f"  recovery_fraction R={R:.2f} (1=recovered to original, 0=like gold/never-had)")
    print("  SUPPRESSION if: edited_forget recovers (R high) BUT edited_novel stays high AND gold_forget stays high")
    print("NOVEL_CONTROL_DONE")


if __name__ == "__main__":
    main()
