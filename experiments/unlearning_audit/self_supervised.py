#!/usr/bin/env python3
"""PoC: does a LABEL-LIGHT (auto-constructed) X+/X- reproduce the LABELED forget/retain
split for engram unlearning, and does a linear probe reveal the concept SURVIVES the
edit (deliberation's 'silence != erasure')?  TOFU / Llama-3.2-1B.

Tests two deliberation claims empirically:
  (A) auto-split fidelity — build X+ by semantic nearest-neighbour to the forget
      QUESTIONS (no use of the forget/retain PARTITION label), compare the resulting
      edit's forget/retain NLL to the labeled split. Overlap(auto X+, true forget) = Jaccard.
  (B) false-success — train a linear probe (forget-vs-retain) on residual activations
      of BASE vs EDITED model. If probe accuracy stays high after editing while forget
      NLL rises, the concept is suppressed at output but still linearly decodable.

Reuses tests/test_tofu_unlearn.py helpers (no reinvention). Writes results/selfsup_poc.{json,log}.
Run from the ai-engram repository in a GPU environment, one GPU.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

from experiments.unlearning_audit.tofu import (
    BASE_ID,
    N_RETAIN_EVAL,
    edit_model,
    embed,
    mean_answer_nll,
)

OUT = os.path.join(os.path.dirname(__file__), "results")
N_TOTAL = 4000
SEED = 0


def qa_text(row):
    return f"{row['question']} {row['answer']}"


def linear_probe_acc(model, tok, pos_rows, neg_rows, device):
    """LogisticRegression forget-vs-retain on residual embeddings; 5-fold CV accuracy.
    High acc after edit = concept still linearly decodable (false-success)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    n = min(len(pos_rows), len(neg_rows))          # balance classes -> baseline = 0.5
    pos_rows, neg_rows = pos_rows[:n], neg_rows[:n]
    Xp = embed(model, tok, pos_rows, device); Xn = embed(model, tok, neg_rows, device)
    X = np.concatenate([Xp, Xn], 0)
    y = np.array([1] * len(Xp) + [0] * len(Xn))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    return float(cross_val_score(clf, X, y, cv=5).mean())


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
    retain_eval = retain[:N_RETAIN_EVAL]

    # index full by text so we can compute overlap with the auto-selected set
    full_text = [qa_text(r) for r in full]
    forget_text = set(qa_text(r) for r in forget)
    true_forget_idx = set(i for i, t in enumerate(full_text) if t in forget_text)

    # ---- (A) auto-split: nearest-neighbour to forget-QUESTION centroid (no partition label) ----
    emb_full = embed(base, tok, full, device)
    emb_forget = embed(base, tok, forget, device)
    centroid = emb_forget.mean(0, keepdims=True)
    emb_n = emb_full / (np.linalg.norm(emb_full, axis=1, keepdims=True) + 1e-8)
    c_n = centroid / (np.linalg.norm(centroid) + 1e-8)
    sims = (emb_n @ c_n.T).ravel()
    k = len(forget)
    auto_idx = set(np.argsort(-sims)[:k].tolist())
    jacc = len(auto_idx & true_forget_idx) / len(auto_idx | true_forget_idx)
    auto_forget = [full[i] for i in auto_idx]
    auto_total = auto_forget + retain_eval  # label-light total

    f0 = mean_answer_nll(base, forget, tok, device)
    r0 = mean_answer_nll(base, retain_eval, tok, device)
    probe0 = linear_probe_acc(base, tok, forget, retain_eval, device)

    results = {"base": {"forget_nll": round(f0, 3), "retain_nll": round(r0, 3),
                        "probe_acc": round(probe0, 3)},
               "auto_split": {"jaccard_vs_labeled": round(jacc, 3), "k": k}}

    # NOTE: total MUST be the large calibration set (full 4000) — the validated recipe
    # (test_tofu_unlearn N_TOTAL=4000 "preserves the alpha calibration"). A small total
    # (e.g. forget+retain_eval) miscalibrates count_ratio/covariance -> catastrophic over-edit.
    full_total = full  # 4000, a superset of forget

    # labeled edit (X+=forget10, X-=full total)
    ed_lab = edit_model(base, forget, full_total, tok, device)
    fl = mean_answer_nll(ed_lab, forget, tok, device)
    rl = mean_answer_nll(ed_lab, retain_eval, tok, device)
    probe_l = linear_probe_acc(ed_lab, tok, forget, retain_eval, device)
    del ed_lab; torch.cuda.empty_cache()
    results["labeled"] = {"forget_nll": round(fl, 3), "forget_d": round(fl - f0, 3),
                          "retain_nll": round(rl, 3), "retain_d": round(rl - r0, 3),
                          "probe_acc": round(probe_l, 3)}

    # label-light edit (X+=auto NN, X-=full total, same calibration set)
    ed_auto = edit_model(base, auto_forget, full_total, tok, device)
    fa = mean_answer_nll(ed_auto, forget, tok, device)
    ra = mean_answer_nll(ed_auto, retain_eval, tok, device)
    probe_a = linear_probe_acc(ed_auto, tok, forget, retain_eval, device)
    del ed_auto; torch.cuda.empty_cache()
    results["label_light"] = {"forget_nll": round(fa, 3), "forget_d": round(fa - f0, 3),
                              "retain_nll": round(ra, 3), "retain_d": round(ra - r0, 3),
                              "probe_acc": round(probe_a, 3)}

    print(json.dumps(results, indent=2))
    with open(f"{OUT}/selfsup_poc.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\n[interpretation]")
    print(f"  auto-split fidelity: Jaccard(auto X+, labeled forget) = {jacc:.3f} "
          f"({'high' if jacc>0.5 else 'LOW — auto split diverges from labeled'})")
    print(f"  false-success probe: base {probe0:.3f} -> labeled {probe_l:.3f} / light {probe_a:.3f} "
          f"(acc staying high while forget NLL rises = concept still decodable)")
    print("SELFSUP_POC_DONE")


if __name__ == "__main__":
    main()
