#!/usr/bin/env python3
"""Test 1 (suppression-vs-erasure): probe TRANSFER + multi-seed.

Disambiguates the preliminary "probe went up (0.725->0.847)" result. A probe trained on
the BASE model's forget-vs-retain, then applied to the EDITED model's activations:
  transfer acc ~ base acc  => the SAME concept direction survives the edit (persistence/suppression)
  transfer acc ~ chance    => base concept direction destroyed (erasure of that direction)
Reports base-CV, edited-CV, and base->edited TRANSFER over multiple probe seeds (mean±std),
plus an activation-norm control (probe-up could be a norm/scale artifact).

Reuses tests/test_tofu_unlearn.py helpers. Writes results/probe_transfer.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

from tests.test_tofu_unlearn import BASE_ID, N_RETAIN_EVAL  # noqa: E402
from experiments.unlearning_audit.common import edit_model, embed

OUT = os.path.join(os.path.dirname(__file__), "results")
N_TOTAL = 4000


def probe_suite(Xb_f, Xb_r, Xe_f, Xe_r, seeds=(0, 1, 2, 3, 4)):
    """base-CV, edited-CV, and base->edited transfer accuracy over seeds (balanced classes)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.decomposition import PCA
    # p>>n guard: 2048 features vs ~300 samples overfits (any two sets separate in high-dim
    # via surface tokens). Reduce to PCA-fit-on-base so separability reflects structure, not
    # dimensionality. Fit PCA on base combined; apply same basis to edited (shared frame).
    N_PC = 50
    rng_n = min(len(Xb_f), len(Xb_r), len(Xe_f), len(Xe_r))
    out = {"base_cv": [], "edited_cv": [], "transfer": [], "chance": 0.5}
    for s in seeds:
        rs = np.random.RandomState(s)
        idx = rs.permutation(rng_n)[:min(rng_n, 190)]
        y = np.r_[np.ones(len(idx)), np.zeros(len(idx))]
        # standardize the COMBINED set (both classes together) so the between-class
        # difference — the discriminative signal — is preserved (NOT per-class).
        Xb = np.concatenate([Xb_f[idx], Xb_r[idx]]); mu_b, sd_b = Xb.mean(0), Xb.std(0) + 1e-6
        Xe = np.concatenate([Xe_f[idx], Xe_r[idx]])
        Xb_z = (Xb - mu_b) / sd_b
        pca = PCA(n_components=N_PC, random_state=s).fit(Xb_z)      # basis from BASE
        Xb_p = pca.transform(Xb_z)
        Xe_p_base = pca.transform((Xe - mu_b) / sd_b)               # edited in BASE frame (transfer)
        Xe_z = (Xe - Xe.mean(0)) / (Xe.std(0) + 1e-6)
        pca_e = PCA(n_components=N_PC, random_state=s).fit(Xe_z)
        Xe_p_own = pca_e.transform(Xe_z)                            # edited own frame (edited_cv)
        out["base_cv"].append(float(cross_val_score(LogisticRegression(max_iter=1000), Xb_p, y, cv=5).mean()))
        out["edited_cv"].append(float(cross_val_score(LogisticRegression(max_iter=1000), Xe_p_own, y, cv=5).mean()))
        clf = LogisticRegression(max_iter=1000).fit(Xb_p, y)       # train on BASE(PCA)
        out["transfer"].append(float((clf.predict(Xe_p_base) == y).mean()))  # apply to EDITED(base frame)
    return out


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
    retain = list(load_dataset("locuslab/TOFU", "retain_perturbed")["train"])[:N_RETAIN_EVAL]

    Xb_f, Xb_r = embed(base, tok, forget, device), embed(base, tok, retain, device)
    norm_b = float(np.linalg.norm(np.concatenate([Xb_f, Xb_r]), axis=1).mean())

    edited = edit_model(base, forget, full, tok, device)
    Xe_f, Xe_r = embed(edited, tok, forget, device), embed(edited, tok, retain, device)
    norm_e = float(np.linalg.norm(np.concatenate([Xe_f, Xe_r]), axis=1).mean())
    del edited; torch.cuda.empty_cache()

    res = probe_suite(Xb_f, Xb_r, Xe_f, Xe_r)
    summ = {k: {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)}
            for k, v in res.items() if isinstance(v, list)}
    out = {"probe": summ, "chance": 0.5,
           "act_norm": {"base": round(norm_b, 2), "edited": round(norm_e, 2),
                        "ratio": round(norm_e / norm_b, 3)}}
    print(json.dumps(out, indent=2))
    with open(f"{OUT}/probe_transfer.json", "w") as fp:
        json.dump(out, fp, indent=2)
    t = summ["transfer"]["mean"]
    print(f"\n[verdict] base->edited TRANSFER = {t:.3f} (chance 0.5)")
    print("  ~base_cv => same concept direction SURVIVES edit (persistence/suppression)")
    print("  ~chance  => base concept direction ERASED")
    print(f"  norm ratio edited/base = {out['act_norm']['ratio']} (far from 1.0 => probe-up may be scale artifact)")
    print("PROBE_TRANSFER_DONE")


if __name__ == "__main__":
    main()
