#!/usr/bin/env python3
"""Pre-download every HF asset the experiment battery needs, into $HF_HOME.

Written for the post-gpu03 move (2026-08-13): gpu03's /node_data_2 HF cache is gone,
so the personal-server run has to re-pull everything from the Hub. All repos below
are public (no token) — verified 2026-08-13.

Usage:
  python win_prefetch_assets.py            # 1B battery + TOFU dataset (~8 GB)
  python win_prefetch_assets.py --with-3b  # adds the 3B xval pair (~14 GB more)
"""
from __future__ import annotations
import argparse, os, sys

MODELS_1B = [
    "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",      # BASE_ID
    "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",  # GOLD_ID (forget10)
    "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95",  # GOLD_ID (forget05 xval)
]
MODELS_3B = [
    "open-unlearning/tofu_Llama-3.2-3B-Instruct_full",
    "open-unlearning/tofu_Llama-3.2-3B-Instruct_retain90",
]
TOFU_CONFIGS = ["full", "forget10_perturbed", "forget05_perturbed", "retain_perturbed"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-3b", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from datasets import load_dataset

    print(f"HF_HOME={os.environ.get('HF_HOME', '(default)')}")
    ids = MODELS_1B + (MODELS_3B if args.with_3b else [])
    for mid in ids:
        print(f"[model] {mid}", flush=True)
        # weights + tokenizer only; skip the .bin duplicates when safetensors exist
        snapshot_download(mid, ignore_patterns=["*.pth", "*.msgpack", "*.h5"])
    for cfg in TOFU_CONFIGS:
        print(f"[data ] locuslab/TOFU:{cfg}", flush=True)
        n = len(load_dataset("locuslab/TOFU", cfg)["train"])
        print(f"        rows={n}", flush=True)
    print("PREFETCH_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
