#!/usr/bin/env python3
"""Pre-download every HF asset the audit battery reads, into $HF_HOME.

Run where there is outbound network (e.g. a cluster login node); jobs then run with
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1. All repos are public (no token).

    uv run python -m experiments.unlearning_audit.prefetch [--with-3b]
"""
from __future__ import annotations

import argparse
import os

MODELS_1B = [
    "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",      # base
    "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",  # gold, forget10
    "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95",  # gold, forget05 crossval
    "madhurjindal/autonlp-Gibberish-Detector-492513457",    # tests/_tofu_evaluate.py
]
MODELS_3B = [
    "open-unlearning/tofu_Llama-3.2-3B-Instruct_full",
    "open-unlearning/tofu_Llama-3.2-3B-Instruct_retain90",
]
TOFU_CONFIGS = [
    "full", "forget10_perturbed", "forget05_perturbed", "retain_perturbed",
    "holdout10", "real_authors_perturbed", "world_facts_perturbed",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-3b", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    print(f"HF_HOME={os.environ.get('HF_HOME', '(default)')}", flush=True)
    for model_id in MODELS_1B + (MODELS_3B if args.with_3b else []):
        path = snapshot_download(model_id, ignore_patterns=["*.pth", "*.msgpack", "*.h5"])
        print(f"[model] {model_id} -> {path}", flush=True)
    for config in TOFU_CONFIGS:
        rows = len(load_dataset("locuslab/TOFU", config)["train"])
        print(f"[data ] locuslab/TOFU:{config} rows={rows}", flush=True)
    print("PREFETCH_DONE")


if __name__ == "__main__":
    main()
