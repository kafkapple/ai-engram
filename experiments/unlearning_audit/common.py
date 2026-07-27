"""Shared runtime helpers for the TOFU unlearning audit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from engram import EditorConfig, EngramEditor, compose, count_ratio, weight_norm
from tests.test_tofu_unlearn import (
    ADAPT_ALPHA,
    ADAPT_P,
    BASE_ID,
    IGNORE,
    _make_collate,
    _QAData,
)

RESULTS = Path(__file__).with_name("results")
GOLD_90_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"


def load_model(model_id: str = BASE_ID, device: str = "cuda"):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)


def load_tokenizer(model_id: str = BASE_ID):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer


def collect_engram(model, forget, full, tokenizer, device: str = "cuda"):
    editor = EngramEditor(model, EditorConfig(storage_device=torch.device(device)))

    def loader(rows):
        return DataLoader(
            _QAData(rows, tokenizer),
            batch_size=8,
            collate_fn=_make_collate(tokenizer.pad_token_id),
        )

    def batch_fn(batch):
        return {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }

    forget_stats = editor.collect_statistics(
        loader(forget), batch_fn=batch_fn, mask_fn=lambda b: b["labels"] != IGNORE
    )
    full_stats = editor.collect_statistics(
        loader(full), batch_fn=batch_fn, mask_fn=lambda b: b["labels"] != IGNORE
    )
    engram = editor.compute_engram_weights(forget_stats, full_stats)
    del forget_stats, full_stats
    torch.cuda.empty_cache()
    return editor, engram


def edit_model(
    model,
    forget,
    full,
    tokenizer,
    device: str = "cuda",
    *,
    alpha: float = ADAPT_ALPHA,
    p: float = ADAPT_P,
):
    editor, engram = collect_engram(model, forget, full, tokenizer, device)
    return editor.apply(
        engram,
        alpha=alpha,
        scale=compose(count_ratio(1.0), weight_norm(p)),
    ).eval()


def cpu_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def embed(model, tokenizer, rows, device: str = "cuda", batch_size: int = 16):
    vectors = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        encoded = tokenizer(
            [f"{row['question']} {row['answer']}" for row in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        hidden = model(**encoded, output_hidden_states=True).hidden_states[-1]
        mask = encoded["attention_mask"].unsqueeze(-1)
        vectors.append(
            ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy()
        )
    return np.concatenate(vectors)
