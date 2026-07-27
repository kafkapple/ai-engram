"""Shared TOFU data, model, edit, and training helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from engram import EditorConfig, EngramEditor, compose, count_ratio, weight_norm

RESULTS = Path(__file__).with_name("results")
CHECKPOINTS = Path(
    os.environ.get("CHECKPOINT_ROOT", Path(__file__).with_name("checkpoints"))
) / "ai_engram"
IGNORE = -100
SYSTEM = "You are a helpful assistant."
DATE = "10 Apr 2025"
BASE_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
GOLD_90_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
PLAIN_ALPHA = 0.6
ADAPT_ALPHA, ADAPT_P = 1.0, 1
N_TOTAL = 4000
N_RETAIN_EVAL = 200


def preprocess(tokenizer, question, answer):
    chat = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    kwargs = {"date_string": DATE}
    full = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False, return_dict=False, **kwargs
    )
    prompt = tokenizer.apply_chat_template(
        chat[:-1], tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs
    )
    if full[-1] != tokenizer.eos_token_id:
        full.append(tokenizer.eos_token_id)
    labels = [IGNORE] * len(prompt) + full[len(prompt) :]
    return {
        "input_ids": torch.tensor(full),
        "labels": torch.tensor(labels),
        "attention_mask": torch.ones(len(full), dtype=torch.long),
    }


class QAData(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows, self.tokenizer = rows, tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return preprocess(self.tokenizer, row["question"], row["answer"])


def make_collate(pad_id):
    def collate(items):
        ids = pad_sequence(
            [item["input_ids"] for item in items],
            batch_first=True,
            padding_value=pad_id,
        )
        labels = pad_sequence(
            [item["labels"] for item in items],
            batch_first=True,
            padding_value=IGNORE,
        )
        return {
            "input_ids": ids,
            "attention_mask": ids.ne(pad_id).long(),
            "labels": labels,
        }

    return collate


@torch.no_grad()
def mean_answer_nll(model, rows, tokenizer, device="cuda", batch_size=16):
    loader = DataLoader(
        QAData(rows, tokenizer),
        batch_size=batch_size,
        collate_fn=make_collate(tokenizer.pad_token_id),
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="none")
    values = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(
            input_ids=ids,
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        losses = loss_fn(
            logits[..., :-1, :].contiguous().transpose(-1, -2),
            labels[..., 1:].contiguous(),
        ).sum(-1)
        values.extend(
            (losses / (labels != IGNORE).sum(-1)).float().cpu().tolist()
        )
    return float(np.mean(values))


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
            QAData(rows, tokenizer),
            batch_size=8,
            collate_fn=make_collate(tokenizer.pad_token_id),
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


def finetune(
    model,
    rows,
    tokenizer,
    device="cuda",
    steps=40,
    *,
    seed=0,
    learning_rate=2e-5,
    batch_size=8,
):
    torch.manual_seed(seed)
    loader = DataLoader(
        QAData(rows, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    done = 0
    while done < steps:
        model.train()
        for batch in loader:
            loss = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            ).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            done += 1
            if done >= steps:
                break
    return model.eval()


def save_json(name, value):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path
