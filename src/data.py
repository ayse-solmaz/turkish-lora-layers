"""Veriyi modele hazırlar: chat template + label masking.

Label masking fikri:
    Model her token için "bir sonrakini tahmin et" diye eğitilir. Ama biz
    modelin soruyu ezberlemesini değil, cevap vermeyi öğrenmesini istiyoruz.
    Bu yüzden soru (prompt) kısmındaki token'ların label'ını -100 yaparız;
    PyTorch'un cross-entropy'si -100'leri yok sayar. Loss sadece cevaptan gelir.

    token :  <system> <user> Fransa'nın başkenti? <assistant> Paris . <|im_end|>
    label :   -100     -100     -100 ...            -100      Paris . <|im_end|>
"""

import json
from pathlib import Path

import torch

IGNORE = -100


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


def encode(row: dict, tokenizer) -> dict:
    user = [{"role": "user", "content": row["prompt"]}]
    full = user + [{"role": "assistant", "content": row["response"]}]

    # Soru kısmı: "...<|im_start|>assistant\n" ile biter
    prompt_text = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    # Tamamı: soru + cevap + <|im_end|>
    full_text = tokenizer.apply_chat_template(full, tokenize=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    assert input_ids[: len(prompt_ids)] == prompt_ids, "prompt, tam metnin başı olmalı"

    labels = [IGNORE] * len(prompt_ids) + input_ids[len(prompt_ids):]
    return {"input_ids": input_ids, "labels": labels}


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path, tokenizer, limit: int | None = None):
        self.examples = [encode(r, tokenizer) for r in read_jsonl(path, limit)]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        return self.examples[i]


def make_collate(pad_id: int):
    """Batch'teki örnekleri en uzununa göre sağdan doldurur (dynamic padding)."""

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attention_mask = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [IGNORE] * pad)  # dolgu da loss'a girmez
            attention_mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }

    return collate
