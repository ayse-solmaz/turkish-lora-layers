"""Türkçe instruction verisini indirir, temizler ve train/val/test olarak yazar.

Kaynaklar:
  - CohereLabs/aya_dataset (Türkçe kısmı)   insan yazımı, Apache-2.0
  - CausalLM/GPT-4-Self-Instruct-Turkish    GPT-4 üretimi, CC-BY-4.0

Test seti: Aya'nın kendi Türkçe test bölümü (250 örnek, eğitimde hiç görülmez).

Çıktı (her satır): {"prompt": ..., "response": ..., "source": ...}

Çalıştırma:
    .venv\\Scripts\\python src/prepare_data.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MAX_TOKENS = 1024  # prompt + response, chat template dahil
VAL_SIZE = 300
SEED = 42
OUT_DIR = Path("data")


def load_aya(split: str) -> list[dict]:
    ds = load_dataset("CohereLabs/aya_dataset", split=split)
    ds = ds.filter(lambda r: r["language_code"] == "tur")
    return [{"prompt": r["inputs"], "response": r["targets"], "source": "aya"} for r in ds]


def load_gpt4() -> list[dict]:
    ds = load_dataset("CausalLM/GPT-4-Self-Instruct-Turkish", split="train")
    return [{"prompt": r["instruction"], "response": r["output"], "source": "gpt4-self-instruct"} for r in ds]


def clean(rows: list[dict], tokenizer, stats: dict, seen: set) -> list[dict]:
    kept = []
    for r in rows:
        r = {**r, "prompt": (r["prompt"] or "").strip(), "response": (r["response"] or "").strip()}
        if len(r["prompt"]) < 5 or len(r["response"]) < 2:
            stats["boş/çok kısa"] += 1
            continue
        key = " ".join(r["prompt"].lower().split())
        if key in seen:
            stats["tekrar"] += 1
            continue
        messages = [{"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["response"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tokens > MAX_TOKENS:
            stats[f">{MAX_TOKENS} token"] += 1
            continue
        seen.add(key)
        kept.append(r)
    return kept


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    stats = {"boş/çok kısa": 0, "tekrar": 0, f">{MAX_TOKENS} token": 0}

    # Önce test setini temizle: aynı prompt'un train'e sızmasını "seen" engeller.
    seen: set = set()
    test = clean(load_aya("test"), tokenizer, stats, seen)

    pool = clean(load_aya("train") + load_gpt4(), tokenizer, stats, seen)
    random.Random(SEED).shuffle(pool)
    val, train = pool[:VAL_SIZE], pool[VAL_SIZE:]

    OUT_DIR.mkdir(exist_ok=True)
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        write_jsonl(OUT_DIR / f"{name}.jsonl", rows)
        by_source = {}
        for r in rows:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        print(f"{name:<6} {len(rows):>5}  {by_source}")
    print("atılanlar:", stats)


if __name__ == "__main__":
    main()
