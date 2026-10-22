"""Hugging Face'e yüklenecek iki repoyu yerelde hazırlar (yüklemez).

    hf_upload/qwen2.5-0.5b-tr-lora-placement/   8 LoRA adapter'ı (alt klasörler) + model kartı + grafikler
    hf_upload/qwen2.5-0.5b-tr-sft/              full fine-tuning modeli + tokenizer + model kartı

Kullanım:
    python scripts/prepare_hf.py --runs_dir kaggle_v2/turkish-lora-layers/outputs/seed42 --bundle results/bundle_seed42
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from plots import NAMES  # noqa: E402

USER = "ayse-solmaz"
LORA_REPO = "qwen2.5-0.5b-tr-lora-placement"
SFT_REPO = "qwen2.5-0.5b-tr-sft"
BASE = "Qwen/Qwen2.5-0.5B"
LORA_EXPS = ["lora_all_r16", "lora_all_r4", "lora_all_r64", "lora_lower_r16", "lora_upper_r16",
             "lora_attn_r16", "lora_attn_r49", "lora_mlp_r16"]
FIGURES = ["figures/val_loss_curves.png", "figures/loss_vs_params.png",
           "figures/learning_vs_forgetting.png", "figures/v1_vs_v2.png",
           "weights/spectra.png", "weights/layer_profile.png"]


def results_table(bundle: Path, exps: list[str]) -> str:
    rows = ["| experiment | trainable params | TR test loss ↓ | EN ppl ↓ | ARC-Easy | finishes answer |",
            "|---|---:|---:|---:|---:|---:|"]
    data = []
    for e in exps:
        ev = json.loads((bundle / "eval" / f"{e}.json").read_text(encoding="utf-8"))
        tl = bundle / "train_logs" / f"{e}.json"
        params = json.loads(tl.read_text(encoding="utf-8"))["trainable_params"] if tl.exists() else 0
        data.append((ev["test_loss"], e, params, ev))
    for loss, e, params, ev in sorted(data):
        p = f"{params / 1e6:.2f}M" if params else "-"
        rows.append(f"| `{e}` | {p} | {loss:.3f} | {ev['en_ppl']:.2f} | {ev['arc_easy']:.3f} | "
                    f"{ev['behavior']['finished_rate']:.0%} |")
    return "\n".join(rows)


LORA_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B
library_name: peft
language: [tr]
tags: [lora, peft, turkish, research, ablation]
datasets: [CohereLabs/aya_dataset, CausalLM/GPT-4-Self-Instruct-Turkish]
---

# Where should LoRA go? — Turkish instruction tuning ablation (Qwen2.5-0.5B)

**8 LoRA adapters from one controlled experiment.** Same base model, same data (6.5k Turkish
instructions), same steps and seed — only *where* LoRA is placed and its rank change.
Full fine-tuning baseline: [{user}/{sft}](https://huggingface.co/{user}/{sft}) ·
Code: [github.com/{user}/turkish-lora-layers](https://github.com/{user}/turkish-lora-layers)

> ⚠️ Research artifact. A 0.5B model fine-tuned on 6.5k examples hallucinates facts and still fails
> to stop in about half of its answers. Use it to study fine-tuning, not as an assistant.

## Adapters

| subfolder | layers | modules | rank |
|---|---|---|---|
| `lora_all_r16` (reference) | all 24 | attention + MLP | 16 |
| `lora_lower_r16` | 0–11 | attention + MLP | 16 |
| `lora_upper_r16` | 12–23 | attention + MLP | 16 |
| `lora_attn_r16` | all | q, k, v, o | 16 |
| `lora_attn_r49` | all | q, k, v, o | 49 (param-matched to MLP r16) |
| `lora_mlp_r16` | all | gate, up, down | 16 |
| `lora_all_r4` / `lora_all_r64` | all | attention + MLP | 4 / 64 |

All adapters also train the two chat-token embedding rows (`<|im_start|>`, `<|im_end|>`, +1,792 params) — see *Lesson learned* below.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = PeftModel.from_pretrained(base, "{user}/{lora}", subfolder="lora_mlp_r16")

msgs = [{{"role": "user", "content": "Bir tren saatte 80 km hızla 3 saat giderse kaç kilometre yol alır?"}}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
out = model.generate(**inputs, max_new_tokens=200, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

<details><summary>Windows: "Can't find weights" error?</summary>

PEFT 0.21 builds the Hub path with `os.path.join`, which uses `\\` on Windows, so `subfolder=` fails there.
Download the subfolder first and load it from disk:

```python
import os
from huggingface_hub import snapshot_download
local = snapshot_download("{user}/{lora}", allow_patterns="lora_mlp_r16/*")
model = PeftModel.from_pretrained(base, os.path.join(local, "lora_mlp_r16"))
```
</details>

## Results (seed 42)

{table}

TR test loss: Aya Turkish test split (250 human-written examples, never seen in training).
EN ppl: WikiText-2 perplexity (forgetting). ARC-Easy: 500 questions (±4 points is noise).
"Finishes answer": share of 24 fixed prompts answered with `<|im_end|>` within 256 tokens.

![loss vs params](figures/loss_vs_params.png)
![val loss curves](figures/val_loss_curves.png)

**Findings (single seed — treat as preliminary):**
- **MLP beats attention** at equal parameter count (6.6M): test loss 2.996 vs 3.086.
- **Upper layers beat lower layers** (4.4M each): 3.068 vs 3.102. Weight analysis agrees: even full fine-tuning changes upper layers more.
- **Rank matters a lot** for teaching a new language: r4 3.080 → r16 2.978 → r64 2.864. Adapters use most of their rank (effective rank 3.2/4, 11/16, 36/64), while full fine-tuning's update has effective rank ~279 — the change is *not* low-rank.
- **Same budget, spread wins:** all modules at r4 (2.2M) ≈ attention-only at r49 (6.6M).
- **LoRA forgets less:** English perplexity 14.7 (base) → 15.6 (LoRA r16) vs 17.3 (full FT).

![learning vs forgetting](figures/learning_vs_forgetting.png)
![singular values](figures/spectra.png)

## Lesson learned: untrained chat tokens

In Qwen2.5-0.5B **base**, the embedding rows of `<|im_start|>` and `<|im_end|>` are identical to ~290
unused padding rows (embeddings are tied with the output head). With frozen embeddings, LoRA can
never make `<|im_end|>` more likely than its identical twins, so the first run's adapters **never
stopped generating** (0% vs 75% for full fine-tuning). Training just those two rows
(`trainable_token_indices` in PEFT) fixed it:

![v1 vs v2](figures/v1_vs_v2.png)

## Training

Qwen2.5-0.5B base · 6,573 examples (Aya Turkish + GPT-4-Self-Instruct-Turkish), ≤1024 tokens ·
loss on answer tokens only · 2 epochs, 822 steps, effective batch 16 · AdamW, lr 2e-4 (LoRA) /
2e-5 (full), cosine, 3% warmup · alpha = 2·rank, dropout 0.05 · fp16 on Kaggle T4.

## Data & license

Adapters: Apache-2.0 (same as the base model). Training data: [Aya dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset)
(Apache-2.0) and [GPT-4-Self-Instruct-Turkish](https://huggingface.co/datasets/CausalLM/GPT-4-Self-Instruct-Turkish) (CC-BY-4.0).
"""

SFT_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B
language: [tr]
tags: [turkish, sft, research]
datasets: [CohereLabs/aya_dataset, CausalLM/GPT-4-Self-Instruct-Turkish]
pipeline_tag: text-generation
---

# Qwen2.5-0.5B Turkish SFT (full fine-tuning)

Full fine-tuning baseline from a LoRA placement study. All 494M parameters trained on 6.5k
Turkish instructions. LoRA variants: [{user}/{lora}](https://huggingface.co/{user}/{lora}) ·
Code: [github.com/{user}/turkish-lora-layers](https://github.com/{user}/turkish-lora-layers)

> ⚠️ Research artifact. Small model, small data: it often states wrong facts with confidence and
> sometimes repeats itself. Not for production use.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{user}/{sft}")
tok = AutoTokenizer.from_pretrained("{user}/{sft}")
msgs = [{{"role": "user", "content": "Sağlıklı yaşam için 3 öneri yaz."}}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
out = model.generate(**inputs, max_new_tokens=200)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Results (seed 42)

{table}

Compared with the best LoRA adapter (r64, 35M params), full fine-tuning learns Turkish best
(test loss 2.743 vs 2.864) but forgets the most English (WikiText-2 perplexity 14.7 → 17.3).

## Training

2 epochs (822 steps), effective batch 16, AdamW lr 2e-5, cosine schedule, 3% warmup,
loss on answer tokens only, max 1024 tokens, fp16 mixed precision on a Kaggle T4. Weights stored in fp16.

## Data & license

Apache-2.0 (same as the base model). Data: [Aya dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset)
(Apache-2.0), [GPT-4-Self-Instruct-Turkish](https://huggingface.co/datasets/CausalLM/GPT-4-Self-Instruct-Turkish) (CC-BY-4.0).
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--bundle", default="results/bundle_seed42")
    p.add_argument("--out_dir", default="hf_upload")
    args = p.parse_args()
    runs, bundle, out = Path(args.runs_dir), Path(args.bundle), Path(args.out_dir)
    fmt = dict(user=USER, lora=LORA_REPO, sft=SFT_REPO)

    # --- LoRA reposu ---
    lora_dir = out / LORA_REPO
    shutil.rmtree(lora_dir, ignore_errors=True)
    for e in LORA_EXPS:
        dst = lora_dir / e
        dst.mkdir(parents=True)
        for f in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copy(runs / e / "model" / f, dst / f)
    (lora_dir / "figures").mkdir()
    for f in FIGURES:
        shutil.copy(ROOT / "results" / f, lora_dir / "figures" / Path(f).name)
    table = results_table(bundle, ["base"] + LORA_EXPS + ["full_ft"])
    (lora_dir / "README.md").write_text(LORA_CARD.format(table=table, **fmt), encoding="utf-8")

    # --- full FT reposu ---
    sft_dir = out / SFT_REPO
    shutil.rmtree(sft_dir, ignore_errors=True)
    sft_dir.mkdir(parents=True)
    for f in ("model.safetensors", "config.json"):
        shutil.copy(runs / "full_ft" / "model" / f, sft_dir / f)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(sft_dir)  # eğitimde tokenizer değişmedi
    # Sohbet modeli olarak kullanılınca <|im_end|>'de dursun (base'in varsayılanı sadece <|endoftext|>)
    gen = GenerationConfig.from_pretrained(BASE)
    gen.eos_token_id = [151645, 151643]
    gen.save_pretrained(sft_dir)
    table = results_table(bundle, ["base", "lora_all_r16", "lora_all_r64", "full_ft"])
    (sft_dir / "README.md").write_text(SFT_CARD.format(table=table, **fmt), encoding="utf-8")

    for d in (lora_dir, sft_dir):
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
        print(f"{d}: {size:.0f} MB, {sum(1 for f in d.rglob('*') if f.is_file())} dosya")


if __name__ == "__main__":
    main()
