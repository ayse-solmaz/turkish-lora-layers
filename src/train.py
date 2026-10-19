"""Tek bir deneyi eğitir: LoRA veya full fine-tuning.

Kullanım:
    python src/train.py --exp lora_all_r16
    python src/train.py --exp full_ft
    python src/train.py --exp lora_all_r16 --limit 16 --max_steps 3   # hızlı deneme

Çıktı: outputs/<exp>/  ->  log.json (loss'lar, süre, bellek) + ağırlıklar/adapter
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from data import SFTDataset, make_collate  # noqa: E402
from experiments import EXPERIMENTS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=EXPERIMENTS.keys())
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)  # efektif batch = 4 x 4 = 16
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--eval_every", type=int, default=50)  # optimizer adımı cinsinden
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="hızlı deneme için örnek sayısı")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--no_grad_ckpt", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model: LoRA veya full
# ---------------------------------------------------------------------------
def build_model(args, exp: dict):
    # Ağırlıklar fp32 yüklenir; hız için ileri/geri geçişte autocast (fp16/bf16) kullanılır.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.config.use_cache = False  # eğitimde KV cache gereksiz

    if not args.no_grad_ckpt:
        # Aktivasyonları saklamak yerine geri geçişte yeniden hesaplar: daha az bellek, ~%30 daha yavaş
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if exp["method"] == "lora":
        config = LoraConfig(
            r=exp["rank"],
            lora_alpha=exp["alpha"],
            lora_dropout=0.05,
            target_modules=exp["modules"],      # hangi lineer katmanlar: q_proj, gate_proj, ...
            layers_to_transform=exp["layers"],  # hangi transformer katmanları: [0..11] gibi
            layers_pattern="layers" if exp["layers"] else None,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)  # base ağırlıkları dondurur, A/B matrislerini ekler

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Eğitilen parametre: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
    return model, trainable


# ---------------------------------------------------------------------------
# Learning rate: kısa warmup, sonra cosine ile sıfıra iner
# ---------------------------------------------------------------------------
def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return f


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype) -> float:
    """Token ağırlıklı ortalama loss (uzun cevaplar daha çok sayılır)."""
    model.eval()
    loss_sum, n_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = model(**batch).loss
        n = (batch["labels"][:, 1:] != -100).sum().item()  # model label'ları 1 kaydırır
        loss_sum += loss.item() * n
        n_tokens += n
    model.train()
    return loss_sum / n_tokens


def main():
    args = parse_args()
    exp = EXPERIMENTS[args.exp]
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = None  # CPU'da sadece hızlı deneme yapıyoruz, fp32 yeter
    print(f"Cihaz: {device}, karışık hassasiyet: {amp_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_dir = Path(args.data_dir)
    train_ds = SFTDataset(data_dir / "train.jsonl", tokenizer, args.limit)
    val_ds = SFTDataset(data_dir / "val.jsonl", tokenizer, args.limit)
    collate = make_collate(tokenizer.pad_token_id)
    g = torch.Generator().manual_seed(args.seed)  # her deney veriyi aynı sırayla görür
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=g)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=collate)

    model, n_trainable = build_model(args, exp)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=exp["lr"], weight_decay=0.0)
    total_steps = args.epochs * math.ceil(len(train_loader) / args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(args.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(warmup, total_steps))
    # fp16'da küçük gradyanlar sıfıra yuvarlanabilir; GradScaler loss'u büyütüp bunu önler
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    log = {"exp": args.exp, "config": exp, "args": vars(args), "trainable_params": n_trainable,
           "train_loss": [], "val_loss": []}
    out = Path(args.out_dir) / args.exp
    out.mkdir(parents=True, exist_ok=True)

    log["val_loss"].append([0, evaluate(model, val_loader, device, amp_dtype)])
    print(f"adım 0 | val loss {log['val_loss'][-1][1]:.4f}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    step, running = 0, []
    model.train()
    done = False
    for epoch in range(args.epochs):
        # Epoch sonunda grad_accum'a tamamlanmamış artık mini-batch'ler varsa at
        optimizer.zero_grad(set_to_none=True)
        running = []
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            running.append(loss.item() * args.grad_accum)

            # Gradyan biriktirme: grad_accum mini-batch'te bir optimizer adımı
            if (i + 1) % args.grad_accum != 0:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            log["train_loss"].append([step, sum(running) / len(running)])
            running = []
            if step % 10 == 0:
                print(f"adım {step}/{total_steps} | train loss {log['train_loss'][-1][1]:.4f} "
                      f"| lr {scheduler.get_last_lr()[0]:.2e}")
            if step % args.eval_every == 0 or step == total_steps:
                log["val_loss"].append([step, evaluate(model, val_loader, device, amp_dtype)])
                print(f"adım {step} | val loss {log['val_loss'][-1][1]:.4f}")
            if step >= total_steps:
                done = True
                break
        if done:
            break

    if log["val_loss"][-1][0] != step:  # son adımda değerlendirilmediyse
        log["val_loss"].append([step, evaluate(model, val_loader, device, amp_dtype)])
    log["train_seconds"] = round(time.time() - start, 1)
    if device.type == "cuda":
        log["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        log["gpu"] = torch.cuda.get_device_name()
    print(f"Bitti: {log['train_seconds']} sn, son val loss {log['val_loss'][-1][1]:.4f}")

    # LoRA: sadece adapter (birkaç MB). Full: tüm model fp16 olarak (~1 GB).
    if exp["method"] == "full":
        model = model.to(torch.float16)
    model.save_pretrained(out / "model")
    tokenizer.save_pretrained(out / "model")
    with open(out / "log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
