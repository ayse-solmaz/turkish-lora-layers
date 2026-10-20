"""Eğitilmiş modelleri değerlendirir.

Her model için:
  1. Türkçe test loss      Aya test seti (250 örnek, eğitimde hiç görülmedi)
  2. Unutma (İngilizce)    WikiText-2 perplexity + ARC-Easy doğruluğu
  3. Örnek cevaplar        sabit prompt seti (src/prompts.py), greedy üretim
  4. Davranış ölçümleri    dil kayması, tekrar, cevabı bitirebilme, talimata uyum

Unutma ölçümleri chat template OLMADAN yapılır: modelin ham dil bilgisine bakıyoruz.

Kullanım:
    python src/evaluate.py --runs_dir outputs/seed42 --exps base lora_all_r16 full_ft
    python src/evaluate.py --runs_dir outputs/seed42 --quick      # küçük, hızlı deneme

Çıktı: results/eval/<deney>.json
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from data import SFTDataset, make_collate  # noqa: E402
from experiments import EXPERIMENTS  # noqa: E402
from prompts import PROMPTS  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B"
ALL = ["base"] + list(EXPERIMENTS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="outputs/seed42")
    p.add_argument("--exps", nargs="+", default=ALL, choices=ALL)
    p.add_argument("--out_dir", default="results/eval")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--quick", action="store_true", help="her ölçümde çok az örnek (deneme için)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model yükleme
# ---------------------------------------------------------------------------
def load_model(exp: str, runs_dir: Path, device, dtype):
    if exp == "base":
        return AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    path = runs_dir / exp / "model"
    if EXPERIMENTS[exp]["method"] == "full":
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    # LoRA'yı ağırlıklara gömer (W + B@A): üretim hızı base model ile aynı olur
    model = PeftModel.from_pretrained(base, path).merge_and_unload()
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# 1. Türkçe test loss (eğitimdekiyle aynı: sadece cevap token'ları)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sft_loss(model, tokenizer, path, device, limit=None) -> float:
    ds = SFTDataset(path, tokenizer, limit)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=make_collate(tokenizer.pad_token_id))
    loss_sum, n_tok = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        n = (batch["labels"][:, 1:] != -100).sum().item()
        loss_sum += loss.item() * n
        n_tok += n
    return loss_sum / n_tok


# ---------------------------------------------------------------------------
# 2a. İngilizce perplexity: WikiText-2 test, 1024 token'lık parçalar
# ---------------------------------------------------------------------------
@torch.no_grad()
def wikitext_ppl(model, tokenizer, device, max_tokens: int) -> float:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    # Metnin tamamını tokenize edip baştan max_tokens alıyoruz ("uzun dizi" uyarısı zararsız:
    # modele hiçbir zaman 1024'ten uzun parça verilmiyor)
    ids = tokenizer("\n\n".join(ds["text"]), add_special_tokens=False)["input_ids"][:max_tokens]
    block = 1024
    chunks = [ids[i:i + block] for i in range(0, len(ids) - block + 1, block)]
    nll, n_tok = 0.0, 0
    for i in range(0, len(chunks), 4):
        x = torch.tensor(chunks[i:i + 4], device=device)
        loss = model(input_ids=x, labels=x).loss
        n = x.numel() - x.shape[0]
        nll += loss.item() * n
        n_tok += n
    return math.exp(nll / n_tok)


# ---------------------------------------------------------------------------
# 2b. ARC-Easy: her şık için log-olasılık, en yüksek olan modelin cevabı
# ---------------------------------------------------------------------------
@torch.no_grad()
def arc_easy(model, tokenizer, device, n_questions: int) -> float:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").select(range(n_questions))
    correct = 0
    for q in ds:
        context = f"Question: {q['question']}\nAnswer:"
        ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        seqs, n_ans = [], []
        for choice in q["choices"]["text"]:
            ans_ids = tokenizer(" " + choice, add_special_tokens=False)["input_ids"]
            seqs.append(ctx_ids + ans_ids)
            n_ans.append(len(ans_ids))
        max_len = max(len(s) for s in seqs)
        x = torch.tensor([s + [tokenizer.pad_token_id] * (max_len - len(s)) for s in seqs], device=device)
        logprobs = torch.log_softmax(model(input_ids=x).logits.float(), dim=-1)
        scores = []
        for j, s in enumerate(seqs):
            # cevap token'larının log-olasılık toplamı (her token'ı bir önceki konum tahmin eder)
            pos = range(len(s) - n_ans[j], len(s))
            scores.append(sum(logprobs[j, p - 1, s[p]].item() for p in pos))
        pred = q["choices"]["label"][max(range(len(scores)), key=scores.__getitem__)]
        correct += pred == q["answerKey"]
    return correct / len(ds)


# ---------------------------------------------------------------------------
# 3. Örnek cevaplar
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, tokenizer, prompts, device, max_new_tokens: int) -> list[dict]:
    tokenizer.padding_side = "left"  # üretimde dolgu sola: cevap herkes için aynı konumda başlar
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    out = []
    for i in range(0, len(prompts), 8):
        group = prompts[i:i + 8]
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": p["text"]}],
                                               tokenize=False, add_generation_prompt=True) for p in group]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=eos, pad_token_id=tokenizer.pad_token_id)
        for p, g in zip(group, gen[:, enc["input_ids"].shape[1]:]):
            ids = g.tolist()
            finished = eos in ids
            ids = ids[: ids.index(eos)] if finished else ids
            out.append({**p, "response": tokenizer.decode(ids, skip_special_tokens=True).strip(),
                        "finished": finished})
    tokenizer.padding_side = "right"
    return out


# ---------------------------------------------------------------------------
# 4. Davranış ölçümleri (basit, açıklanabilir sezgiler)
# ---------------------------------------------------------------------------
TR_WORDS = set("ve bir bu da de için ile ama çok daha gibi olarak olan en ne mi değil var yok ki "
               "kadar sonra önce her şey nasıl neden çünkü ise veya yani".split())
EN_WORDS = set("the and of to a in is it that for with as are this be on or by an not you "
               "can your from have which they".split())


def detect_lang(text: str) -> str:
    words = re.findall(r"[a-zçğıöşü]+", text.lower())
    tr = sum(w in TR_WORDS for w in words) + 2 * len(re.findall(r"[çğıöşü]", text.lower()))
    en = sum(w in EN_WORDS for w in words)
    if tr == en == 0:
        return "?"
    return "tr" if tr > en else "en"


def repetition(text: str) -> float:
    """Tekrar eden 4-gram oranı: 0 = hiç tekrar yok, 1'e yaklaştıkça döngüde."""
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def check_instruction(text: str, check: dict) -> bool:
    if check["type"] == "items":
        items = [l for l in text.splitlines() if re.match(r"^\s*(\d+[.)]|[-*•])\s+", l)]
        return len(items) == check["n"]
    if check["type"] == "max_sentences":
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        return len(sentences) <= check["n"]
    if check["type"] == "max_words":
        return len(text.split()) <= check["n"]
    raise ValueError(check)


def behavior(gens: list[dict]) -> dict:
    tr = [g for g in gens if g["lang"] == "tr" and not g["id"].startswith("tr_ceviri")]
    en = [g for g in gens if g["lang"] == "en"]
    checks = [g for g in gens if "check" in g]
    for g in gens:
        g["detected_lang"] = detect_lang(g["response"])
        g["repetition"] = round(repetition(g["response"]), 3)
        if "check" in g:
            g["check_passed"] = check_instruction(g["response"], g["check"])
    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "tr_answered_in_tr": mean(g["detected_lang"] == "tr" for g in tr),
        "en_answered_in_en": mean(g["detected_lang"] == "en" for g in en),
        "mean_repetition": mean(g["repetition"] for g in gens),
        "finished_rate": mean(g["finished"] for g in gens),
        "instruction_pass": mean(g["check_passed"] for g in checks),
        "mean_words": mean(len(g["response"].split()) for g in gens),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    runs_dir, out_dir = Path(args.runs_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q = args.quick
    sizes = dict(test_limit=8 if q else None, wiki_tokens=4096 if q else 200_000,
                 arc_n=5 if q else 500, prompts=PROMPTS[:3] + PROMPTS[-1:] if q else PROMPTS,
                 max_new=32 if q else 256)

    for exp in args.exps:
        start = time.time()
        print(f"== {exp}", flush=True)
        model = load_model(exp, runs_dir, device, dtype)
        res = {"exp": exp}
        res["test_loss"] = sft_loss(model, tokenizer, Path(args.data_dir) / "test.jsonl", device, sizes["test_limit"])
        print(f"   türkçe test loss : {res['test_loss']:.4f}", flush=True)
        res["en_ppl"] = wikitext_ppl(model, tokenizer, device, sizes["wiki_tokens"])
        print(f"   ingilizce ppl    : {res['en_ppl']:.2f}", flush=True)
        res["arc_easy"] = arc_easy(model, tokenizer, device, sizes["arc_n"])
        print(f"   ARC-Easy         : {res['arc_easy']:.3f}", flush=True)
        gens = generate(model, tokenizer, sizes["prompts"], device, sizes["max_new"])
        res["behavior"] = behavior(gens)
        res["generations"] = gens
        print(f"   davranış         : {json.dumps({k: round(v, 3) for k, v in res['behavior'].items()})}", flush=True)
        res["eval_seconds"] = round(time.time() - start, 1)
        with open(out_dir / f"{exp}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
