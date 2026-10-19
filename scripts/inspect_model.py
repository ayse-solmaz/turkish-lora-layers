"""Modelin içine bakış: katmanlar, modüller ve parametre sayıları.

Ağırlıkları indirmeden çalışır: sadece config dosyası indirilir ve model
"meta" cihazında kurulur (tensörlerin şekli var, içi boş). Bu yüzden
CPU'da saniyeler içinde biter.

Çalıştırma:
    .venv\\Scripts\\python scripts/inspect_model.py
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B"

ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]


def fmt(n: int) -> str:
    """1234567 -> '1.23M'"""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return str(n)


# ---------------------------------------------------------------------------
# 1. Config: modelin "planı"
# ---------------------------------------------------------------------------
config = AutoConfig.from_pretrained(MODEL_ID)
print("=" * 70)
print(f"MODEL: {MODEL_ID}")
print("=" * 70)
print(f"Katman sayısı (num_hidden_layers)  : {config.num_hidden_layers}")
print(f"Gizli boyut (hidden_size)          : {config.hidden_size}")
print(f"MLP ara boyutu (intermediate_size) : {config.intermediate_size}")
print(f"Attention head sayısı              : {config.num_attention_heads}")
print(f"Key/Value head sayısı (GQA)        : {config.num_key_value_heads}")
print(f"Kelime dağarcığı (vocab_size)      : {config.vocab_size}")
print(f"Embedding ile lm_head paylaşımlı mı: {config.tie_word_embeddings}")

# ---------------------------------------------------------------------------
# 2. Modeli meta cihazında kur ve tek bir katmanın içine bak
# ---------------------------------------------------------------------------
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

layers = model.model.layers
print("\n" + "=" * 70)
print("TEK BİR TRANSFORMER KATMANI (layers[0])")
print("=" * 70)
print(layers[0])

# ---------------------------------------------------------------------------
# 3. Parametreleri gruplara göre say
# ---------------------------------------------------------------------------
groups = {"embedding": 0, "attention": 0, "mlp": 0, "norm": 0, "diğer": 0}
for name, p in model.named_parameters():  # paylaşımlı ağırlıklar bir kez sayılır
    if "embed_tokens" in name:
        groups["embedding"] += p.numel()
    elif "self_attn" in name:
        groups["attention"] += p.numel()
    elif "mlp" in name:
        groups["mlp"] += p.numel()
    elif "norm" in name:
        groups["norm"] += p.numel()
    else:
        groups["diğer"] += p.numel()

total = sum(groups.values())
print("\n" + "=" * 70)
print("PARAMETRELER NEREDE?")
print("=" * 70)
for g, n in groups.items():
    print(f"{g:<10} {fmt(n):>8}  ({100 * n / total:5.1f}%)")
print(f"{'TOPLAM':<10} {fmt(total):>8}")

n_layers = config.num_hidden_layers
attn_per_layer = groups["attention"] // n_layers
mlp_per_layer = groups["mlp"] // n_layers
print(f"\nKatman başına attention: {fmt(attn_per_layer)}")
print(f"Katman başına MLP      : {fmt(mlp_per_layer)}  "
      f"(attention'ın {mlp_per_layer / attn_per_layer:.1f} katı)")

# ---------------------------------------------------------------------------
# 4. Her lineer modülün şekli ve LoRA maliyeti
#
#    LoRA, W (out x in) matrisini dondurur ve yanına iki küçük matris ekler:
#        A: (r x in)   B: (out x r)     ->   W' = W + B @ A
#    Eğitilen parametre = r * (in + out)
# ---------------------------------------------------------------------------
layer0 = layers[0]
shapes = {}
for m in ATTN_MODULES:
    w = getattr(layer0.self_attn, m).weight
    shapes[m] = (w.shape[1], w.shape[0])  # (in, out)
for m in MLP_MODULES:
    w = getattr(layer0.mlp, m).weight
    shapes[m] = (w.shape[1], w.shape[0])

print("\n" + "=" * 70)
print("LİNEER MODÜLLER (katman başına)  ->  r=16 ile LoRA maliyeti")
print("=" * 70)
print(f"{'modül':<10} {'in':>6} {'out':>6} {'tam ağırlık':>12} {'LoRA r=16':>10}")
for m, (i, o) in shapes.items():
    print(f"{m:<10} {i:>6} {o:>6} {fmt(i * o):>12} {fmt(16 * (i + o)):>10}")


def lora_params(rank: int, modules: list[str], layer_ids: list[int]) -> int:
    per_layer = sum(rank * (shapes[m][0] + shapes[m][1]) for m in modules)
    return per_layer * len(layer_ids)


# ---------------------------------------------------------------------------
# 5. Deney matrisi: her deney kaç parametre eğitiyor?
# ---------------------------------------------------------------------------
all_layers = list(range(n_layers))
lower = all_layers[: n_layers // 2]
upper = all_layers[n_layers // 2:]
all_mods = ATTN_MODULES + MLP_MODULES

# Attention'ı MLP r=16 ile eşitleyecek rank'i bul
mlp_r16 = lora_params(16, MLP_MODULES, all_layers)
attn_per_rank = lora_params(1, ATTN_MODULES, all_layers)
matched_attn_r = round(mlp_r16 / attn_per_rank)

experiments = {
    "referans (tümü, r=16)": lora_params(16, all_mods, all_layers),
    "alt yarı (tümü, r=16)": lora_params(16, all_mods, lower),
    "üst yarı (tümü, r=16)": lora_params(16, all_mods, upper),
    "attention, r=16": lora_params(16, ATTN_MODULES, all_layers),
    f"attention, r={matched_attn_r} (eşitlenmiş)": lora_params(matched_attn_r, ATTN_MODULES, all_layers),
    "mlp, r=16": mlp_r16,
    "rank 4 (tümü)": lora_params(4, all_mods, all_layers),
    "rank 64 (tümü)": lora_params(64, all_mods, all_layers),
    "full fine-tuning": total,
}

print("\n" + "=" * 70)
print("DENEY MATRİSİ: eğitilen parametre sayısı")
print("=" * 70)
for name, n in experiments.items():
    print(f"{name:<34} {fmt(n):>8}  ({100 * n / total:6.3f}% of model)")
