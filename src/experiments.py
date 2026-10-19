"""Deney matrisi: her deney tek bir sözlükle tanımlanır.

Qwen2.5-0.5B'de 24 katman var: alt yarı 0-11, üst yarı 12-23.
Parametre sayıları için: scripts/inspect_model.py
"""

N_LAYERS = 24
LOWER = list(range(N_LAYERS // 2))
UPPER = list(range(N_LAYERS // 2, N_LAYERS))

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
ALL = ATTN + MLP

LORA_LR = 2e-4
FULL_LR = 2e-5  # full fine-tuning tüm ağırlıkları değiştirir, ~10 kat düşük LR ister


def lora(rank: int, modules: list[str], layers: list[int] | None = None) -> dict:
    return {
        "method": "lora",
        "rank": rank,
        "alpha": 2 * rank,  # alpha/rank = 2 sabit: farklı rank'lerde aynı ölçek
        "modules": modules,
        "layers": layers,  # None = tüm katmanlar
        "lr": LORA_LR,
    }


EXPERIMENTS = {
    # referans
    "lora_all_r16": lora(16, ALL),
    # 1-2. alt mı üst mü?
    "lora_lower_r16": lora(16, ALL, LOWER),
    "lora_upper_r16": lora(16, ALL, UPPER),
    # 3. attention mı MLP mi? (r=49 attention ~ r=16 MLP parametre sayısı)
    "lora_attn_r16": lora(16, ATTN),
    "lora_attn_r49": lora(49, ATTN),
    "lora_mlp_r16": lora(16, MLP),
    # 4. rank
    "lora_all_r4": lora(4, ALL),
    "lora_all_r64": lora(64, ALL),
    # 5. full fine-tuning
    "full_ft": {"method": "full", "lr": FULL_LR},
}
