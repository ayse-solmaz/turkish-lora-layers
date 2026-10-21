# Seed 42 sonuçları (Kaggle T4 x2, fp16)

Notebook: https://www.kaggle.com/code/aysenursolmaz/notebook0473565067 (toplam 4.2 saat)

Base model (eğitimsiz) val loss: **2.8208**

| deney | eğitilen param (M) | son val loss | süre (dk) | tepe bellek (GB) |
|---|---:|---:|---:|---:|
| full_ft | 494.03 | **1.8197** | 47.8 | 12.00 |
| lora_all_r64 | 35.19 | 1.9798 | 59.5 | 6.49 |
| lora_all_r16 | 8.80 | 2.1816 | 56.9 | 6.06 |
| lora_mlp_r16 | 6.64 | 2.2072 | 46.6 | 6.03 |
| lora_upper_r16 | 4.40 | 2.3385 | 44.7 | 5.99 |
| lora_all_r4 | 2.20 | 2.3472 | 54.9 | 5.96 |
| lora_attn_r49 | 6.62 | 2.3730 | 42.8 | 6.03 |
| lora_lower_r16 | 4.40 | 2.4468 | 44.6 | 5.99 |
| lora_attn_r16 | 2.16 | 2.4947 | 42.9 | 5.96 |

## bf16 pilotuyla tutarlılık

Pilotta biten 6 deneyin hepsi fp16 ile en fazla 0.0013 farkla aynı çıktı
(ör. full_ft 1.8191 → 1.8197, lora_all_r16 2.1816 → 2.1816). fp16'ya geçiş sonuçları
değiştirmedi; sadece her deneyi ~3 saatten ~45-60 dakikaya indirdi.

## Karşılaştırmalar (tek seed, sadece val loss; kesin sonuç değil)

- **Attention mı MLP mi** (parametre eşit, ~6.6M): MLP 2.207 < attention 2.373.
- **Alt mı üst mü** (4.4M): üst 2.339 < alt 2.447.
- **Rank:** r4 2.347 → r16 2.182 → r64 1.980.
- **Aynı bütçe (~2.2M):** tüm modüllere r4 (2.347) < sadece attention r16 (2.495).
