# Pilot koşu (seed 42, Kaggle T4 x2, bf16)

Bu koşuda T4'te bf16 yazılımla taklit edildiği için her deney ~3 saat sürdü ve
notebook 12 saatlik sınıra takıldı. 9 deneyden 6'sı bitti. Sonraki koşular fp16
kullanır; bu sayılar fp16 sonuçlarıyla tutarlılık kontrolü için saklanıyor.

Ham loglar: `gpu0.log`, `gpu1.log`

| deney | eğitilen param | son val loss | süre (sn) |
|---|---:|---:|---:|
| (eğitimsiz base) | 0 | 2.8208 | - |
| full_ft | 494.0M | **1.8191** | 12147 |
| lora_all_r64 | 35.2M | 1.9804 | 12488 |
| lora_all_r16 | 8.8M | 2.1816 | 11967 |
| lora_upper_r16 | 4.4M | 2.3398 | 11059 |
| lora_attn_r49 | 6.6M | 2.3732 | 10511 |
| lora_lower_r16 | 4.4M | 2.4471 | 10833 |
| lora_attn_r16 | 2.2M | bitmedi | |
| lora_mlp_r16 | 6.6M | bitmedi | |
| lora_all_r4 | 2.2M | bitmedi | |
