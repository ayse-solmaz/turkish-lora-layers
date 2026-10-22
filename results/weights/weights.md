# Ağırlık analizi

Etkin rank: ΔW enerjisinin %90'ını taşıyan tekil değer sayısı (katman/modül ortalaması).

| deney | ort. göreli değişim | alt yarı / üst yarı | attention / MLP | etkin rank (ort.) | verilen rank |
|---|---:|---:|---:|---:|---:|
| full_ft | 1.76% | 1.68% / 1.84% | 1.67% / 1.88% | 279.2 | sınırsız |
| lora_all_r16 | 3.43% | 3.20% / 3.67% | 3.22% / 3.72% | 11.0 | 16 |
| lora_all_r4 | 2.09% | 1.92% / 2.25% | 1.90% / 2.34% | 3.2 | 4 |
| lora_all_r64 | 5.49% | 5.15% / 5.82% | 5.36% / 5.66% | 36.2 | 64 |
| lora_attn_r16 | 4.76% | 4.41% / 5.11% | 4.76% / nan% | 9.2 | 16 |
| lora_attn_r49 | 6.97% | 6.43% / 7.51% | 6.97% / nan% | 22.4 | 49 |
| lora_lower_r16 | 3.69% | 3.69% / nan% | 3.39% / 4.10% | 10.8 | 16 |
| lora_mlp_r16 | 3.82% | 3.47% / 4.17% | nan% / 3.82% | 10.9 | 16 |
| lora_upper_r16 | 3.95% | nan% / 3.95% | 3.63% / 4.37% | 10.2 | 16 |
