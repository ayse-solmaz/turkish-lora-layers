# Sonuç özeti

| deney | param (M) | TR test loss | EN ppl | ARC-Easy | TR→TR | EN→EN | talimat | tekrar | bitirme |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full FT | 494.03 | 2.743 | 17.25 | 0.646 | 100% | 80% | 60% | 0.22 | 75% |
| LoRA tümü r64 | 35.19 | 2.956 | 17.09 | 0.632 | 100% | 80% | 20% | 0.60 | 0% |
| LoRA tümü r16 | 8.80 | 3.078 | 15.59 | 0.656 | 100% | 80% | 20% | 0.60 | 0% |
| LoRA MLP r16 | 6.64 | 3.098 | 15.51 | 0.658 | 94% | 80% | 20% | 0.59 | 0% |
| LoRA tümü r4 | 2.20 | 3.181 | 15.15 | 0.652 | 100% | 60% | 20% | 0.64 | 0% |
| LoRA attention r49 | 6.62 | 3.185 | 15.90 | 0.642 | 100% | 80% | 20% | 0.66 | 0% |
| LoRA üst yarı r16 | 4.40 | 3.201 | 15.44 | 0.622 | 100% | 100% | 0% | 0.68 | 0% |
| LoRA alt yarı r16 | 4.40 | 3.212 | 15.28 | 0.644 | 100% | 100% | 0% | 0.71 | 0% |
| LoRA attention r16 | 2.16 | 3.242 | 15.42 | 0.668 | 100% | 100% | 20% | 0.68 | 0% |
| base (eğitimsiz) | 0.00 | 3.565 | 14.74 | 0.636 | 89% | 100% | 20% | 0.75 | 0% |

- **TR→TR / EN→EN:** Türkçe/İngilizce soruya aynı dilde cevap oranı (dil kayması)
- **talimat:** madde sayısı, cümle sayısı, kelime sınırı kontrollerini geçme oranı
- **tekrar:** tekrar eden 4-gram oranı (0 = hiç tekrar yok)
- **bitirme:** cevabı 256 token dolmadan kendi bitirme oranı
