# Sonuç özeti

| deney | param (M) | TR test loss | EN ppl | ARC-Easy | TR→TR | EN→EN | talimat | tekrar | bitirme |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full FT | 494.03 | 2.743 | 17.27 | 0.650 | 94% | 80% | 60% | 0.22 | 75% |
| LoRA tümü r64 | 35.19 | 2.864 | 17.02 | 0.626 | 94% | 60% | 0% | 0.11 | 75% |
| LoRA tümü r16 | 8.80 | 2.978 | 15.62 | 0.642 | 100% | 80% | 20% | 0.35 | 46% |
| LoRA MLP r16 | 6.64 | 2.996 | 15.52 | 0.652 | 94% | 80% | 40% | 0.31 | 54% |
| LoRA üst yarı r16 | 4.40 | 3.068 | 15.52 | 0.630 | 100% | 80% | 0% | 0.31 | 58% |
| LoRA tümü r4 | 2.20 | 3.080 | 15.16 | 0.650 | 94% | 80% | 20% | 0.40 | 46% |
| LoRA attention r49 | 6.63 | 3.086 | 15.96 | 0.640 | 94% | 100% | 0% | 0.33 | 46% |
| LoRA alt yarı r16 | 4.40 | 3.102 | 15.26 | 0.636 | 100% | 100% | 0% | 0.33 | 54% |
| LoRA attention r16 | 2.16 | 3.142 | 15.46 | 0.668 | 94% | 100% | 40% | 0.33 | 50% |
| base (eğitimsiz) | 0.00 | 3.565 | 14.74 | 0.636 | 89% | 100% | 20% | 0.75 | 0% |

- **TR→TR / EN→EN:** Türkçe/İngilizce soruya aynı dilde cevap oranı (dil kayması)
- **talimat:** madde sayısı, cümle sayısı, kelime sınırı kontrollerini geçme oranı
- **tekrar:** tekrar eden 4-gram oranı (0 = hiç tekrar yok)
- **bitirme:** cevabı 256 token dolmadan kendi bitirme oranı
