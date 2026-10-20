# Türkçe küçük modellerde LoRA'yı nereye koymalı?

**Performans, unutma ve öğrenmenin nerede gerçekleştiği**

Küçük bir base model (Qwen2.5-0.5B) Türkçe instruction verisiyle farklı yöntemlerle eğitilir ve karşılaştırılır:

| Deney | Ne değişiyor |
|---|---|
| Baseline | Eğitilmemiş base model |
| Referans | LoRA, tüm katmanlar + tüm modüller, r=16 |
| Alt / Üst | LoRA sadece alt yarı / sadece üst yarı katmanlarda |
| Attention / MLP | LoRA sadece attention / sadece MLP modüllerinde (parametre sayısı eşitlenmiş) |
| Rank | r = 4, 16, 64 |
| Full FT | Tüm ağırlıklar eğitilir |

## Ölçümler

- **Türkçe performans:** held-out loss, örnek cevaplar
- **Unutma:** İngilizce perplexity, genel benchmark
- **Öğrenme nerede oldu:** katman başına ağırlık değişimi, LoRA rank kullanımı (tekil değerler)
- **Davranış:** dil kayması (İngilizceye geçme oranı), tekrar oranı
- **Maliyet:** eğitim süresi, GPU belleği, eğitilen parametre sayısı

## Yapı

```
scripts/   tek seferlik analiz scriptleri
src/       veri, eğitim ve değerlendirme kodu
configs/   deney ayarları
results/   loglar, grafikler, tablolar
```

## Kurulum (yerel)

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

Eğitim Kaggle/Colab GPU'sunda çalışır; analiz scriptleri CPU'da çalışır.

## Kaggle'da çalıştırma

Yeni notebook, **GPU T4 x2** ve **Internet On**. Tek hücre:

```
# 1) Eğitim (9 deney, ~4 saat)
!git clone https://github.com/ayse-solmaz/turkish-lora-layers.git && cd turkish-lora-layers && bash scripts/kaggle_run.sh

# 2) Değerlendirme (eğitim notebook'unun çıktısı Input olarak eklenmiş olmalı)
!git clone https://github.com/ayse-solmaz/turkish-lora-layers.git && cd turkish-lora-layers && bash scripts/kaggle_eval.sh
```

Değerlendirme `eval_bundle.zip` üretir; yerelde grafikler:

```bash
python src/plots.py --bundle results/bundle_seed42
```
