# v1: sohbet token'ları dondurulmuş koşu (seed 42)

Bu koşuda LoRA deneylerinde `<|im_start|>` / `<|im_end|>` embedding satırları dondurulmuştu.
Qwen2.5-0.5B base'de bu satırlar eğitilmemiş ve ~290 kullanılmayan token'ın satırıyla birebir aynı
(embedding ve lm_head paylaşımlı). Sonuç: LoRA modelleri `<|im_end|>` üretemedi, cevaplarını hiç
bitiremedi (bitirme oranı %0) ve tekrar döngüsüne girdi. Full FT embedding'i de eğittiği için
%75 oranında bitirebildi.

Düzeltme (v2): LoRA'da sadece bu 2 satır da eğitiliyor (`trainable_token_indices`, +1792 parametre).

- Eğitim: https://www.kaggle.com/code/aysenursolmaz/notebook0473565067
- Değerlendirme: https://www.kaggle.com/code/aysenursolmaz/notebooke0562ec221
- `seed42_fp16.md`: eğitim tablosu, `summary.md`: değerlendirme tablosu, `samples.md`: örnek cevaplar
