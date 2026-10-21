#!/usr/bin/env bash
# Kaggle'da değerlendirme: eğitim notebook'unun çıktısı "Input" olarak eklenmiş olmalı.
# Kaggle notebook hücresine yazılacak tek şey:
#   !git clone https://github.com/ayse-solmaz/turkish-lora-layers.git && cd turkish-lora-layers && bash scripts/kaggle_eval.sh
SEED=${1:-42}

echo "================ KURULUM"
pip install -q -U "transformers>=4.56" "peft>=0.17" "datasets>=3.0" accelerate 2>&1 | grep -v bigframes
pip uninstall -y -q torchao
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1

# Eğitilmiş modeller: RUNS_DIR verilmişse orası (kaggle_run.sh aynı notebook'ta çağırınca),
# yoksa Input olarak eklenen eğitim notebook'unda ara: /kaggle/input/<notebook>/.../outputs/seed42
RUNS=${RUNS_DIR:-$(find /kaggle/input -type d -path "*outputs/seed$SEED" 2>/dev/null | head -1)}
if [ -z "$RUNS" ]; then
  echo "!! Eğitilmiş modeller bulunamadı. Sağ panelden Input -> eğitim notebook'unu ekle."
  exit 1
fi
echo "Modeller: $RUNS"
ls "$RUNS"

echo "================ VERİ"
python src/prepare_data.py | tail -4

echo "================ DEĞERLENDİRME"
GPU0="base full_ft lora_all_r16 lora_lower_r16 lora_attn_r16"
GPU1="lora_all_r4 lora_all_r64 lora_upper_r16 lora_attn_r49 lora_mlp_r16"
ev() {  # ev <gpu> <deneyler>
  CUDA_VISIBLE_DEVICES=$1 python src/evaluate.py --runs_dir "$RUNS" --exps $2 2>&1 \
    | grep --line-buffered -v -E "Warning: You are sending|Loading weights" \
    | sed -u "s/^/[gpu$1] /"
}
if [ "$(nvidia-smi -L | wc -l)" -ge 2 ]; then
  ev 0 "$GPU0" &
  ev 1 "$GPU1" &
  wait
else
  ev 0 "$GPU0 $GPU1"
fi

echo "================ PAKET"
# Grafikler için gereken her şey tek küçük zip'te: eğitim logları + değerlendirme sonuçları
mkdir -p bundle/train_logs bundle/eval
for d in "$RUNS"/*/; do
  e=$(basename "$d"); [ -f "$d/log.json" ] && cp "$d/log.json" "bundle/train_logs/$e.json"
done
cp results/eval/*.json bundle/eval/
(cd bundle && zip -q -r /kaggle/working/eval_bundle.zip .)
ls -la /kaggle/working/eval_bundle.zip
echo "Bitti: Output sekmesinden eval_bundle.zip dosyasını indir."
