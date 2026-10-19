#!/usr/bin/env bash
# Deneyleri sırayla eğitir (Kaggle/Colab GPU'sunda).
# Kullanım:
#   bash scripts/run_all.sh                       # tüm deneyler, seed 42
#   bash scripts/run_all.sh 42 "full_ft lora_all_r4"   # sadece seçilenler
#
# Bir deney çökerse diğerleri yine de koşar; sonunda çökenler listelenir.
SEED=${1:-42}
EXPS=${2:-"lora_all_r16 lora_lower_r16 lora_upper_r16 \
lora_attn_r16 lora_attn_r49 lora_mlp_r16 \
lora_all_r4 lora_all_r64 full_ft"}

[ -f data/train.jsonl ] || python src/prepare_data.py || exit 1

FAILED=""
for EXP in $EXPS; do
  if [ -f "outputs/seed$SEED/$EXP/log.json" ]; then
    echo "== $EXP zaten bitmiş, atlanıyor"; continue
  fi
  echo "== $EXP (seed $SEED)"
  python src/train.py --exp "$EXP" --seed "$SEED" --out_dir "outputs/seed$SEED" \
    || { echo "!! $EXP ÇÖKTÜ"; FAILED="$FAILED $EXP"; }
done

if [ -n "$FAILED" ]; then
  echo "!! Çöken deneyler:$FAILED"
  exit 1
fi
