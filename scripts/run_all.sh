#!/usr/bin/env bash
# Tüm deneyleri sırayla eğitir (Kaggle/Colab GPU'sunda).
# Kullanım: bash scripts/run_all.sh [seed]
set -e
SEED=${1:-42}

[ -f data/train.jsonl ] || python src/prepare_data.py

for EXP in lora_all_r16 lora_lower_r16 lora_upper_r16 \
           lora_attn_r16 lora_attn_r49 lora_mlp_r16 \
           lora_all_r4 lora_all_r64 full_ft; do
  if [ -f "outputs/seed$SEED/$EXP/log.json" ]; then
    echo "== $EXP zaten bitmiş, atlanıyor"; continue
  fi
  echo "== $EXP (seed $SEED)"
  python src/train.py --exp "$EXP" --seed "$SEED" --out_dir "outputs/seed$SEED"
done
