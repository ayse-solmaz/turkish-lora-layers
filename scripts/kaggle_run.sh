#!/usr/bin/env bash
# Kaggle'da her şeyi tek komutla çalıştırır: kurulum, veri, 9 deney, özet.
# Kaggle notebook hücresine yazılacak tek şey:
#   !git clone https://github.com/ayse-solmaz/turkish-lora-layers.git && cd turkish-lora-layers && bash scripts/kaggle_run.sh
SEED=${1:-42}

echo "================ KURULUM"
pip install -q -U "transformers>=4.56" "peft>=0.17" "datasets>=3.0" accelerate 2>&1 | grep -v bigframes
# Kaggle'ın eski torchao'su yeni peft ile çakışıyor; kullanmıyoruz
pip uninstall -y -q torchao
nvidia-smi -L

echo "================ VERİ"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1   # print'ler beklemeden loga düşsün
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/prepare_data.py || exit 1

echo "================ EĞİTİM (seed $SEED)"
mkdir -p logs
GPU0="full_ft lora_all_r16 lora_lower_r16 lora_attn_r16 lora_all_r4"
GPU1="lora_all_r64 lora_upper_r16 lora_attn_r49 lora_mlp_r16"

# Çıktı hem dosyaya hem ekrana gider; satır başındaki [gpuX] hangi GPU olduğunu söyler
run() {  # run <gpu> <deneyler>
  CUDA_VISIBLE_DEVICES=$1 bash scripts/run_all.sh "$SEED" "$2" 2>&1 \
    | grep --line-buffered -v -E "Warning: You are sending|Loading weights|Writing model shards" \
    | sed -u "s/^/[gpu$1] /" | tee "logs/gpu$1.log"
}

if [ "$(nvidia-smi -L | wc -l)" -ge 2 ]; then
  run 0 "$GPU0" &
  run 1 "$GPU1" &
  wait
else
  run 0 "$GPU0 $GPU1"
fi

echo "================ ÖZET"
grep -h -E "^\[gpu.\] (== |!!|Bitti)" logs/*.log
python - <<'EOF'
import glob, json
rows = []
for p in sorted(glob.glob("outputs/seed*/*/log.json")):
    log = json.load(open(p))
    rows.append((log["val_loss"][-1][1], log["exp"], log["trainable_params"] / 1e6,
                 log["train_seconds"] / 60, log.get("peak_memory_gb")))
print(f"{'deney':<16} {'param(M)':>9} {'val loss':>9} {'süre(dk)':>9} {'bellek':>7}")
for loss, exp, params, mins, mem in sorted(rows):
    print(f"{exp:<16} {params:>9.2f} {loss:>9.4f} {mins:>9.1f} {mem!s:>7}")
if not rows:
    print("Hiç deney tamamlanmadı; yukarıdaki loga bak.")
EOF
