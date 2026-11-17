#!/usr/bin/env bash
# LoRA fine-tune a small model on your Mac with MLX.
# Usage: bash finetune/train_lora.sh [model] [iters] [adapter_dir]
# Takes roughly 20-60 min for a 1.5B model on a 16 GB M-series Mac.
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"   # swap for the newest small instruct model you like
ITERS="${2:-1000}"
ADAPTER="${3:-adapters/v1}"

if [ ! -f data/finetune/train.jsonl ]; then
  echo "No dataset yet. Run: uv run python -m finetune.make_dataset" >&2
  exit 1
fi

mkdir -p "$ADAPTER"
echo "Training $MODEL for $ITERS iterations -> $ADAPTER"

uv run mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data data/finetune \
  --iters "$ITERS" \
  --batch-size 4 \
  --num-layers 16 \
  --learning-rate 1e-4 \
  --steps-per-report 20 \
  --steps-per-eval 100 \
  --max-seq-length 2048 \
  --adapter-path "$ADAPTER" \
  2>&1 | tee "$ADAPTER/train.log"

# Save the settings next to the adapter so every run is reproducible.
cat > "$ADAPTER/run.json" <<EOF
{"model": "$MODEL", "iters": $ITERS, "batch_size": 4, "num_layers": 16, "learning_rate": 1e-4}
EOF

echo
echo "Done. Compare with:  uv run python -m finetune.compare --model $MODEL --adapter $ADAPTER"
echo "Fuse for serving:     uv run mlx_lm.fuse --model $MODEL --adapter-path $ADAPTER --save-path models/cart-parser-v1"
