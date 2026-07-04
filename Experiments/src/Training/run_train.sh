#!/bin/bash
# End-to-end CelebA training and evaluation pipeline.
#
# Trains NUM_MODELS U-Nets on complementary halves of CelebA (n images each),
# then runs the sample-split evaluation suite (score comparison, loss curves,
# paired generation, cosine-similarity aggregation).
#
# Prerequisites:
#   - Raw CelebA images at $RAW_CELEBA (img_align_celeba directory).
#   - Preprocessed tensor CelebA32.pt at Experiments/Data/CelebA/ (created by
#     src/Utils/preprocess_celeba.py if missing).
#
# Usage (standalone, not via SLURM):
#   cd Experiments/src/Training
#   bash run_train.sh                     # defaults: 15 models, n=1024
#   bash run_train.sh --models 3 --n 512  # override

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via env vars or --flag syntax parsed below)
# ---------------------------------------------------------------------------
NUM_MODELS="${NUM_MODELS:-15}"
N="${N:-1024}"
IMG_SIZE="${IMG_SIZE:-32}"
LR="${LR:-0.0001}"
OPTIM="${OPTIM:-Adam}"
NBASE="${NBASE:-32}"
BATCH="${BATCH:-512}"
EVAL_N="${EVAL_N:-50}"
NS="${NS:-1000}"

# Simple flag parser
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) NUM_MODELS="$2"; shift 2 ;;
        --n)      N="$2"; shift 2 ;;
        --size)   IMG_SIZE="$2"; shift 2 ;;
        --lr)     LR="$2"; shift 2 ;;
        --optim)  OPTIM="$2"; shift 2 ;;
        --nbase)  NBASE="$2"; shift 2 ;;
        *)        echo "Unknown option: $1"; exit 1 ;;
    esac
done

LAST_IDX=$((NUM_MODELS - 1))

# ---------------------------------------------------------------------------
# Resolve paths relative to Experiments/
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"     # Experiments/
TRAIN_DIR="$EXP_ROOT/src/Training"
GEN_DIR="$EXP_ROOT/src/Generation"
UTILS_DIR="$EXP_ROOT/src/Utils"
DATA_DIR="$EXP_ROOT/Data/CelebA"

# ---------------------------------------------------------------------------
# Step 0: preprocess CelebA if CelebA32.pt is missing
# ---------------------------------------------------------------------------
if [[ ! -f "$DATA_DIR/CelebA32.pt" ]]; then
    echo "=== Preprocessing CelebA ==="
    RAW_CELEBA="${RAW_CELEBA:-$EXP_ROOT/../data/img_align_celeba}"
    if [[ ! -d "$RAW_CELEBA" ]]; then
        echo "Error: raw CelebA directory not found at $RAW_CELEBA"
        echo "Set RAW_CELEBA=/path/to/img_align_celeba and re-run."
        exit 1
    fi
    cd "$UTILS_DIR"
    python preprocess_celeba.py \
        --raw-data-path "$RAW_CELEBA" \
        --output-path "$DATA_DIR/" \
        --size "$IMG_SIZE"
fi

# ---------------------------------------------------------------------------
# Step 1: train NUM_MODELS U-Nets
# ---------------------------------------------------------------------------
echo "=== Training $NUM_MODELS U-Net models (n=$N, size=$IMG_SIZE) ==="
cd "$TRAIN_DIR"

for INDEX in $(seq 0 "$LAST_IDX"); do
    echo "--- model $INDEX / $LAST_IDX ---"
    python run_Unet.py \
        -n "$N" \
        -i "$INDEX" \
        -s "$IMG_SIZE" \
        -LR "$LR" \
        -O "$OPTIM" \
        -W "$NBASE" \
        -t -1 \
        --index "$INDEX" \
        -se "$INDEX"
done

# ---------------------------------------------------------------------------
# Step 2: score comparison (compare_scores.py) at multiple diffusion times
# ---------------------------------------------------------------------------
echo "=== Score comparison (indices 0-$LAST_IDX) ==="
cd "$GEN_DIR"

for T in 50 100 150 200; do
    python compare_scores.py \
        -n "$N" -is 0 -ie "$LAST_IDX" \
        -s "$IMG_SIZE" -LR "$LR" -O "$OPTIM" -W "$NBASE" \
        -t "$T" -Ns "$NS" -B "$BATCH"
done

# ---------------------------------------------------------------------------
# Step 3: per-model test loss
# ---------------------------------------------------------------------------
echo "=== Per-model test loss ==="
for INDEX in $(seq 0 "$LAST_IDX"); do
    python loss_compute.py \
        -n "$N" -i "$INDEX" \
        -s "$IMG_SIZE" -LR "$LR" -O "$OPTIM" -W "$NBASE" \
        -B "$BATCH" --eval_N "$EVAL_N"
done

# ---------------------------------------------------------------------------
# Step 4: paired sample generation
# ---------------------------------------------------------------------------
echo "=== Sample-split inference ==="
python sample_split_inference.py \
    --scan_all_pairs \
    -n "$N" -s "$IMG_SIZE" -LR "$LR" -O "$OPTIM" -W "$NBASE" -B "$BATCH"

# ---------------------------------------------------------------------------
# Step 5: aggregate cosine similarity + loss
# ---------------------------------------------------------------------------
echo "=== Cosine-similarity / loss aggregation ==="
python cos_dis_aggregate.py \
    -n "$N" -s "$IMG_SIZE" -LR "$LR" -O "$OPTIM" -W "$NBASE" -B "$BATCH"

echo "=== Pipeline complete ==="