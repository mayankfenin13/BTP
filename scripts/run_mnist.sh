#!/bin/bash
# Optimized MNIST experiments for SISA and Arcane
# Focus: Excellent speedup and accuracy

set -e  # Exit on error

BASE_DIR="runs/mnist_optimized"
mkdir -p "$BASE_DIR"

echo "=========================================="
echo "MNIST Optimized Experiments"
echo "=========================================="
echo ""

# SISA Configuration: 16 shards × 20 slices = 20 epochs
# More shards = better speedup (fewer affected shards)
# More slices = more epochs = better accuracy
echo "Training SISA on MNIST..."
echo "  Shards: 16, Slices: 20, Epochs: 20"
echo "  Full dataset (no limit)"
echo ""

SISA_DIR="${BASE_DIR}/sisa_mnist_s16x20_e20"
python sisa/train_sisa.py \
    --dataset mnist \
    --epochs 20 \
    --shards 16 \
    --slices 20 \
    --batch-size 128 \
    --lr 0.001 \
    --out "$SISA_DIR"

echo ""
echo "Running SISA unlearning (5% unlearn)..."
python sisa/unlearn_sisa.py \
    --run-dir "$SISA_DIR" \
    --unlearn-frac 0.05

echo ""
echo "Generating SISA plots..."
python common/plot.py --run-dir "$SISA_DIR"

echo ""
echo "✓ SISA MNIST complete!"
echo ""

# Arcane Configuration: 20 blocks = 20 epochs
# More blocks = more epochs = better accuracy
echo "Training Arcane on MNIST..."
echo "  Blocks: 20, Epochs: 20"
echo "  Full dataset (no limit)"
echo ""

ARCANE_DIR="${BASE_DIR}/arcane_mnist_b20_e20"
python arcane/train_arcane.py \
    --dataset mnist \
    --epochs 20 \
    --blocks 20 \
    --batch-size 128 \
    --lr 0.001 \
    --out "$ARCANE_DIR"

echo ""
echo "Running Arcane unlearning (5% unlearn)..."
python arcane/unlearn_arcane.py \
    --run-dir "$ARCANE_DIR" \
    --unlearn-frac 0.05

echo ""
echo "Generating Arcane plots..."
python common/plot.py --run-dir "$ARCANE_DIR"

echo ""
echo "✓ Arcane MNIST complete!"
echo ""

# Print summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="

if [ -f "$SISA_DIR/summary.json" ]; then
    echo ""
    echo "SISA Training:"
    python -c "import json; d=json.load(open('$SISA_DIR/summary.json')); print(f'  Ensemble Accuracy: {d.get(\"ensemble_acc\", \"N/A\"):.4f}')"
fi

if [ -f "$SISA_DIR/metrics/unlearn_summary.json" ]; then
    echo ""
    echo "SISA Unlearning:"
    python -c "import json; d=json.load(open('$SISA_DIR/metrics/unlearn_summary.json')); print(f'  Baseline Accuracy: {d.get(\"baseline_acc\", \"N/A\"):.4f}'); print(f'  SISA Accuracy: {d.get(\"ensemble_acc_after_unlearn\", \"N/A\"):.4f}'); print(f'  Speedup (train): {d.get(\"speedup_train_only\", \"N/A\"):.2f}x'); print(f'  Affected Shards: {d.get(\"num_affected_shards\", \"N/A\")} / 16')"
fi

if [ -f "$ARCANE_DIR/summary.json" ]; then
    echo ""
    echo "Arcane Training:"
    python -c "import json; d=json.load(open('$ARCANE_DIR/summary.json')); print(f'  Accuracy: {d.get(\"arcane_acc\", \"N/A\"):.4f}')"
fi

if [ -f "$ARCANE_DIR/metrics/unlearn_summary.json" ]; then
    echo ""
    echo "Arcane Unlearning:"
    python -c "import json; d=json.load(open('$ARCANE_DIR/metrics/unlearn_summary.json')); print(f'  Baseline Accuracy: {d.get(\"baseline_acc\", \"N/A\"):.4f}'); print(f'  Arcane Accuracy: {d.get(\"arcane_acc_after_unlearn\", \"N/A\"):.4f}'); print(f'  Speedup: {d.get(\"speedup\", \"N/A\"):.2f}x'); print(f'  Tail Blocks: {d.get(\"affected_tail_blocks\", \"N/A\")} / 20')"
fi

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

