#!/bin/bash
# Run CIFAR-10 experiments for SISA and Arcane only
# Fashion-MNIST was already working, so we just need CIFAR-10

set -e  # Exit on error

BASE_DIR="runs/full_datasets_optimized"
mkdir -p "$BASE_DIR"

echo "=========================================="
echo "CIFAR-10 Experiments (SISA + Arcane)"
echo "=========================================="
echo ""

# SISA Configuration: 10 shards × 40 slices = 40 epochs
echo "Training SISA on CIFAR-10..."
echo "  Shards: 10, Slices: 40, Epochs: 40"
echo "  Full dataset (50,000 samples)"
echo ""

SISA_DIR="${BASE_DIR}/sisa_cifar10_s10x40_e40"
python sisa/train_sisa.py \
    --dataset cifar10 \
    --epochs 40 \
    --shards 10 \
    --slices 40 \
    --batch-size 128 \
    --lr 0.001 \
    --out "$SISA_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "Running SISA unlearning (5% unlearn)..."
    python sisa/unlearn_sisa.py \
        --run-dir "$SISA_DIR" \
        --unlearn-frac 0.05
    
    echo ""
    echo "Generating SISA plots..."
    python common/plot.py --run-dir "$SISA_DIR"
    
    echo ""
    echo "✓ SISA CIFAR-10 complete!"
else
    echo "✗ SISA CIFAR-10 training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo ""

# Arcane Configuration: 40 blocks = 40 epochs
echo "Training Arcane on CIFAR-10..."
echo "  Blocks: 40, Epochs: 40"
echo "  Full dataset (50,000 samples)"
echo ""

ARCANE_DIR="${BASE_DIR}/arcane_cifar10_b40_e40"
python arcane/train_arcane.py \
    --dataset cifar10 \
    --epochs 40 \
    --blocks 40 \
    --batch-size 128 \
    --lr 0.001 \
    --out "$ARCANE_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "Running Arcane unlearning (5% unlearn)..."
    python arcane/unlearn_arcane.py \
        --run-dir "$ARCANE_DIR" \
        --unlearn-frac 0.05
    
    echo ""
    echo "Generating Arcane plots..."
    python common/plot.py --run-dir "$ARCANE_DIR"
    
    echo ""
    echo "✓ Arcane CIFAR-10 complete!"
else
    echo "✗ Arcane CIFAR-10 training failed!"
    exit 1
fi

# Print summary
echo ""
echo "=========================================="
echo "CIFAR-10 SUMMARY"
echo "=========================================="

if [ -f "$SISA_DIR/summary.json" ]; then
    echo ""
    echo "SISA Training:"
    python -c "import json; d=json.load(open('$SISA_DIR/summary.json')); print(f'  Ensemble Accuracy: {d.get(\"ensemble_acc\", \"N/A\"):.4f}')"
fi

if [ -f "$SISA_DIR/metrics/unlearn_summary.json" ]; then
    echo ""
    echo "SISA Unlearning:"
    python -c "import json; d=json.load(open('$SISA_DIR/metrics/unlearn_summary.json')); print(f'  Baseline Accuracy: {d.get(\"baseline_acc\", \"N/A\"):.4f}'); print(f'  SISA Accuracy: {d.get(\"ensemble_acc_after_unlearn\", \"N/A\"):.4f}'); print(f'  Speedup (train): {d.get(\"speedup_train_only\", \"N/A\"):.2f}x'); print(f'  Affected Shards: {d.get(\"num_affected_shards\", \"N/A\")} / 10')"
fi

if [ -f "$ARCANE_DIR/summary.json" ]; then
    echo ""
    echo "Arcane Training:"
    python -c "import json; d=json.load(open('$ARCANE_DIR/summary.json')); print(f'  Accuracy: {d.get(\"arcane_acc\", \"N/A\"):.4f}')"
fi

if [ -f "$ARCANE_DIR/metrics/unlearn_summary.json" ]; then
    echo ""
    echo "Arcane Unlearning:"
    python -c "import json; d=json.load(open('$ARCANE_DIR/metrics/unlearn_summary.json')); print(f'  Baseline Accuracy: {d.get(\"baseline_acc\", \"N/A\"):.4f}'); print(f'  Arcane Accuracy: {d.get(\"arcane_acc_after_unlearn\", \"N/A\"):.4f}'); print(f'  Speedup: {d.get(\"speedup\", \"N/A\"):.2f}x'); print(f'  Tail Blocks: {d.get(\"affected_tail_blocks\", \"N/A\")} / 40')"
fi

echo ""
echo "=========================================="
echo "All CIFAR-10 experiments completed!"
echo "=========================================="

