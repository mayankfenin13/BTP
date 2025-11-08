#!/bin/bash
# Run Arcane CIFAR-10 after SISA completes
# This script runs Arcane training and unlearning

set -e

BASE_DIR="runs/full_datasets_optimized"
ARCANE_DIR="${BASE_DIR}/arcane_cifar10_b40_e40"

echo "=========================================="
echo "Training Arcane on CIFAR-10..."
echo "  Blocks: 40, Epochs: 40"
echo "  Full dataset (50,000 samples)"
echo ""

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
    
    # Print summary
    if [ -f "$ARCANE_DIR/metrics/unlearn_summary.json" ]; then
        echo ""
        echo "Arcane Results:"
        python -c "import json; d=json.load(open('$ARCANE_DIR/metrics/unlearn_summary.json')); print(f'  Baseline Accuracy: {d.get(\"baseline_acc\", \"N/A\"):.4f}'); print(f'  Arcane Accuracy: {d.get(\"arcane_acc_after_unlearn\", \"N/A\"):.4f}'); print(f'  Speedup: {d.get(\"speedup\", \"N/A\"):.2f}x')"
    fi
else
    echo "✗ Arcane CIFAR-10 training failed!"
    exit 1
fi

