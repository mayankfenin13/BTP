#!/bin/bash
# Run SISA CIFAR-10 with moderate speedup (~10x) by affecting more shards
# This script reruns unlearning with 60% of shards affected instead of 20%

set -e  # Exit on error

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="runs/all_datasets"
SISA_DIR="${BASE_DIR}/sisa_cifar10_s5x30_e30"

echo "=========================================="
echo "SISA CIFAR-10 Unlearning (Moderate Speedup)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Affecting 60% of shards (3 out of 5)"
echo "  - Target speedup: ~10x (instead of 1290x)"
echo "  - Unlearn fraction: 5%"
echo ""

# Check if training has been done
if [ ! -d "$REPO_ROOT/$SISA_DIR" ] || [ ! -f "$REPO_ROOT/$SISA_DIR/summary.json" ]; then
    echo "ERROR: SISA training not found at $REPO_ROOT/$SISA_DIR"
    echo "Please run training first:"
    echo "  python sisa/train_sisa.py --dataset cifar10 --epochs 30 --shards 5 --slices 30 --batch-size 128 --lr 0.001 --limit-per-class 2000 --out $REPO_ROOT/$SISA_DIR"
    exit 1
fi

echo "Running SISA unlearning with moderate shard affectation..."
echo "  Using modified unlearn script with --affect-frac-shards 0.6"
echo ""

python sisa/unlearn_sisa_moderate.py \
    --run-dir "$REPO_ROOT/$SISA_DIR" \
    --unlearn-frac 0.05 \
    --affect-frac-shards 0.6

if [ $? -eq 0 ]; then
    echo ""
    echo "Generating plots..."
    python common/plot.py --run-dir "$REPO_ROOT/$SISA_DIR"
    
    echo ""
    echo "✓ SISA CIFAR-10 unlearning complete!"
    echo ""
    echo "Results:"
    if [ -f "$REPO_ROOT/$SISA_DIR/metrics/unlearn_summary.json" ]; then
        python -c "
import json
d = json.load(open('$REPO_ROOT/$SISA_DIR/metrics/unlearn_summary.json'))
print(f'  Affected Shards: {d.get(\"num_affected_shards\", \"N/A\")} / 5')
print(f'  Baseline Time:   {d.get(\"baseline_train_time_s\", \"N/A\"):.2f}s')
print(f'  SISA Time:       {d.get(\"sisa_total_time_s\", \"N/A\"):.2f}s')
print(f'  Speedup:         {d.get(\"speedup_train_only\", \"N/A\"):.2f}x')
print(f'  Baseline Acc:    {d.get(\"baseline_acc\", \"N/A\"):.4f}')
print(f'  SISA Acc:        {d.get(\"ensemble_acc_after_unlearn\", \"N/A\"):.4f}')
"
    fi
else
    echo "✗ SISA unlearning failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="

