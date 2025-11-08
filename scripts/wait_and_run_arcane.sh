#!/bin/bash
# Wait for SISA to complete, then run Arcane CIFAR-10

SISA_DIR="runs/full_datasets_optimized/sisa_cifar10_s10x40_e40"
ARCANE_SCRIPT="scripts/run_arcane_cifar10.sh"

echo "Waiting for SISA CIFAR-10 training to complete..."
echo ""

# Wait for SISA to finish (check every 30 seconds)
while [ ! -f "$SISA_DIR/summary.json" ]; do
    # Count completed shards
    completed=0
    for i in {0..9}; do
        if [ -f "$SISA_DIR/checkpoints/shard_$i/slice_39_post.pt" ]; then
            completed=$((completed + 1))
        fi
    done
    echo "Progress: $completed/10 shards complete ($(date +%H:%M:%S))"
    sleep 30
done

echo ""
echo "✓ SISA training completed!"
echo ""

# Run SISA unlearning
echo "Running SISA unlearning..."
python sisa/unlearn_sisa.py \
    --run-dir "$SISA_DIR" \
    --unlearn-frac 0.05

echo ""
echo "Generating SISA plots..."
python common/plot.py --run-dir "$SISA_DIR"

echo ""
echo "=========================================="
echo "Now running Arcane CIFAR-10..."
echo "=========================================="
echo ""

# Run Arcane
bash "$ARCANE_SCRIPT"

echo ""
echo "=========================================="
echo "All CIFAR-10 experiments complete!"
echo "=========================================="

