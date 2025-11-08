#!/bin/bash
# Check progress of CIFAR-10 experiments

SISA_DIR="runs/full_datasets_optimized/sisa_cifar10_s10x40_e40"
ARCANE_DIR="runs/full_datasets_optimized/arcane_cifar10_b40_e40"

echo "=========================================="
echo "CIFAR-10 Training Progress"
echo "=========================================="
echo ""

# SISA Progress
echo "SISA Progress:"
if [ -f "$SISA_DIR/summary.json" ]; then
    echo "  Status: ✓ COMPLETED"
    python -c "import json; d=json.load(open('$SISA_DIR/summary.json')); print(f'  Ensemble Accuracy: {d.get(\"ensemble_acc\", \"N/A\"):.4f}')"
else
    completed=0
    for i in {0..9}; do
        if [ -f "$SISA_DIR/checkpoints/shard_$i/slice_39_post.pt" ] 2>/dev/null; then
            completed=$((completed + 1))
        fi
    done
    echo "  Status: In Progress ($completed/10 shards complete)"
    
    # Show current shard progress
    for i in {0..9}; do
        count=$(find "$SISA_DIR/checkpoints/shard_$i" -name "*_post.pt" 2>/dev/null | wc -l)
        if [ $count -gt 0 ] && [ $count -lt 40 ]; then
            echo "    Shard $i: $count/40 slices"
        fi
    done
fi

echo ""
echo "Arcane Progress:"
if [ -f "$ARCANE_DIR/summary.json" ]; then
    echo "  Status: ✓ COMPLETED"
    python -c "import json; d=json.load(open('$ARCANE_DIR/summary.json')); print(f'  Accuracy: {d.get(\"arcane_acc\", \"N/A\"):.4f}')"
else
    count=$(find "$ARCANE_DIR/checkpoints" -name "*_post.pt" 2>/dev/null | wc -l)
    if [ $count -gt 0 ]; then
        echo "  Status: In Progress ($count/40 blocks complete)"
    else
        echo "  Status: Not started or just started"
    fi
fi

echo ""
echo "=========================================="
echo "Process Status:"
ps aux | grep -E "train_sisa|train_arcane" | grep -v grep | awk '{print "  " $11 " " $12 " " $13 " " $14 " " $15 " " $16 " " $17 " " $18 " " $19 " " $20}'
echo "=========================================="

