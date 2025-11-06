#!/bin/bash
# Comprehensive script to run SISA and ARCANE on Fashion-MNIST, CIFAR-10, and SVHN
# This script runs experiments with optimized hyperparameters for good accuracy

set -e  # Exit on error

BASE_DIR="runs/all_datasets"
mkdir -p "$BASE_DIR"

echo "=========================================="
echo "Running Experiments on All Datasets"
echo "=========================================="
echo ""

# Function to run SISA
run_sisa() {
    local dataset=$1
    local epochs=$2
    local shards=$3
    local slices=$4
    local batch_size=$5
    local lr=$6
    local limit=$7
    local out_dir="${BASE_DIR}/sisa_${dataset}_s${shards}x${slices}_e${epochs}"
    
    echo "Training SISA on ${dataset}..."
    echo "  Epochs: ${epochs}, Shards: ${shards}, Slices: ${slices}"
    echo "  Batch size: ${batch_size}, LR: ${lr}"
    if [ ! -z "$limit" ]; then
        echo "  Limit per class: ${limit}"
    fi
    
    python sisa/train_sisa.py \
        --dataset "$dataset" \
        --epochs "$epochs" \
        --shards "$shards" \
        --slices "$slices" \
        --batch-size "$batch_size" \
        --lr "$lr" \
        ${limit:+--limit-per-class "$limit"} \
        --out "$out_dir"
    
    echo "Running SISA unlearning (1% unlearn)..."
    python sisa/unlearn_sisa.py \
        --run-dir "$out_dir" \
        --unlearn-frac 0.01
    
    echo "✓ SISA ${dataset} complete!"
    echo ""
}

# Function to run ARCANE
run_arcane() {
    local dataset=$1
    local epochs=$2
    local blocks=$3
    local batch_size=$4
    local lr=$5
    local limit=$6
    local out_dir="${BASE_DIR}/arcane_${dataset}_b${blocks}_e${epochs}"
    
    echo "Training ARCANE on ${dataset}..."
    echo "  Epochs: ${epochs}, Blocks: ${blocks}"
    echo "  Batch size: ${batch_size}, LR: ${lr}"
    if [ ! -z "$limit" ]; then
        echo "  Limit per class: ${limit}"
    fi
    
    python arcane/train_arcane.py \
        --dataset "$dataset" \
        --epochs "$epochs" \
        --blocks "$blocks" \
        --batch-size "$batch_size" \
        --lr "$lr" \
        ${limit:+--limit-per-class "$limit"} \
        --out "$out_dir"
    
    echo "Running ARCANE unlearning (1% unlearn)..."
    python arcane/unlearn_arcane.py \
        --run-dir "$out_dir" \
        --unlearn-frac 0.01
    
    echo "✓ ARCANE ${dataset} complete!"
    echo ""
}

# ============================================
# Fashion-MNIST (28x28 grayscale, similar to MNIST)
# ============================================
echo "=========================================="
echo "FASHION-MNIST Experiments"
echo "=========================================="
echo ""

# Fashion-MNIST: Use LeNet, similar to MNIST but need more epochs for good accuracy
run_sisa "fashionmnist" 15 5 10 128 0.001 ""
run_arcane "fashionmnist" 15 6 128 0.001 ""

# ============================================
# CIFAR-10 (32x32 RGB, more challenging)
# ============================================
echo "=========================================="
echo "CIFAR-10 Experiments"
echo "=========================================="
echo ""

# CIFAR-10: Uses ResNet18, needs more epochs and lower batch size
# Note: CIFAR-10 is large, so we use limit for faster runs
# For full dataset, remove --limit-per-class
run_sisa "cifar10" 30 5 10 64 0.001 1000
run_arcane "cifar10" 30 6 64 0.001 1000

# ============================================
# SVHN (32x32 RGB, street view digits)
# ============================================
echo "=========================================="
echo "SVHN Experiments"
echo "=========================================="
echo ""

# SVHN: Uses ResNet18, similar to CIFAR-10
run_sisa "svhn" 25 5 10 64 0.001 ""
run_arcane "svhn" 25 6 64 0.001 ""

# ============================================
# Summary
# ============================================
echo "=========================================="
echo "All Experiments Complete!"
echo "=========================================="
echo ""
echo "Results saved in: $BASE_DIR"
echo ""
echo "To view results, check:"
echo "  - summary.json: Final accuracy"
echo "  - metrics/unlearn_summary.json: Unlearning results"
echo ""

# Generate summary table
echo "Generating accuracy summary..."
python3 << 'EOF'
import json
import os
import glob

base_dir = "runs/all_datasets"
results = []

for run_dir in glob.glob(f"{base_dir}/*"):
    if not os.path.isdir(run_dir):
        continue
    
    summary_path = os.path.join(run_dir, "summary.json")
    unlearn_path = os.path.join(run_dir, "metrics", "unlearn_summary.json")
    
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
            method = "SISA" if "sisa" in run_dir else "ARCANE"
            dataset = os.path.basename(run_dir).split("_")[1]
            
            acc_key = "ensemble_acc" if "sisa" in run_dir else "arcane_acc"
            if acc_key in summary:
                results.append({
                    "method": method,
                    "dataset": dataset,
                    "accuracy": f"{summary[acc_key]:.4f}",
                    "unlearn_acc": "N/A"
                })
                
                if os.path.exists(unlearn_path):
                    with open(unlearn_path) as f2:
                        unlearn = json.load(f2)
                        unlearn_key = "ensemble_acc_after_unlearn" if "sisa" in run_dir else "arcane_acc_after_unlearn"
                        if unlearn_key in unlearn:
                            results[-1]["unlearn_acc"] = f"{unlearn[unlearn_key]:.4f}"

print("\n" + "="*60)
print("ACCURACY SUMMARY")
print("="*60)
print(f"{'Method':<8} {'Dataset':<15} {'Initial Acc':<12} {'After Unlearn':<12}")
print("-"*60)
for r in sorted(results, key=lambda x: (x["dataset"], x["method"])):
    print(f"{r['method']:<8} {r['dataset']:<15} {r['accuracy']:<12} {r['unlearn_acc']:<12}")
print("="*60)
EOF

