#!/usr/bin/env python3
"""
Optimized script to run SISA and ARCANE on MNIST
Focus on excellent speedup and accuracy
"""

import os
import subprocess
import sys
import json
from pathlib import Path

BASE_DIR = "runs/mnist_optimized"

# Optimized configurations for MNIST
# Strategy: More shards/blocks for speedup, more epochs for accuracy
MNIST_CONFIGS = {
    "sisa": {
        "epochs": 20,           # Metadata: total epochs = slices
        "shards": 16,           # More shards = better speedup (fewer affected shards)
        "slices": 20,           # 20 slices = 20 epochs total (1 per slice)
        "batch_size": 128,
        "lr": 1e-3,
        "limit_per_class": None  # Full dataset for best accuracy
    },
    "arcane": {
        "epochs": 20,           # Metadata: total epochs = blocks
        "blocks": 20,            # 20 blocks = 20 epochs total
        "batch_size": 128,
        "lr": 1e-3,
        "limit_per_class": None  # Full dataset for best accuracy
    }
}

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Exit code: {e.returncode}")
        return False

def run_sisa():
    """Run SISA training and unlearning on MNIST"""
    config = MNIST_CONFIGS["sisa"]
    out_dir = f"{BASE_DIR}/sisa_mnist_s{config['shards']}x{config['slices']}_e{config['epochs']}"
    
    cmd = [
        "python", "sisa/train_sisa.py",
        "--dataset", "mnist",
        "--epochs", str(config["epochs"]),
        "--shards", str(config["shards"]),
        "--slices", str(config["slices"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--out", out_dir
    ]
    
    if not run_command(cmd, f"Training SISA on MNIST (shards={config['shards']}, slices={config['slices']})"):
        return False
    
    # Run unlearning
    cmd = [
        "python", "sisa/unlearn_sisa.py",
        "--run-dir", out_dir,
        "--unlearn-frac", "0.05"  # 5% unlearn for good speedup demonstration
    ]
    
    if not run_command(cmd, f"SISA unlearning on MNIST"):
        return False
    
    # Generate plots
    cmd = ["python", "common/plot.py", "--run-dir", out_dir]
    run_command(cmd, f"Generating plots for SISA MNIST")
    
    return True

def run_arcane():
    """Run ARCANE training and unlearning on MNIST"""
    config = MNIST_CONFIGS["arcane"]
    out_dir = f"{BASE_DIR}/arcane_mnist_b{config['blocks']}_e{config['epochs']}"
    
    cmd = [
        "python", "arcane/train_arcane.py",
        "--dataset", "mnist",
        "--epochs", str(config["epochs"]),
        "--blocks", str(config["blocks"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--out", out_dir
    ]
    
    if not run_command(cmd, f"Training ARCANE on MNIST (blocks={config['blocks']})"):
        return False
    
    # Run unlearning
    cmd = [
        "python", "arcane/unlearn_arcane.py",
        "--run-dir", out_dir,
        "--unlearn-frac", "0.05"  # 5% unlearn for good speedup demonstration
    ]
    
    if not run_command(cmd, f"ARCANE unlearning on MNIST"):
        return False
    
    # Generate plots
    cmd = ["python", "common/plot.py", "--run-dir", out_dir]
    run_command(cmd, f"Generating plots for ARCANE MNIST")
    
    return True

def print_summary():
    """Print accuracy and speedup summary from all runs"""
    print("\n" + "="*60)
    print("MNIST EXPERIMENT SUMMARY")
    print("="*60)
    
    # SISA results
    sisa_dir = f"{BASE_DIR}/sisa_mnist_s{MNIST_CONFIGS['sisa']['shards']}x{MNIST_CONFIGS['sisa']['slices']}_e{MNIST_CONFIGS['sisa']['epochs']}"
    sisa_summary_path = os.path.join(sisa_dir, "summary.json")
    sisa_unlearn_path = os.path.join(sisa_dir, "metrics", "unlearn_summary.json")
    
    if os.path.exists(sisa_summary_path):
        with open(sisa_summary_path) as f:
            sisa_summary = json.load(f)
        print(f"\nSISA Training:")
        print(f"  Ensemble Accuracy: {sisa_summary.get('ensemble_acc', 'N/A'):.4f}")
        print(f"  Per-shard Accuracy: {[f'{a:.4f}' for a in sisa_summary.get('per_shard_acc', [])]}")
    
    if os.path.exists(sisa_unlearn_path):
        with open(sisa_unlearn_path) as f:
            sisa_unlearn = json.load(f)
        print(f"\nSISA Unlearning:")
        print(f"  Baseline Accuracy:  {sisa_unlearn.get('baseline_acc', 'N/A'):.4f}")
        print(f"  SISA Accuracy:      {sisa_unlearn.get('ensemble_acc_after_unlearn', 'N/A'):.4f}")
        print(f"  Speedup (train):    {sisa_unlearn.get('speedup_train_only', 'N/A'):.2f}x")
        print(f"  Speedup (total):    {sisa_unlearn.get('speedup_total', 'N/A'):.2f}x")
        print(f"  Affected Shards:    {sisa_unlearn.get('num_affected_shards', 'N/A')} / {MNIST_CONFIGS['sisa']['shards']}")
        print(f"  Avg Tail Slices:    {sisa_unlearn.get('avg_tail_slices', 'N/A'):.1f} / {MNIST_CONFIGS['sisa']['slices']}")
    
    # ARCANE results
    arcane_dir = f"{BASE_DIR}/arcane_mnist_b{MNIST_CONFIGS['arcane']['blocks']}_e{MNIST_CONFIGS['arcane']['epochs']}"
    arcane_summary_path = os.path.join(arcane_dir, "summary.json")
    arcane_unlearn_path = os.path.join(arcane_dir, "metrics", "unlearn_summary.json")
    
    if os.path.exists(arcane_summary_path):
        with open(arcane_summary_path) as f:
            arcane_summary = json.load(f)
        print(f"\nARCANE Training:")
        print(f"  Accuracy:           {arcane_summary.get('arcane_acc', 'N/A'):.4f}")
    
    if os.path.exists(arcane_unlearn_path):
        with open(arcane_unlearn_path) as f:
            arcane_unlearn = json.load(f)
        print(f"\nARCANE Unlearning:")
        print(f"  Baseline Accuracy:  {arcane_unlearn.get('baseline_acc', 'N/A'):.4f}")
        print(f"  ARCANE Accuracy:    {arcane_unlearn.get('arcane_acc_after_unlearn', 'N/A'):.4f}")
        print(f"  Speedup:            {arcane_unlearn.get('speedup', 'N/A'):.2f}x")
        print(f"  Tail Blocks:        {arcane_unlearn.get('affected_tail_blocks', 'N/A')} / {MNIST_CONFIGS['arcane']['blocks']}")
    
    print("\n" + "="*60)

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    print("="*60)
    print("MNIST Optimized Experiments")
    print("="*60)
    print("\nConfiguration:")
    print(f"  SISA:  {MNIST_CONFIGS['sisa']['shards']} shards × {MNIST_CONFIGS['sisa']['slices']} slices = {MNIST_CONFIGS['sisa']['slices']} epochs")
    print(f"  ARCANE: {MNIST_CONFIGS['arcane']['blocks']} blocks = {MNIST_CONFIGS['arcane']['blocks']} epochs")
    print(f"  Full dataset (no limit)")
    print()
    
    # Run SISA
    if not run_sisa():
        print("SISA experiment failed!")
        sys.exit(1)
    
    # Run ARCANE
    if not run_arcane():
        print("ARCANE experiment failed!")
        sys.exit(1)
    
    # Print summary
    print_summary()
    
    print("\n✓ All MNIST experiments completed!")

if __name__ == "__main__":
    main()

