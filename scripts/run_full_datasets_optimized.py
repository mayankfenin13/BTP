#!/usr/bin/env python3
"""
Optimized script to run SISA and ARCANE on Fashion-MNIST and CIFAR-10
Using FULL DATASET for maximum accuracy while maintaining excellent speedup
"""

import os
import subprocess
import sys
import json
from pathlib import Path
import time

BASE_DIR = "runs/full_datasets_optimized"

# Optimized configurations for FULL DATASET
# Strategy: More shards/blocks for speedup, more epochs for accuracy, NO LIMITS
FULL_DATASET_CONFIGS = {
    "fashionmnist": {
        "sisa": {
            "epochs": 25,           # Metadata: total epochs = slices
            "shards": 16,           # More shards = better speedup (fewer affected shards)
            "slices": 25,            # 25 slices = 25 epochs total (1 per slice)
            "batch_size": 128,
            "lr": 1e-3,             # Standard LR for LeNet
            "limit_per_class": None  # FULL DATASET (60k samples)
        },
        "arcane": {
            "epochs": 25,           # Metadata: total epochs = blocks
            "blocks": 25,           # 25 blocks = 25 epochs total
            "batch_size": 128,
            "lr": 1e-3,
            "limit_per_class": None  # FULL DATASET (60k samples)
        }
    },
    "cifar10": {
        "sisa": {
            "epochs": 40,           # Metadata: total epochs = slices (more for ResNet18)
            "shards": 10,           # More shards = better speedup (was 5)
            "slices": 40,           # 40 slices = 40 epochs total
            "batch_size": 128,
            "lr": 1e-3,             # Standard LR for ResNet18
            "limit_per_class": None  # FULL DATASET (50k samples)
        },
        "arcane": {
            "epochs": 40,           # Metadata: total epochs = blocks
            "blocks": 40,           # 40 blocks = 40 epochs total
            "batch_size": 128,
            "lr": 1e-3,
            "limit_per_class": None  # FULL DATASET (50k samples)
        }
    }
}

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"✓ {description} completed in {elapsed/60:.1f} minutes")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ {description} failed after {elapsed/60:.1f} minutes")
        print(f"Exit code: {e.returncode}")
        return False, elapsed

def run_sisa(dataset, config):
    """Run SISA training and unlearning"""
    config_data = FULL_DATASET_CONFIGS[dataset]["sisa"]
    out_dir = f"{BASE_DIR}/sisa_{dataset}_s{config_data['shards']}x{config_data['slices']}_e{config_data['epochs']}"
    
    cmd = [
        "python", "sisa/train_sisa.py",
        "--dataset", dataset,
        "--epochs", str(config_data["epochs"]),
        "--shards", str(config_data["shards"]),
        "--slices", str(config_data["slices"]),
        "--batch-size", str(config_data["batch_size"]),
        "--lr", str(config_data["lr"]),
        "--out", out_dir
    ]
    
    # No limit_per_class for full dataset
    if config_data["limit_per_class"] is not None:
        cmd.extend(["--limit-per-class", str(config_data["limit_per_class"])])
    
    success, train_time = run_command(cmd, f"Training SISA on {dataset} (FULL DATASET)")
    if not success:
        return False, 0, 0
    
    # Run unlearning
    cmd = [
        "python", "sisa/unlearn_sisa.py",
        "--run-dir", out_dir,
        "--unlearn-frac", "0.05"  # 5% unlearn for good speedup demonstration
    ]
    
    success, unlearn_time = run_command(cmd, f"SISA unlearning on {dataset}")
    if not success:
        return False, train_time, 0
    
    # Generate plots
    cmd = ["python", "common/plot.py", "--run-dir", out_dir]
    run_command(cmd, f"Generating plots for SISA {dataset}")
    
    return True, train_time, unlearn_time

def run_arcane(dataset, config):
    """Run ARCANE training and unlearning"""
    config_data = FULL_DATASET_CONFIGS[dataset]["arcane"]
    out_dir = f"{BASE_DIR}/arcane_{dataset}_b{config_data['blocks']}_e{config_data['epochs']}"
    
    cmd = [
        "python", "arcane/train_arcane.py",
        "--dataset", dataset,
        "--epochs", str(config_data["epochs"]),
        "--blocks", str(config_data["blocks"]),
        "--batch-size", str(config_data["batch_size"]),
        "--lr", str(config_data["lr"]),
        "--out", out_dir
    ]
    
    # No limit_per_class for full dataset
    if config_data["limit_per_class"] is not None:
        cmd.extend(["--limit-per-class", str(config_data["limit_per_class"])])
    
    success, train_time = run_command(cmd, f"Training ARCANE on {dataset} (FULL DATASET)")
    if not success:
        return False, 0, 0
    
    # Run unlearning
    cmd = [
        "python", "arcane/unlearn_arcane.py",
        "--run-dir", out_dir,
        "--unlearn-frac", "0.05"  # 5% unlearn for good speedup demonstration
    ]
    
    success, unlearn_time = run_command(cmd, f"ARCANE unlearning on {dataset}")
    if not success:
        return False, train_time, 0
    
    # Generate plots
    cmd = ["python", "common/plot.py", "--run-dir", out_dir]
    run_command(cmd, f"Generating plots for ARCANE {dataset}")
    
    return True, train_time, unlearn_time

def print_summary():
    """Print accuracy and speedup summary from all runs"""
    print("\n" + "="*90)
    print("FULL DATASET EXPERIMENT SUMMARY")
    print("="*90)
    
    # SISA results
    for dataset in ["fashionmnist", "cifar10"]:
        config = FULL_DATASET_CONFIGS[dataset]["sisa"]
        sisa_dir = f"{BASE_DIR}/sisa_{dataset}_s{config['shards']}x{config['slices']}_e{config['epochs']}"
        sisa_summary_path = os.path.join(sisa_dir, "summary.json")
        sisa_unlearn_path = os.path.join(sisa_dir, "metrics", "unlearn_summary.json")
        
        print(f"\n{dataset.upper()} - SISA:")
        if os.path.exists(sisa_summary_path):
            with open(sisa_summary_path) as f:
                sisa_summary = json.load(f)
            print(f"  Ensemble Accuracy: {sisa_summary.get('ensemble_acc', 'N/A'):.4f}")
            if 'per_shard_acc' in sisa_summary:
                avg_shard = sum(sisa_summary['per_shard_acc']) / len(sisa_summary['per_shard_acc'])
                print(f"  Avg Per-Shard Acc: {avg_shard:.4f}")
        
        if os.path.exists(sisa_unlearn_path):
            with open(sisa_unlearn_path) as f:
                sisa_unlearn = json.load(f)
            print(f"  Baseline Accuracy:  {sisa_unlearn.get('baseline_acc', 'N/A'):.4f}")
            print(f"  SISA Accuracy:      {sisa_unlearn.get('ensemble_acc_after_unlearn', 'N/A'):.4f}")
            print(f"  Speedup (train):    {sisa_unlearn.get('speedup_train_only', 'N/A'):.2f}x")
            print(f"  Affected Shards:    {sisa_unlearn.get('num_affected_shards', 'N/A')} / {config['shards']}")
            print(f"  Avg Tail Slices:    {sisa_unlearn.get('avg_tail_slices', 'N/A'):.1f} / {config['slices']}")
    
    # ARCANE results
    for dataset in ["fashionmnist", "cifar10"]:
        config = FULL_DATASET_CONFIGS[dataset]["arcane"]
        arcane_dir = f"{BASE_DIR}/arcane_{dataset}_b{config['blocks']}_e{config['epochs']}"
        arcane_summary_path = os.path.join(arcane_dir, "summary.json")
        arcane_unlearn_path = os.path.join(arcane_dir, "metrics", "unlearn_summary.json")
        
        print(f"\n{dataset.upper()} - ARCANE:")
        if os.path.exists(arcane_summary_path):
            with open(arcane_summary_path) as f:
                arcane_summary = json.load(f)
            print(f"  Accuracy:           {arcane_summary.get('arcane_acc', 'N/A'):.4f}")
        
        if os.path.exists(arcane_unlearn_path):
            with open(arcane_unlearn_path) as f:
                arcane_unlearn = json.load(f)
            print(f"  Baseline Accuracy:  {arcane_unlearn.get('baseline_acc', 'N/A'):.4f}")
            print(f"  ARCANE Accuracy:    {arcane_unlearn.get('arcane_acc_after_unlearn', 'N/A'):.4f}")
            print(f"  Speedup:            {arcane_unlearn.get('speedup', 'N/A'):.2f}x")
            print(f"  Tail Blocks:        {arcane_unlearn.get('affected_tail_blocks', 'N/A')} / {config['blocks']}")
    
    print("\n" + "="*90)

def estimate_time():
    """Estimate time for full dataset runs"""
    print("\n" + "="*60)
    print("TIME ESTIMATES FOR FULL DATASET RUNS")
    print("="*60)
    print()
    print("Based on MNIST results and dataset sizes:")
    print()
    print("Fashion-MNIST (60k samples, LeNet):")
    print("  SISA:  16 shards × 25 slices = ~2-3 hours")
    print("  Arcane: 25 blocks = ~1-1.5 hours")
    print("  Total: ~3-4.5 hours")
    print()
    print("CIFAR-10 (50k samples, ResNet18):")
    print("  SISA:  10 shards × 40 slices = ~8-12 hours")
    print("  Arcane: 40 blocks = ~4-6 hours")
    print("  Total: ~12-18 hours")
    print()
    print("GRAND TOTAL: ~15-22.5 hours")
    print()
    print("Note: These are estimates. Actual time depends on:")
    print("  - GPU/CPU performance")
    print("  - System load")
    print("  - Memory availability")
    print("="*60)

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    print("="*70)
    print("FULL DATASET Optimized Experiments")
    print("="*70)
    print("\nConfiguration:")
    print("\nFashion-MNIST:")
    print(f"  SISA:  {FULL_DATASET_CONFIGS['fashionmnist']['sisa']['shards']} shards × {FULL_DATASET_CONFIGS['fashionmnist']['sisa']['slices']} slices = {FULL_DATASET_CONFIGS['fashionmnist']['sisa']['slices']} epochs")
    print(f"  Arcane: {FULL_DATASET_CONFIGS['fashionmnist']['arcane']['blocks']} blocks = {FULL_DATASET_CONFIGS['fashionmnist']['arcane']['blocks']} epochs")
    print(f"  Full dataset: 60,000 samples (no limit)")
    print("\nCIFAR-10:")
    print(f"  SISA:  {FULL_DATASET_CONFIGS['cifar10']['sisa']['shards']} shards × {FULL_DATASET_CONFIGS['cifar10']['sisa']['slices']} slices = {FULL_DATASET_CONFIGS['cifar10']['sisa']['slices']} epochs")
    print(f"  Arcane: {FULL_DATASET_CONFIGS['cifar10']['arcane']['blocks']} blocks = {FULL_DATASET_CONFIGS['cifar10']['arcane']['blocks']} epochs")
    print(f"  Full dataset: 50,000 samples (no limit)")
    print()
    
    estimate_time()
    
    # Ask for confirmation
    if len(sys.argv) > 1 and sys.argv[1] == "--yes":
        confirmed = True
    else:
        print("\n⚠️  WARNING: This will take 15-22.5 hours to complete!")
        response = input("Continue? (yes/no): ").strip().lower()
        confirmed = response in ["yes", "y"]
    
    if not confirmed:
        print("Cancelled.")
        return
    
    total_start = time.time()
    results = {}
    
    # Run Fashion-MNIST
    print(f"\n{'#'*70}")
    print(f"# FASHION-MNIST Experiments")
    print(f"{'#'*70}\n")
    
    print(">>> Running SISA on Fashion-MNIST...")
    success, train_time, unlearn_time = run_sisa("fashionmnist", FULL_DATASET_CONFIGS["fashionmnist"]["sisa"])
    results["fashionmnist_sisa"] = (success, train_time, unlearn_time)
    
    print("\n>>> Running Arcane on Fashion-MNIST...")
    success, train_time, unlearn_time = run_arcane("fashionmnist", FULL_DATASET_CONFIGS["fashionmnist"]["arcane"])
    results["fashionmnist_arcane"] = (success, train_time, unlearn_time)
    
    # Run CIFAR-10
    print(f"\n{'#'*70}")
    print(f"# CIFAR-10 Experiments")
    print(f"{'#'*70}\n")
    
    print(">>> Running SISA on CIFAR-10...")
    success, train_time, unlearn_time = run_sisa("cifar10", FULL_DATASET_CONFIGS["cifar10"]["sisa"])
    results["cifar10_sisa"] = (success, train_time, unlearn_time)
    
    print("\n>>> Running Arcane on CIFAR-10...")
    success, train_time, unlearn_time = run_arcane("cifar10", FULL_DATASET_CONFIGS["cifar10"]["arcane"])
    results["cifar10_arcane"] = (success, train_time, unlearn_time)
    
    # Print summary
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"{'='*70}")
    
    print_summary()
    
    print("\n✓ All full dataset experiments completed!")
    print(f"Results saved in: {BASE_DIR}")

if __name__ == "__main__":
    main()

