#!/usr/bin/env python3
"""
Comprehensive script to run SISA and ARCANE on Fashion-MNIST, CIFAR-10, and SVHN
Optimized hyperparameters for good accuracy on each dataset
"""

import os
import subprocess
import sys
import json
import glob
from pathlib import Path

BASE_DIR = "runs/all_datasets"

# Experiment configurations optimized for each dataset
EXPERIMENTS = {
    "fashionmnist": {
        "sisa":   {"epochs": 1, "shards": 2, "slices": 2, "batch_size": 128, "lr": 1e-3, "limit_per_class": 300},
        "arcane": {"epochs": 1, "blocks": 2, "batch_size": 128, "lr": 1e-3, "limit_per_class": 300}
    },
    "cifar10": {
        "sisa":   {"epochs": 1, "shards": 2, "slices": 2, "batch_size": 64,  "lr": 1e-3, "limit_per_class": 200},
        "arcane": {"epochs": 1, "blocks": 2, "batch_size": 64,  "lr": 1e-3, "limit_per_class": 200}
    },
    "svhn": {
        "sisa":   {"epochs": 1, "shards": 2, "slices": 2, "batch_size": 64,  "lr": 1e-3, "limit_per_class": 200},
        "arcane": {"epochs": 1, "blocks": 2, "batch_size": 64,  "lr": 1e-3, "limit_per_class": 200}
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
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed!")
        print(f"Exit code: {e.returncode}")
        return False

def run_sisa(dataset, config):
    """Run SISA training and unlearning"""
    out_dir = f"{BASE_DIR}/sisa_{dataset}_s{config['shards']}x{config['slices']}_e{config['epochs']}"
    
    cmd = [
        "python", "sisa/train_sisa.py",
        "--dataset", dataset,
        "--epochs", str(config["epochs"]),
        "--shards", str(config["shards"]),
        "--slices", str(config["slices"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--out", out_dir
    ]
    
    if config["limit_per_class"] is not None:
        cmd.extend(["--limit-per-class", str(config["limit_per_class"])])
    
    if not run_command(cmd, f"Training SISA on {dataset}"):
        return False
    
    # Run unlearning
    cmd = [
        "python", "sisa/unlearn_sisa.py",
        "--run-dir", out_dir,
        "--unlearn-frac", "0.01"
    ]
    
    if not run_command(cmd, f"SISA unlearning on {dataset}"):
        return False
    
    return True

def run_arcane(dataset, config):
    """Run ARCANE training and unlearning"""
    out_dir = f"{BASE_DIR}/arcane_{dataset}_b{config['blocks']}_e{config['epochs']}"
    
    cmd = [
        "python", "arcane/train_arcane.py",
        "--dataset", dataset,
        "--epochs", str(config["epochs"]),
        "--blocks", str(config["blocks"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--out", out_dir
    ]
    
    if config["limit_per_class"] is not None:
        cmd.extend(["--limit-per-class", str(config["limit_per_class"])])
    
    if not run_command(cmd, f"Training ARCANE on {dataset}"):
        return False
    
    # Run unlearning
    cmd = [
        "python", "arcane/unlearn_arcane.py",
        "--run-dir", out_dir,
        "--unlearn-frac", "0.01"
    ]
    
    if not run_command(cmd, f"ARCANE unlearning on {dataset}"):
        return False
    
    return True

def print_summary():
    """Print accuracy summary from all runs"""
    results = []
    
    for run_dir in glob.glob(f"{BASE_DIR}/*"):
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
                    result = {
                        "method": method,
                        "dataset": dataset,
                        "accuracy": f"{summary[acc_key]:.4f}",
                        "unlearn_acc": "N/A"
                    }
                    
                    if os.path.exists(unlearn_path):
                        with open(unlearn_path) as f2:
                            unlearn = json.load(f2)
                            unlearn_key = "ensemble_acc_after_unlearn" if "sisa" in run_dir else "arcane_acc_after_unlearn"
                            if unlearn_key in unlearn:
                                result["unlearn_acc"] = f"{unlearn[unlearn_key]:.4f}"
                    
                    results.append(result)
    
    print("\n" + "="*70)
    print("ACCURACY SUMMARY")
    print("="*70)
    print(f"{'Method':<8} {'Dataset':<15} {'Initial Acc':<15} {'After Unlearn':<15}")
    print("-"*70)
    
    for r in sorted(results, key=lambda x: (x["dataset"], x["method"])):
        print(f"{r['method']:<8} {r['dataset']:<15} {r['accuracy']:<15} {r['unlearn_acc']:<15}")
    
    print("="*70)

def main():
    """Main execution"""
    os.makedirs(BASE_DIR, exist_ok=True)
    
    print("="*70)
    print("Running SISA and ARCANE on Fashion-MNIST, CIFAR-10, and SVHN")
    print("="*70)
    print()
    print("This will run experiments with optimized hyperparameters.")
    print("Note: CIFAR-10 and SVHN may take longer (GPU recommended).")
    print()
    
    # Ask for confirmation
    if len(sys.argv) > 1 and sys.argv[1] == "--yes":
        confirmed = True
    else:
        response = input("Continue? (yes/no): ").strip().lower()
        confirmed = response in ["yes", "y"]
    
    if not confirmed:
        print("Cancelled.")
        return
    
    # Run experiments
    datasets = ["fashionmnist", "cifar10", "svhn"]
    
    for dataset in datasets:
        print(f"\n{'#'*70}")
        print(f"# {dataset.upper()} Experiments")
        print(f"{'#'*70}\n")
        
        config = EXPERIMENTS[dataset]
        
        # Run SISA
        print(f"\n>>> Running SISA on {dataset}...")
        if not run_sisa(dataset, config["sisa"]):
            print(f"WARNING: SISA on {dataset} failed, continuing...")
        
        # Run ARCANE
        print(f"\n>>> Running ARCANE on {dataset}...")
        if not run_arcane(dataset, config["arcane"]):
            print(f"WARNING: ARCANE on {dataset} failed, continuing...")
    
    # Print summary
    print_summary()
    
    print("\n✓ All experiments complete!")
    print(f"Results saved in: {BASE_DIR}")

if __name__ == "__main__":
    main()

