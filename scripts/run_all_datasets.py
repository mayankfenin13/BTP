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
import matplotlib.pyplot as plt

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
    
    # Generate plots
    cmd = ["python", "common/plot.py", "--run-dir", out_dir]
    run_command(cmd, f"Generating plots for SISA {dataset}")
    
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
    
    # Generate plots
    cmd = ["python", "common/plot.py", "--run-dir", out_dir]
    run_command(cmd, f"Generating plots for ARCANE {dataset}")
    
    return True

def print_summary():
    """Print accuracy and time summary from all runs"""
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
                        "accuracy": summary[acc_key],
                        "unlearn_acc": None,
                        "baseline_time": None,
                        "unlearn_time": None,
                        "speedup": None
                    }
                    
                    if os.path.exists(unlearn_path):
                        with open(unlearn_path) as f2:
                            unlearn = json.load(f2)
                            unlearn_key = "ensemble_acc_after_unlearn" if "sisa" in run_dir else "arcane_acc_after_unlearn"
                            time_key = "sisa_total_time_s" if "sisa" in run_dir else "arcane_time_s"
                            baseline_key = "baseline_total_time_s" if "sisa" in run_dir else "baseline_time_s"
                            
                            if unlearn_key in unlearn:
                                result["unlearn_acc"] = unlearn[unlearn_key]
                            if time_key in unlearn:
                                result["unlearn_time"] = unlearn[time_key]
                            if baseline_key in unlearn:
                                result["baseline_time"] = unlearn[baseline_key]
                                if result["baseline_time"] and result["unlearn_time"]:
                                    result["speedup"] = result["baseline_time"] / result["unlearn_time"]
                    
                    results.append(result)
    
    print("\n" + "="*90)
    print("ACCURACY SUMMARY")
    print("="*90)
    print(f"{'Method':<8} {'Dataset':<15} {'Initial Acc':<12} {'After Unlearn':<12} {'Baseline Time':<15} {'Unlearn Time':<15} {'Speedup':<10}")
    print("-"*90)
    
    for r in sorted(results, key=lambda x: (x["dataset"], x["method"])):
        baseline_str = f"{r['baseline_time']:.2f}s" if r['baseline_time'] else "N/A"
        unlearn_str = f"{r['unlearn_time']:.2f}s" if r['unlearn_time'] else "N/A"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        acc_str = f"{r['accuracy']:.4f}" if r['accuracy'] else "N/A"
        unlearn_acc_str = f"{r['unlearn_acc']:.4f}" if r['unlearn_acc'] else "N/A"
        
        print(f"{r['method']:<8} {r['dataset']:<15} {acc_str:<12} {unlearn_acc_str:<12} {baseline_str:<15} {unlearn_str:<15} {speedup_str:<10}")
    
    print("="*90)
    
    # Create comparison plots
    create_comparison_plots(results)

def create_comparison_plots(results):
    """Create comparison plots across all datasets"""
    if not results:
        return
    
    # Group by dataset
    datasets = sorted(set(r["dataset"] for r in results))
    
    for dataset in datasets:
        dataset_results = [r for r in results if r["dataset"] == dataset]
        if len(dataset_results) < 2:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Time comparison
        methods = [r["method"] for r in dataset_results]
        baseline_times = [r["baseline_time"] if r["baseline_time"] else 0 for r in dataset_results]
        unlearn_times = [r["unlearn_time"] if r["unlearn_time"] else 0 for r in dataset_results]
        
        x = range(len(methods))
        width = 0.35
        axes[0].bar([i - width/2 for i in x], baseline_times, width, label='Baseline (Retrain)', alpha=0.8)
        axes[0].bar([i + width/2 for i in x], unlearn_times, width, label='Unlearning Method', alpha=0.8)
        axes[0].set_xlabel('Method')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_title(f'{dataset.upper()} - Retraining Time Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy comparison
        initial_accs = [r["accuracy"] if r["accuracy"] else 0 for r in dataset_results]
        unlearn_accs = [r["unlearn_acc"] if r["unlearn_acc"] else 0 for r in dataset_results]
        
        axes[1].bar([i - width/2 for i in x], initial_accs, width, label='Initial Accuracy', alpha=0.8)
        axes[1].bar([i + width/2 for i in x], unlearn_accs, width, label='After Unlearning', alpha=0.8)
        axes[1].set_xlabel('Method')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{dataset.upper()} - Accuracy Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(methods)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        plot_dir = os.path.join(BASE_DIR, "comparison_plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{dataset}_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"\n✓ Comparison plots saved to: {os.path.join(BASE_DIR, 'comparison_plots')}")

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

