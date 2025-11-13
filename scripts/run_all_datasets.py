#!/usr/bin/env python3
"""
Comprehensive script to run SISA and ARCANE on Fashion-MNIST and CIFAR-10
Optimized hyperparameters for good accuracy on each dataset
"""

import os
import subprocess
import sys
import json
import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend (no display required)
import matplotlib.pyplot as plt

# Resolve repository root (parent of this scripts/ directory) and use absolute paths
REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = str(REPO_ROOT / "runs" / "all_datasets")

# Experiment configurations optimized for good accuracy and speedup
# Note: epochs parameter is now metadata (actual training uses 1 epoch per slice/block)
# Total training epochs = slices (SISA) or blocks (ARCANE)
EXPERIMENTS = {
    "fashionmnist": {
        "sisa":   {
            "epochs": 24,           # Metadata: total epochs = slices
            "shards": 16,           # More shards = better speedup (fewer affected shards)
            "slices": 24,           # 24 slices = 24 epochs total (1 per slice)
            "batch_size": 128,
            "lr": 5e-3,
            "limit_per_class": 2000  # More data for better accuracy
        },
        "arcane": {
            "epochs": 12,           # Metadata: total epochs = blocks (previous setting)
            "blocks": 12,           # 12 blocks = 12 epochs total
            "batch_size": 128,
            "lr": 1e-3,
            "limit_per_class": 2000
        }
    },
    "cifar10": {
        "sisa":   {
            "epochs": 30,           # Metadata: total epochs = slices (increased for better accuracy)
            "shards": 5,            # More shards = better speedup
            "slices": 30,           # 30 slices = 30 epochs total - more epochs for better accuracy
            "batch_size": 128,      # Increased for GPU (was 64)
            "lr": 1e-3,             # Lower LR for better convergence with more epochs
            "limit_per_class": 2000  # Increased from 400 to 2000 for better accuracy (vs full 5000)
        },
        "arcane": {
            "epochs": 30,           # Metadata: total epochs = blocks (increased for better accuracy)
            "blocks": 30,           # 30 blocks = 30 epochs total - more epochs for better accuracy
            "batch_size": 128,      # Increased for GPU (was 64)
            "lr": 1e-3,             # Lower LR for better convergence with more epochs
            "limit_per_class": 2000  # Increased from 400 to 2000 for better accuracy (vs full 5000)
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
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, cwd=str(REPO_ROOT))
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed!")
        print(f"Exit code: {e.returncode}")
        return False

def run_sisa(dataset, config, affect_frac_shards=0.2):
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
        "--unlearn-frac", "0.05",
        "--affect-frac-shards", str(affect_frac_shards)
    ]
    
    if not run_command(cmd, f"SISA unlearning on {dataset} (affecting {affect_frac_shards*100:.0f}% of shards)"):
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
        "--unlearn-frac", "0.05"
    ]
    
    if not run_command(cmd, f"ARCANE unlearning on {dataset}"):
        return False
    
    # Generate plots
    cmd = ["python", "common/plot.py", "--run-dir", out_dir]
    run_command(cmd, f"Generating plots for ARCANE {dataset}")
    
    return True

def print_summary():
    """Print accuracy and time summary from all runs"""
    results_dict = {}  # (method, dataset) -> (result, mtime)
    
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
                
                # Include both SISA and ARCANE for Fashion-MNIST and CIFAR-10
                
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
                            
                            # Use speedup from JSON (speedup_train_only for SISA, speedup for ARCANE)
                            if "sisa" in run_dir:
                                if "speedup_train_only" in unlearn:
                                    result["speedup"] = unlearn["speedup_train_only"]
                            else:
                                if "speedup" in unlearn:
                                    result["speedup"] = unlearn["speedup"]
                    
                    # Get modification time to keep only most recent run
                    mtime = os.path.getmtime(run_dir)
                    key = (method, dataset)
                    
                    # Keep only the most recent run for each method-dataset combination
                    if key not in results_dict or mtime > results_dict[key][1]:
                        results_dict[key] = (result, mtime)
    
    # Convert to list of results only
    results = [r[0] for r in results_dict.values()]
    
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
    
    # Filter out SVHN (not in current experiments)
    results = [r for r in results if r["dataset"] != "svhn"]
    
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
    print("Running SISA CIFAR-10 with Moderate Speedup (10-20x)")
    print("="*70)
    print()
    print("This will run SISA CIFAR-10 with 60% shard affectation.")
    print("Target speedup: 10-20x (instead of 1290x)")
    print("Other runs are commented out for this execution.")
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
    
    # Run experiments - ONLY SISA CIFAR-10 with moderate speedup (10-20x)
    # Commented out other runs for this execution
    datasets = ["cifar10"]  # Only CIFAR-10
    # datasets = ["fashionmnist", "cifar10"]  # Original: both datasets
    
    for dataset in datasets:
        print(f"\n{'#'*70}")
        print(f"# {dataset.upper()} Experiments")
        print(f"{'#'*70}\n")
        
        config = EXPERIMENTS[dataset]
        
        # Run SISA
        print(f"\n>>> Running SISA on {dataset}...")
        # For CIFAR-10, use affect_frac_shards=0.6 (60% = 3 shards) to get ~10-20x speedup
        affect_frac = 0.6 if dataset == "cifar10" else 0.2
        if not run_sisa(dataset, config["sisa"], affect_frac_shards=affect_frac):
            print(f"WARNING: SISA on {dataset} failed.")
        
        # Run ARCANE (if configured) - COMMENTED OUT FOR THIS RUN
        # if "arcane" in config:
        #     print(f"\n>>> Running ARCANE on {dataset}...")
        #     if not run_arcane(dataset, config["arcane"]):
        #         print(f"WARNING: ARCANE on {dataset} failed.")
    
    # Print summary
    print_summary()
    
    print("\n✓ All experiments complete!")
    print(f"Results saved in: {BASE_DIR}")

if __name__ == "__main__":
    main()

