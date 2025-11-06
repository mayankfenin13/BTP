#!/usr/bin/env bash
set -e
# FAST profiles targeted for <tight runtime> budgets on CPU/GPU.
# These use small per-class caps and few epochs while keeping the algorithms intact.

# --- SISA on MNIST / FashionMNIST ---
python sisa/train_sisa.py --dataset mnist --epochs 1 --shards 5 --slices 6 --batch-size 128 --limit-per-class 300 --out runs/fast/sisa_mnist_5x6_cap300_e1
python sisa/unlearn_sisa.py --run-dir runs/fast/sisa_mnist_5x6_cap300_e1 --unlearn-frac 0.02
python common/plot.py --run-dir runs/fast/sisa_mnist_5x6_cap300_e1 --title "SISA MNIST (5x6 cap300 e1)"

python sisa/train_sisa.py --dataset fashionmnist --epochs 1 --shards 5 --slices 6 --batch-size 128 --limit-per-class 300 --out runs/fast/sisa_fmnist_5x6_cap300_e1
python sisa/unlearn_sisa.py --run-dir runs/fast/sisa_fmnist_5x6_cap300_e1 --unlearn-frac 0.02
python common/plot.py --run-dir runs/fast/sisa_fmnist_5x6_cap300_e1 --title "SISA FashionMNIST (5x6 cap300 e1)"

# --- ARCANE on MNIST / FashionMNIST ---
python arcane/train_arcane.py --dataset mnist --epochs 1 --blocks 4 --repfrac 0.1 --batch-size 128 --limit-per-class 300 --out runs/fast/arcane_mnist_b4_cap300_e1
python arcane/unlearn_arcane.py --run-dir runs/fast/arcane_mnist_b4_cap300_e1 --unlearn-frac 0.05
python common/plot.py --run-dir runs/fast/arcane_mnist_b4_cap300_e1 --title "ARCANE MNIST (b4 cap300 e1)"

python arcane/train_arcane.py --dataset fashionmnist --epochs 1 --blocks 4 --repfrac 0.1 --batch-size 128 --limit-per-class 300 --out runs/fast/arcane_fmnist_b4_cap300_e1
python arcane/unlearn_arcane.py --run-dir runs/fast/arcane_fmnist_b4_cap300_e1 --unlearn-frac 0.05
python common/plot.py --run-dir runs/fast/arcane_fmnist_b4_cap300_e1 --title "ARCANE FashionMNIST (b4 cap300 e1)"
