#!/usr/bin/env bash
set -e
python sisa/train_sisa.py --dataset mnist --epochs 2 --shards 5 --slices 5 --batch-size 128 --out runs/sisa_mnist_demo
python sisa/unlearn_sisa.py --run-dir runs/sisa_mnist_demo --unlearn-frac 0.01
python common/plot.py --run-dir runs/sisa_mnist_demo --title "SISA MNIST (demo)"

python arcane/train_arcane.py --dataset mnist --epochs 2 --blocks 5 --repfrac 0.1 --batch-size 128 --out runs/arcane_mnist_demo
python arcane/unlearn_arcane.py --run-dir runs/arcane_mnist_demo --unlearn-frac 0.01
python common/plot.py --run-dir runs/arcane_mnist_demo --title "ARCANE MNIST (demo)"
