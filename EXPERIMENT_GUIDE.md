# Experiment Guide: Running on Fashion-MNIST, CIFAR-10, and SVHN

This guide explains how to run SISA and ARCANE on the three additional datasets and achieve good accuracy.

## 📊 Dataset Overview

### Fashion-MNIST
- **Type**: 28×28 grayscale clothing images
- **Size**: 60k train / 10k test
- **Model**: LeNet (same as MNIST)
- **Why**: Harder than MNIST, avoids over-optimistic results
- **Expected Accuracy**: ~85-90% (similar to MNIST)

### CIFAR-10
- **Type**: 32×32 RGB natural images
- **Size**: 50k train / 10k test
- **Model**: ResNet18
- **Why**: More realistic, commonly used in literature
- **Expected Accuracy**: ~75-85% (with proper training)
- **Note**: Needs GPU for reasonable training time

### SVHN
- **Type**: 32×32 RGB street-view house numbers
- **Size**: 73k train / 26k test
- **Model**: ResNet18
- **Why**: Commonly used in unlearning literature
- **Expected Accuracy**: ~85-92% (digits are easier than CIFAR-10)

---

## 🚀 Quick Start

### Option 1: Run All Datasets (Automated)

```bash
# Using Python script (recommended)
python scripts/run_all_datasets.py

# Or using bash script
bash scripts/run_all_datasets.sh
```

### Option 2: Run Individual Datasets

#### Fashion-MNIST

**SISA:**
```bash
python sisa/train_sisa.py \
    --dataset fashionmnist \
    --epochs 15 \
    --shards 5 \
    --slices 10 \
    --batch-size 128 \
    --lr 0.001 \
    --out runs/all_datasets/sisa_fashionmnist

python sisa/unlearn_sisa.py \
    --run-dir runs/all_datasets/sisa_fashionmnist \
    --unlearn-frac 0.01
```

**ARCANE:**
```bash
python arcane/train_arcane.py \
    --dataset fashionmnist \
    --epochs 15 \
    --blocks 6 \
    --batch-size 128 \
    --lr 0.001 \
    --out runs/all_datasets/arcane_fashionmnist

python arcane/unlearn_arcane.py \
    --run-dir runs/all_datasets/arcane_fashionmnist \
    --unlearn-frac 0.01
```

#### CIFAR-10

**SISA:**
```bash
python sisa/train_sisa.py \
    --dataset cifar10 \
    --epochs 30 \
    --shards 5 \
    --slices 10 \
    --batch-size 64 \
    --lr 0.001 \
    --limit-per-class 1000 \
    --out runs/all_datasets/sisa_cifar10

python sisa/unlearn_sisa.py \
    --run-dir runs/all_datasets/sisa_cifar10 \
    --unlearn-frac 0.01
```

**ARCANE:**
```bash
python arcane/train_arcane.py \
    --dataset cifar10 \
    --epochs 30 \
    --blocks 6 \
    --batch-size 64 \
    --lr 0.001 \
    --limit-per-class 1000 \
    --out runs/all_datasets/arcane_cifar10

python arcane/unlearn_arcane.py \
    --run-dir runs/all_datasets/arcane_cifar10 \
    --unlearn-frac 0.01
```

#### SVHN

**SISA:**
```bash
python sisa/train_sisa.py \
    --dataset svhn \
    --epochs 25 \
    --shards 5 \
    --slices 10 \
    --batch-size 64 \
    --lr 0.001 \
    --out runs/all_datasets/sisa_svhn

python sisa/unlearn_sisa.py \
    --run-dir runs/all_datasets/sisa_svhn \
    --unlearn-frac 0.01
```

**ARCANE:**
```bash
python arcane/train_arcane.py \
    --dataset svhn \
    --epochs 25 \
    --blocks 6 \
    --batch-size 64 \
    --lr 0.001 \
    --out runs/all_datasets/arcane_svhn

python arcane/unlearn_arcane.py \
    --run-dir runs/all_datasets/arcane_svhn \
    --unlearn-frac 0.01
```

---

## 📈 Hyperparameter Tuning Guide

### Why These Hyperparameters?

#### Fashion-MNIST
- **Epochs: 15** - More epochs needed than MNIST (harder task)
- **Batch Size: 128** - Good balance for 28×28 images
- **LR: 0.001** - Standard Adam learning rate
- **Model: LeNet** - Same architecture as MNIST (28×28 grayscale)

#### CIFAR-10
- **Epochs: 30** - More complex images, need more training
- **Batch Size: 64** - Smaller for ResNet18 (32×32 RGB needs more memory)
- **LR: 0.001** - Can experiment with 0.0005 for better convergence
- **Model: ResNet18** - Standard for CIFAR-10
- **Limit: 1000** - For faster runs; remove for full dataset

#### SVHN
- **Epochs: 25** - Similar complexity to CIFAR-10
- **Batch Size: 64** - Same as CIFAR-10
- **LR: 0.001** - Standard
- **Model: ResNet18** - Standard for 32×32 RGB images

---

## 🎯 Achieving Good Accuracy

### Tips for Better Accuracy:

1. **More Epochs**: Increase epochs if accuracy is low
   ```bash
   --epochs 50  # For CIFAR-10/SVHN
   ```

2. **Lower Learning Rate**: For better convergence
   ```bash
   --lr 0.0005  # For CIFAR-10/SVHN
   ```

3. **More Shards (SISA)**: More ensemble models
   ```bash
   --shards 10  # Instead of 5
   ```

4. **More Blocks (ARCANE)**: Finer-grained checkpoints
   ```bash
   --blocks 10  # Instead of 6
   ```

5. **Full Dataset**: Remove `--limit-per-class` for best results
   ```bash
   # Remove --limit-per-class 1000
   ```

6. **GPU Acceleration**: Use GPU for CIFAR-10/SVHN
   ```bash
   # Ensure CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## 📊 Expected Results

### Fashion-MNIST
- **SISA**: ~85-90% (ensemble of 5 models)
- **ARCANE**: ~85-88% (single model)
- **After Unlearning**: ~84-89% (minimal drop)

### CIFAR-10
- **SISA**: ~75-85% (ensemble of 5 models)
- **ARCANE**: ~70-80% (single model)
- **After Unlearning**: ~74-84% (minimal drop)

### SVHN
- **SISA**: ~90-95% (ensemble of 5 models)
- **ARCANE**: ~85-92% (single model)
- **After Unlearning**: ~89-94% (minimal drop)

---

## 🔍 Checking Results

After running experiments, check results:

```bash
# View summary
cat runs/all_datasets/sisa_fashionmnist_*/summary.json
cat runs/all_datasets/sisa_fashionmnist_*/metrics/unlearn_summary.json

# Or use Python
python3 << 'EOF'
import json
import os

run_dir = "runs/all_datasets/sisa_fashionmnist_s5x10_e15"
with open(f"{run_dir}/summary.json") as f:
    print("Initial Accuracy:", json.load(f))
with open(f"{run_dir}/metrics/unlearn_summary.json") as f:
    print("After Unlearning:", json.load(f))
EOF
```

---

## ⚠️ Common Issues

### 1. Out of Memory (CIFAR-10/SVHN)
- **Solution**: Reduce batch size
  ```bash
  --batch-size 32  # Instead of 64
  ```

### 2. Slow Training (CPU)
- **Solution**: Use GPU or reduce dataset size
  ```bash
  --limit-per-class 500  # For faster runs
  ```

### 3. Low Accuracy
- **Solution**: 
  - Increase epochs
  - Lower learning rate
  - Remove dataset limits
  - Use more shards/blocks

### 4. Dataset Download Issues
- **Solution**: First run downloads datasets
  ```bash
  # Test download manually
  python -c "from common.data import get_dataset; get_dataset('cifar10', train=True)"
  ```

---

## 📝 Comparison with MNIST

| Dataset | MNIST | Fashion-MNIST | CIFAR-10 | SVHN |
|---------|-------|---------------|----------|------|
| **Size** | 28×28 G | 28×28 G | 32×32 RGB | 32×32 RGB |
| **Model** | LeNet | LeNet | ResNet18 | ResNet18 |
| **Epochs** | 5-8 | 15 | 30 | 25 |
| **SISA Acc** | ~87% | ~87% | ~80% | ~92% |
| **Complexity** | Low | Medium | High | Medium |

---

## 🎓 Next Steps

1. **Compare Results**: Create plots comparing all datasets
2. **Tune Hyperparameters**: Experiment with different settings
3. **Full Dataset Runs**: Remove limits for publication-ready results
4. **Multiple Runs**: Average over multiple seeds for statistical significance

Good luck with your experiments! 🚀

