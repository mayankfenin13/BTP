# Complete Dataset Usage and Settings Summary

This document provides a comprehensive overview of all datasets used, whether full or partial data was used, and all settings (epochs, slices, blocks, etc.) for both SISA and ARCANE algorithms.

---

## 📊 Dataset Usage Summary

### MNIST
- **Full Dataset**: ✅ **YES** - Used entire dataset (60,000 training samples)
- **Limit per class**: `None` (no capping)
- **Total samples**: ~60,000 (full dataset)

### Fashion-MNIST
- **Full Dataset**: ❌ **NO** - Used subset of data
- **Limit per class**: `2000` samples per class
- **Total samples**: ~20,000 (2,000 × 10 classes)

### CIFAR-10
- **Full Dataset**: ❌ **NO** - Used subset of data
- **Limit per class**: `2000` samples per class
- **Total samples**: ~20,000 (2,000 × 10 classes)

---

## 🔬 SISA Algorithm Settings

### MNIST (Optimized Run)
**Location**: `runs/mnist_optimized/sisa_mnist_s16x20_e20/`

| Setting | Value |
|---------|-------|
| **Dataset** | MNIST |
| **Data Usage** | Full dataset (60,000 samples) |
| **Limit per class** | None |
| **Shards** | 16 |
| **Slices per shard** | 20 |
| **Epochs per slice** | 1 |
| **Total epochs per shard** | 20 (1 epoch × 20 slices) |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Seed** | 42 |
| **Model** | LeNet |

**Results**:
- Initial ensemble accuracy: 86.43%
- Unlearn speedup: **1513x**
- Baseline time: 243.3s
- Unlearn time: 0.16s

---

### Fashion-MNIST
**Location**: `runs/all_datasets/sisa_fashionmnist_s16x24_e24/`

| Setting | Value |
|---------|-------|
| **Dataset** | Fashion-MNIST |
| **Data Usage** | Subset (20,000 samples) |
| **Limit per class** | 2,000 |
| **Shards** | 16 |
| **Slices per shard** | 24 |
| **Epochs per slice** | 1 |
| **Total epochs per shard** | 24 (1 epoch × 24 slices) |
| **Batch size** | 128 |
| **Learning rate** | 0.005 |
| **Seed** | 42 |
| **Model** | LeNet |

**Results**:
- Initial ensemble accuracy: 62.13%
- Unlearn speedup: 1.27x
- Baseline time: 89.3s
- Unlearn time: 70.1s
- Affected shards: 4 out of 16
- Average tail slices: 5.0 out of 24

---

### CIFAR-10
**Location**: `runs/all_datasets/sisa_cifar10_s5x30_e30/`

| Setting | Value |
|---------|-------|
| **Dataset** | CIFAR-10 |
| **Data Usage** | Subset (20,000 samples) |
| **Limit per class** | 2,000 |
| **Shards** | 5 |
| **Slices per shard** | 30 |
| **Epochs per slice** | 1 |
| **Total epochs per shard** | 30 (1 epoch × 30 slices) |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Seed** | 42 |
| **Model** | ResNet18 (small_cnn_cifar) |

**Results**:
- Initial ensemble accuracy: 30.5%
- Unlearn speedup: 3.87x
- Baseline time: 3693.7s
- Unlearn time: 954.7s
- Affected shards: 2 out of 5
- Average tail slices: 9.0 out of 30

---

## 🎯 ARCANE Algorithm Settings

### MNIST (Optimized Run)
**Location**: `runs/mnist_optimized/arcane_mnist_b20_e20/`

| Setting | Value |
|---------|-------|
| **Dataset** | MNIST |
| **Data Usage** | Full dataset (60,000 samples) |
| **Limit per class** | None |
| **Blocks** | 20 |
| **Epochs per block** | 1 |
| **Total epochs** | 20 (1 epoch × 20 blocks) |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Seed** | 42 |
| **Model** | LeNet |

**Results**:
- Initial accuracy: 98.78%
- Unlearn speedup: 12.55x
- Baseline time: 199.6s
- Unlearn time: 15.9s
- Affected tail blocks: 6 out of 20
- Earliest affected block: 14

---

### Fashion-MNIST
**Location**: `runs/all_datasets/arcane_fashionmnist_b12_e12/`

| Setting | Value |
|---------|-------|
| **Dataset** | Fashion-MNIST |
| **Data Usage** | Subset (20,000 samples) |
| **Limit per class** | 2,000 |
| **Blocks** | 12 |
| **Epochs per block** | 1 |
| **Total epochs** | 12 (1 epoch × 12 blocks) |
| **Batch size** | 128 |
| **Learning rate** | 0.01 |
| **Seed** | 42 |
| **Model** | LeNet |

**Results**:
- Initial accuracy: 81.18%
- Unlearn speedup: 3.61x
- Baseline time: 25.5s
- Unlearn time: 7.1s
- Affected tail blocks: 4 out of 12
- Earliest affected block: 8

---

### CIFAR-10
**Location**: `runs/all_datasets/arcane_cifar10_b30_e30/`

| Setting | Value |
|---------|-------|
| **Dataset** | CIFAR-10 |
| **Data Usage** | Subset (20,000 samples) |
| **Limit per class** | 2,000 |
| **Blocks** | 30 |
| **Epochs per block** | 1 |
| **Total epochs** | 30 (1 epoch × 30 blocks) |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Seed** | 42 |
| **Model** | ResNet18 (small_cnn_cifar) |

**Results**:
- Initial accuracy: 54.51%
- Unlearn speedup: 13.91x
- Baseline time: 3604.7s
- Unlearn time: 259.2s
- Affected tail blocks: 9 out of 30
- Earliest affected block: 21

---

## 📋 Quick Reference Table

| Dataset | Algorithm | Full Dataset? | Limit/Class | Shards/Blocks | Slices/Blocks | Epochs | Batch Size | LR | Speedup |
|---------|-----------|---------------|-------------|---------------|---------------|--------|------------|-----|---------|
| **MNIST** | SISA | ✅ Yes | None | 16 shards | 20 slices | 20 | 128 | 0.001 | **1513x** |
| **MNIST** | ARCANE | ✅ Yes | None | 20 blocks | 20 blocks | 20 | 128 | 0.001 | 12.55x |
| **Fashion-MNIST** | SISA | ❌ No | 2000 | 16 shards | 24 slices | 24 | 128 | 0.005 | 1.27x |
| **Fashion-MNIST** | ARCANE | ❌ No | 2000 | 12 blocks | 12 blocks | 12 | 128 | 0.01 | 3.61x |
| **CIFAR-10** | SISA | ❌ No | 2000 | 5 shards | 30 slices | 30 | 128 | 0.001 | 3.87x |
| **CIFAR-10** | ARCANE | ❌ No | 2000 | 30 blocks | 30 blocks | 30 | 128 | 0.001 | 13.91x |

---

## 🔍 Key Observations

### Data Usage Patterns

1. **MNIST**: Used **full dataset** for both algorithms
   - Best accuracy results
   - Best speedup for SISA (1513x)
   - Full 60,000 samples provide best performance

2. **Fashion-MNIST & CIFAR-10**: Used **subset** (2,000 per class = 20,000 total)
   - Faster training times
   - Lower accuracy (especially CIFAR-10 SISA at 30.5%)
   - Still demonstrates unlearning effectiveness

### Training Strategy

**All experiments use incremental training**:
- **SISA**: 1 epoch per slice (slices are added incrementally)
- **ARCANE**: 1 epoch per block (blocks are added incrementally)
- This allows checkpointing at each step for efficient unlearning

### Why Subsets Were Used

For Fashion-MNIST and CIFAR-10, subsets were used likely for:
- **Faster experimentation**: Full datasets take longer to train
- **Resource constraints**: Less disk space for checkpoints
- **Demonstration purposes**: Still shows unlearning effectiveness

### Retraining Strategy

**During unlearning, both algorithms retrain using**:
- **SISA**: Only affected shards, only from the earliest affected slice onwards
- **ARCANE**: Only from the earliest affected block onwards
- **Both exclude unlearn data** from retraining

**This is NOT full retraining** - it's incremental retraining from checkpoints, which is why speedups are achieved!

---

## 📝 Notes

1. **Epochs parameter**: The `epochs` parameter in configs is metadata. Actual training uses **1 epoch per slice/block**.

2. **Total training epochs**:
   - SISA: `slices` epochs per shard (each shard trained independently)
   - ARCANE: `blocks` epochs total (single model trained incrementally)

3. **Data splitting**:
   - SISA: Data → Shards → Slices (nested splitting)
   - ARCANE: Data → Blocks (single-level splitting)

4. **Unlearning fraction**: All experiments used 5% unlearning fraction (`--unlearn-frac 0.05`)

---

## 🎯 Summary

- **MNIST**: Full dataset for both algorithms ✅
- **Fashion-MNIST**: Subset (2,000/class) for both algorithms ❌
- **CIFAR-10**: Subset (2,000/class) for both algorithms ❌

**Best results**: MNIST with full dataset achieved 1513x speedup with SISA!

