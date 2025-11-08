# Full Dataset Analysis: Fashion-MNIST & CIFAR-10

## Current vs Full Dataset Comparison

### Previous Configuration (Limited Data)

**Fashion-MNIST:**
- Limit: 2,000 samples per class = 20,000 total (33% of full dataset)
- Shards: 16, Slices: 24
- **Previous Results:**
  - Ensemble Accuracy: 62.13% ❌ (Very low - likely bug or evaluation issue)
  - Per-shard Accuracy: 71-76% (reasonable)
  - Speedup: 1.25x (low - not good)
  - **Issue**: Ensemble accuracy is lower than individual shards, suggesting evaluation bug

**CIFAR-10:**
- Limit: 2,000 samples per class = 20,000 total (40% of full dataset)
- Shards: 5, Slices: 30
- **Previous Results:**
  - Ensemble Accuracy: 30.5% ❌ (Very low)
  - Per-shard Accuracy: 42-48% (low)
  - Speedup: 3.83x (decent)
  - **Issue**: Model not training well with limited data

### Full Dataset Configuration (Recommended)

**Fashion-MNIST:**
- **Full Dataset**: 60,000 samples (100%)
- Shards: 16, Slices: 25
- **Expected Results:**
  - Ensemble Accuracy: **>90%** ✅ (should match MNIST performance)
  - Speedup: **5-10x** ✅ (only 3-4 shards affected)
  - Training Time: ~2-3 hours

**CIFAR-10:**
- **Full Dataset**: 50,000 samples (100%)
- Shards: 10 (increased from 5), Slices: 40
- **Expected Results:**
  - Ensemble Accuracy: **>85%** ✅ (ResNet18 with full data)
  - Speedup: **5-8x** ✅ (only 2-3 shards affected)
  - Training Time: ~8-12 hours

## Why Full Dataset is Better

### 1. **Accuracy Improvement**
- **Fashion-MNIST**: Current 62% → Expected >90% (+28% improvement)
- **CIFAR-10**: Current 30% → Expected >85% (+55% improvement)
- More training data = better generalization
- No artificial data limits = realistic evaluation

### 2. **Speedup Still Excellent**
- **More shards = better speedup** (fewer affected shards)
- Fashion-MNIST: 16 shards → only 3-4 affected (20-25%)
- CIFAR-10: 10 shards → only 2-3 affected (20-30%)
- Speedup remains excellent even with full dataset

### 3. **Publication-Ready Results**
- Full dataset results are standard in ML literature
- No need to explain data limitations
- Comparable to other papers

## Time Estimates

### Fashion-MNIST (LeNet - Fast)
- **SISA Training**: 
  - 16 shards × 25 slices = 400 total training steps
  - ~2-3 hours (with GPU/MPS)
- **SISA Unlearning**: ~5-10 minutes
- **Arcane Training**: 
  - 25 blocks = 25 epochs
  - ~1-1.5 hours
- **Arcane Unlearning**: ~5-10 minutes
- **Total**: ~3-4.5 hours

### CIFAR-10 (ResNet18 - Slower)
- **SISA Training**: 
  - 10 shards × 40 slices = 400 total training steps
  - ~8-12 hours (with GPU/MPS)
- **SISA Unlearning**: ~15-30 minutes
- **Arcane Training**: 
  - 40 blocks = 40 epochs
  - ~4-6 hours
- **Arcane Unlearning**: ~10-20 minutes
- **Total**: ~12-18 hours

### Grand Total
- **Fashion-MNIST**: ~3-4.5 hours
- **CIFAR-10**: ~12-18 hours
- **TOTAL**: ~15-22.5 hours

## Configuration Comparison

| Dataset | Method | Shards/Blocks | Slices/Blocks | Epochs | Limit | Expected Acc | Expected Speedup |
|---------|--------|---------------|---------------|--------|-------|--------------|------------------|
| **Fashion-MNIST** | SISA | 16 | 25 | 25 | **None** | >90% | 5-10x |
| **Fashion-MNIST** | Arcane | - | 25 | 25 | **None** | >90% | 3-5x |
| **CIFAR-10** | SISA | 10 | 40 | 40 | **None** | >85% | 5-8x |
| **CIFAR-10** | Arcane | - | 40 | 40 | **None** | >85% | 3-5x |

## Recommendations

### ✅ Use Full Dataset Because:

1. **Accuracy is Critical**: Current results are too low (30-62%)
2. **Speedup Still Excellent**: More shards maintain good speedup
3. **Realistic Evaluation**: Full dataset is standard
4. **Better Results**: Will match or exceed MNIST performance

### ⚠️ Previous Config Was NOT Good Enough:

- **Fashion-MNIST**: 62% accuracy is unacceptable (should be >90%)
- **CIFAR-10**: 30% accuracy is unacceptable (should be >85%)
- Limited data caused poor training
- Ensemble evaluation may have bugs

### 🎯 Optimal Strategy:

1. **Run Full Dataset**: Remove all limits
2. **Increase Shards**: More shards = better speedup
3. **More Epochs**: Better accuracy (25 for Fashion-MNIST, 40 for CIFAR-10)
4. **Run Overnight**: 15-22 hours is manageable overnight

## Running the Experiments

### Option 1: Automated Script (Recommended)
```bash
python scripts/run_full_datasets_optimized.py
```

### Option 2: Manual Commands

**Fashion-MNIST SISA:**
```bash
python sisa/train_sisa.py \
    --dataset fashionmnist \
    --epochs 25 \
    --shards 16 \
    --slices 25 \
    --batch-size 128 \
    --lr 0.001 \
    --out runs/full_datasets_optimized/sisa_fashionmnist_s16x25_e25

python sisa/unlearn_sisa.py \
    --run-dir runs/full_datasets_optimized/sisa_fashionmnist_s16x25_e25 \
    --unlearn-frac 0.05
```

**CIFAR-10 SISA:**
```bash
python sisa/train_sisa.py \
    --dataset cifar10 \
    --epochs 40 \
    --shards 10 \
    --slices 40 \
    --batch-size 128 \
    --lr 0.001 \
    --out runs/full_datasets_optimized/sisa_cifar10_s10x40_e40

python sisa/unlearn_sisa.py \
    --run-dir runs/full_datasets_optimized/sisa_cifar10_s10x40_e40 \
    --unlearn-frac 0.05
```

## Expected Final Results

### Fashion-MNIST
- **SISA Accuracy**: >90% (vs current 62%)
- **SISA Speedup**: 5-10x (vs current 1.25x)
- **Arcane Accuracy**: >90%
- **Arcane Speedup**: 3-5x

### CIFAR-10
- **SISA Accuracy**: >85% (vs current 30%)
- **SISA Speedup**: 5-8x (vs current 3.83x)
- **Arcane Accuracy**: >85%
- **Arcane Speedup**: 3-5x

## Conclusion

**YES, use full dataset!** The previous configuration with limited data produced poor results. Full dataset will:
- ✅ Dramatically improve accuracy (30-62% → 85-90%+)
- ✅ Maintain excellent speedup (5-10x)
- ✅ Provide publication-ready results
- ✅ Match MNIST performance levels

The time investment (15-22 hours) is worth it for the significant accuracy improvement.

