# MNIST Optimized Configuration

This document describes the optimized configuration for running SISA and Arcane on MNIST with focus on **excellent speedup and accuracy**.

## Configuration Strategy

### SISA Configuration
- **Shards**: 16 (more shards = better speedup)
  - With 16 shards, only ~20% (3-4 shards) are typically affected during unlearning
  - This provides excellent speedup since we only retrain a small fraction of models
  
- **Slices**: 20 (20 epochs total)
  - More epochs = better accuracy
  - Each slice gets 1 epoch of training
  - Total training: 20 epochs per shard
  
- **Dataset**: Full MNIST (60,000 training samples)
  - No limit per class for maximum accuracy
  
- **Hyperparameters**:
  - Batch size: 128
  - Learning rate: 0.001
  - Model: LeNet (standard for MNIST)

### Arcane Configuration
- **Blocks**: 20 (20 epochs total)
  - More blocks = more epochs = better accuracy
  - Each block gets 1 epoch of training
  
- **Dataset**: Full MNIST (60,000 training samples)
  - No limit per class for maximum accuracy
  
- **Hyperparameters**:
  - Batch size: 128
  - Learning rate: 0.001
  - Model: LeNet (standard for MNIST)

## Expected Results

### Speedup
- **SISA**: Expected speedup of **5-10x** or more
  - Only 3-4 out of 16 shards need retraining
  - Only tail slices (last 20% of slices) need retraining
  - Average tail slices: ~4 out of 20 slices per affected shard
  
- **Arcane**: Expected speedup of **3-5x** or more
  - Only tail blocks (last 30% of blocks) need retraining
  - Average tail blocks: ~6 out of 20 blocks

### Accuracy
- **SISA**: Expected accuracy **>95%**
  - Ensemble of 16 models provides robust predictions
  - 20 epochs per shard ensures good convergence
  
- **Arcane**: Expected accuracy **>95%**
  - 20 epochs ensures good convergence
  - Full dataset provides maximum training data

## Running the Experiments

### Option 1: Automated Script (Recommended)
```bash
bash scripts/run_mnist.sh
```

### Option 2: Python Script
```bash
python scripts/run_mnist_optimized.py
```

### Option 3: Manual Commands

**SISA:**
```bash
# Training
python sisa/train_sisa.py \
    --dataset mnist \
    --epochs 20 \
    --shards 16 \
    --slices 20 \
    --batch-size 128 \
    --lr 0.001 \
    --out runs/mnist_optimized/sisa_mnist_s16x20_e20

# Unlearning
python sisa/unlearn_sisa.py \
    --run-dir runs/mnist_optimized/sisa_mnist_s16x20_e20 \
    --unlearn-frac 0.05

# Plots
python common/plot.py --run-dir runs/mnist_optimized/sisa_mnist_s16x20_e20
```

**Arcane:**
```bash
# Training
python arcane/train_arcane.py \
    --dataset mnist \
    --epochs 20 \
    --blocks 20 \
    --batch-size 128 \
    --lr 0.001 \
    --out runs/mnist_optimized/arcane_mnist_b20_e20

# Unlearning
python arcane/unlearn_arcane.py \
    --run-dir runs/mnist_optimized/arcane_mnist_b20_e20 \
    --unlearn-frac 0.05

# Plots
python common/plot.py --run-dir runs/mnist_optimized/arcane_mnist_b20_e20
```

## Output Locations

- **SISA Results**: `runs/mnist_optimized/sisa_mnist_s16x20_e20/`
  - `summary.json`: Training accuracy
  - `metrics/unlearn_summary.json`: Unlearning metrics and speedup
  - `plots/`: Visualization plots

- **Arcane Results**: `runs/mnist_optimized/arcane_mnist_b20_e20/`
  - `summary.json`: Training accuracy
  - `metrics/unlearn_summary.json`: Unlearning metrics and speedup
  - `plots/`: Visualization plots

## Key Metrics to Check

1. **Training Accuracy**: Should be >95% for both methods
2. **Speedup**: 
   - SISA: Check `speedup_train_only` in unlearn_summary.json
   - Arcane: Check `speedup` in unlearn_summary.json
3. **Affected Components**:
   - SISA: `num_affected_shards` should be ~3-4 out of 16
   - Arcane: `affected_tail_blocks` should be ~6 out of 20
4. **Accuracy After Unlearning**: Should remain >94% (minimal drop)

## Why This Configuration Works

1. **More Shards (SISA)**: 
   - Distributes data across more models
   - Fewer models affected during unlearning
   - Better speedup with minimal accuracy loss

2. **More Epochs**: 
   - 20 epochs ensures good convergence
   - Full dataset provides maximum training signal
   - Better final accuracy

3. **Full Dataset**: 
   - No artificial limits
   - Maximum training data for best accuracy
   - Realistic evaluation scenario

4. **Balanced Hyperparameters**:
   - Batch size 128: Good for GPU efficiency
   - Learning rate 0.001: Standard for Adam optimizer
   - LeNet: Appropriate model size for MNIST

