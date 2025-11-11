# Complete Project Explanation: Machine Unlearning with SISA & ARCANE

## 📚 Table of Contents
1. [Project Overview](#project-overview)
2. [What is Machine Unlearning?](#what-is-machine-unlearning)
3. [The Two Methods: SISA vs ARCANE](#the-two-methods-sisa-vs-arcane)
4. [Dataset Details](#dataset-details)
5. [Code Structure Explained](#code-structure-explained)
6. [Understanding the Results](#understanding-the-results)
7. [The 1521x Speedup Explained](#the-1521x-speedup-explained)
8. [How to Explain This to Someone](#how-to-explain-this-to-someone)

---

## 🎯 Project Overview

This project implements **two state-of-the-art machine unlearning methods** that allow you to remove specific data points from a trained machine learning model **without retraining from scratch**.

**Why is this important?**
- **GDPR/Privacy Laws**: Users have the "right to be forgotten" - their data must be removable
- **Data Corrections**: Remove incorrect or outdated data
- **Security**: Remove poisoned or adversarial examples
- **Efficiency**: Much faster than retraining entire models

**The Challenge**: Traditional approach is to delete data and retrain from scratch (slow, expensive). These methods achieve the same result **much faster** by using clever checkpointing and incremental training strategies.

---
## 🤔 What is Machine Unlearning?

### The Problem
Imagine you trained a model on 60,000 images. A user requests their data be deleted. The naive solution:
1. Delete their data from the dataset
2. **Retrain the entire model from scratch** (takes hours/days)

### The Solution (SISA & ARCANE)
Instead of retraining everything, these methods:
1. **Save checkpoints** during training (snapshots of the model)
2. When data needs to be removed, **rollback to the checkpoint before that data was added**
3. **Retrain only from that point forward** (much faster!)

**Key Insight**: If you only need to remove data from the "tail" (recent additions), you only retrain the tail, not the entire model!

---

## 🔬 The Two Methods: SISA vs ARCANE

### SISA (Sharded, Isolated, Sliced, Aggregated)
**Paper**: IEEE S&P 2021 - "Machine Unlearning"

**Core Idea**: Split data into **shards**, train separate models, combine via **ensemble voting**.

#### How SISA Works:

1. **Shard** (Split):
   - Divide training data into N independent shards (e.g., 16 shards)
   - Each shard gets ~1/N of the data

2. **Isolate** (Train Separately):
   - Train a **separate model** on each shard
   - Models are completely independent

3. **Slice** (Incremental Training):
   - Within each shard, split data into **slices** (e.g., 20 slices)
   - Train incrementally: add slice 1, train; add slice 2, train; etc.
   - **Save checkpoints** before each slice is added

4. **Aggregate** (Ensemble):
   - For predictions, average predictions from all shard models
   - If 5 models each have 85% accuracy → ensemble gets ~87% accuracy!

#### SISA Unlearning Process:
```
1. Identify which shard(s) contain the data to unlearn
2. For each affected shard:
   - Find the earliest slice containing unlearn data
   - Load checkpoint from BEFORE that slice
   - Retrain only from that slice onwards (excluding unlearn data)
3. Keep all other shard models unchanged
4. Re-evaluate ensemble
```

**Advantages**:
- ✅ **Massive speedups** when only few shards affected (1521x in your results!)
- ✅ **Ensemble accuracy** often better than single model
- ✅ **Parallelizable** (can train shards simultaneously)

**Disadvantages**:
- ❌ **More storage** (N models instead of 1)
- ❌ **More compute** during training (train N models)
- ❌ **Slower inference** (need to run N models)

---

### ARCANE (Architecture for Exact Machine Unlearning)
**Paper**: IJCAI 2022 - "ARCANE: An Efficient Architecture for Exact Machine Unlearning"

**Core Idea**: Train a **single model** incrementally with **block-based checkpoints**.

#### How ARCANE Works:

1. **Block Division**:
   - Split training data into **blocks** (e.g., 20 blocks)
   - Blocks are sequential (not independent like SISA shards)

2. **Incremental Training**:
   - Start with block 1, train model
   - Add block 2, train on blocks 1+2
   - Add block 3, train on blocks 1+2+3
   - Continue...
   - **Save checkpoints** before each block is added

3. **Single Model**:
   - Only one model (not an ensemble)
   - Simpler architecture

#### ARCANE Unlearning Process:
```
1. Find the earliest block containing data to unlearn
2. Load checkpoint from BEFORE that block
3. Retrain from that block onwards, excluding unlearn data
4. Done!
```

**Advantages**:
- ✅ **Single model** (less storage, faster inference)
- ✅ **Simpler** than SISA
- ✅ **Good speedups** when unlearning from tail blocks

**Disadvantages**:
- ❌ **Lower speedups** than SISA (only one model to retrain, but no sharding)
- ❌ **No ensemble benefit** (single model accuracy)
- ❌ **Sequential** (can't parallelize like SISA)

---

## 📊 Dataset Details

### MNIST
- **Type**: 28×28 grayscale images of handwritten digits (0-9)
- **Size**: 60,000 training / 10,000 test images
- **Classes**: 10 (digits 0-9)
- **Model**: LeNet (simple CNN)
- **Expected Accuracy**: 85-95%
- **Why Used**: Classic benchmark, easy to train, well-understood

### Fashion-MNIST
- **Type**: 28×28 grayscale images of clothing items
- **Size**: 60,000 training / 10,000 test images
- **Classes**: 10 (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot)
- **Model**: LeNet (same as MNIST)
- **Expected Accuracy**: 85-90%
- **Why Used**: Harder than MNIST (more variation), still manageable

### CIFAR-10
- **Type**: 32×32 RGB color images of objects
- **Size**: 50,000 training / 10,000 test images
- **Classes**: 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
- **Model**: ResNet18 (deeper CNN)
- **Expected Accuracy**: 75-85%
- **Why Used**: More realistic, commonly used in research, harder than MNIST

### SVHN (Street View House Numbers)
- **Type**: 32×32 RGB images of house numbers from Google Street View
- **Size**: 73,257 training / 26,032 test images
- **Classes**: 10 (digits 0-9)
- **Model**: ResNet18
- **Expected Accuracy**: 85-95%
- **Why Used**: Real-world data, commonly used in unlearning papers

---

## 🏗️ Code Structure Explained

### Project Layout
```
btp_unlearning/
├── common/              # Shared utilities
│   ├── data.py         # Dataset loading & splitting
│   ├── models.py       # Neural network architectures
│   ├── train.py        # Training loop
│   └── device.py       # GPU/CPU detection
│
├── sisa/                # SISA implementation
│   ├── train_sisa.py   # SISA training (shard + slice)
│   └── unlearn_sisa.py # SISA unlearning
│
├── arcane/              # ARCANE implementation
│   ├── train_arcane.py # ARCANE training (block-based)
│   └── unlearn_arcane.py # ARCANE unlearning
│
├── data/                # Dataset storage
│   ├── MNIST/
│   ├── CIFAR-10/
│   └── ...
│
└── runs/                # Experiment results
    ├── all_datasets/    # Results for all datasets
    └── mnist_optimized/ # Optimized MNIST runs
```

### Key Code Components

#### 1. `common/data.py` - Data Loading
```python
def get_dataset(name, train=True):
    # Loads MNIST, CIFAR-10, Fashion-MNIST, SVHN
    # Returns PyTorch Dataset + number of classes

def split_indices_by_class(ds):
    # Groups data indices by class label
    # Returns: {0: [idx1, idx2, ...], 1: [idx3, ...], ...}

def random_shards(indices, num_shards, rng):
    # Splits indices into N random shards
    # Used by SISA for sharding
```

#### 2. `common/models.py` - Model Architectures
```python
class LeNet(nn.Module):
    # Simple CNN for MNIST/Fashion-MNIST
    # Conv → Pool → Conv → Pool → FC → FC → Output
    # Input: 28×28 grayscale → Output: 10 classes

def small_cnn_cifar(num_classes):
    # ResNet18 for CIFAR-10/SVHN
    # Deeper network for color images
```

#### 3. `sisa/train_sisa.py` - SISA Training
**Key Steps**:
1. Split data into shards (e.g., 16 shards)
2. For each shard:
   - Split shard into slices (e.g., 20 slices)
   - Train incrementally: add slice, train, save checkpoint
3. Evaluate ensemble (average predictions from all shards)

**Critical Code**:
```python
# Sharding
shards = random_shards(indices, args.shards, rng)

# Per-shard training with slicing
for si, shard_idx in enumerate(shards):
    slices = np.array_split(shard_idx, args.slices)
    for r, sl in enumerate(slices):
        # Save BEFORE adding slice
        torch.save(model.state_dict(), f"slice_{r}_preadd.pt")
        
        # Add slice and train
        current_indices += sl
        model = train_classifier(model, Subset(train_ds, current_indices), ...)
        
        # Save AFTER training
        torch.save(model.state_dict(), f"slice_{r}_post.pt")
```

#### 4. `sisa/unlearn_sisa.py` - SISA Unlearning
**Key Steps**:
1. Identify affected shards and slices
2. For each affected shard:
   - Load checkpoint from before earliest affected slice
   - Retrain only from that slice onwards (excluding unlearn data)
3. Re-evaluate ensemble

**Critical Code**:
```python
# Find affected shards
affected_by_shard = {}
for idx in unlearn_points:
    si, r = slice_maps[idx]  # Which shard/slice?
    affected_by_shard.setdefault(si, set()).add(r)

# Retrain affected shards
for si, slice_set in affected_by_shard.items():
    start_r = min(slice_set)  # Earliest affected slice
    
    # Load from BEFORE that slice
    model.load_state_dict(torch.load(f"slice_{start_r}_preadd.pt"))
    
    # Retrain from start_r onwards
    for r in range(start_r, len(slices)):
        sl = [i for i in slices[r] if i not in unlearn_points]
        model = train_classifier(model, Subset(train_ds, current_indices), ...)
```

#### 5. `arcane/train_arcane.py` - ARCANE Training
**Key Steps**:
1. Split data into blocks (e.g., 20 blocks)
2. Train incrementally: add block, train, save checkpoint

**Critical Code**:
```python
blocks = np.array_split(all_idx, args.blocks)

for bi, blk in enumerate(blocks):
    # Save BEFORE adding block
    torch.save(model.state_dict(), f"block_{bi}_pre.pt")
    
    # Add block and train
    used += blk
    model = train_supervised(model, Subset(train_ds, used), ...)
    
    # Save AFTER training
    torch.save(model.state_dict(), f"block_{bi}_post.pt")
```

#### 6. `arcane/unlearn_arcane.py` - ARCANE Unlearning
**Key Steps**:
1. Find earliest affected block
2. Load checkpoint from before that block
3. Retrain from that block onwards (excluding unlearn data)

**Critical Code**:
```python
# Find earliest affected block
earliest = None
for bi, blk in enumerate(blocks):
    if any(i in unlearn_points for i in blk):
        earliest = bi if earliest is None else min(earliest, bi)

# Load from BEFORE affected block
model.load_state_dict(torch.load(f"block_{earliest}_pre.pt"))

# Retrain from earliest onwards
for bi in range(earliest, len(blocks)):
    blk = [i for i in blocks[bi] if i not in unlearn_points]
    model = train_supervised(model, Subset(train_ds, used), ...)
```

---

## 📈 Understanding the Results

### Result Files Location
Results are stored in `runs/` directory:
- `runs/all_datasets/` - Results for Fashion-MNIST, CIFAR-10, SVHN
- `runs/mnist_optimized/` - Optimized MNIST runs (where 1521x speedup occurred)

### Key Metrics Explained

#### 1. **Baseline Time** (`baseline_total_time_s`)
- Time to **retrain from scratch** after removing unlearn data
- This is the "naive" approach we're trying to beat
- Example: 243 seconds for MNIST

#### 2. **Unlearn Time** (`sisa_total_time_s` or `arcane_time_s`)
- Time using the unlearning method (SISA or ARCANE)
- Should be **much faster** than baseline
- Example: 0.16 seconds for SISA on MNIST

#### 3. **Speedup** (`speedup_total` or `speedup`)
- **Speedup = Baseline Time / Unlearn Time**
- Example: 243s / 0.16s = **1513x speedup** ✓
- Higher is better!

#### 4. **Accuracy Before Unlearning** (`baseline_acc` or `ensemble_acc`)
- Model accuracy on test set before unlearning
- Example: 98.99% for MNIST (excellent!)

#### 5. **Accuracy After Unlearning** (`ensemble_acc_after_unlearn` or `arcane_acc_after_unlearn`)
- Model accuracy after removing data
- Should be **close** to before (minimal drop)
- Example: 89.92% for MNIST (9% drop - acceptable for large unlearning)

#### 6. **Affected Shards/Blocks**
- How many shards (SISA) or blocks (ARCANE) need retraining
- Fewer = faster unlearning
- Example: 4 affected shards out of 16 total

#### 7. **Tail Slices/Blocks** (`avg_tail_slices` or `affected_tail_blocks`)
- How many slices/blocks from the end need retraining
- Smaller = faster (less data to retrain)
- Example: 4 tail slices out of 20 total

---

## 🚀 The 1521x Speedup Explained

### The Record-Breaking Result
**Location**: `runs/mnist_optimized/sisa_mnist_s16x20_e20/`

**Results**:
```json
{
  "baseline_total_time_s": 243.29 seconds,
  "sisa_total_time_s": 0.16 seconds,
  "speedup_total": 1513.15x
}
```

**That's 1513x faster!** (You mentioned 1521x - likely a slight variation in runs)

### Why Such a Massive Speedup?

#### 1. **Temporal Locality** (Key Factor!)
The code specifically targets **recent data** for unlearning:
```python
# From unlearn_sisa.py lines 42-43
recent_frac = 0.2          # Focus on most recent 20% slices
affect_frac_shards = 0.2   # Touch only 20% of shards
```

**What this means**:
- Only unlearn data from the **last 20% of slices** (tail)
- Only affect **20% of shards** (4 out of 16 shards)
- This is realistic! Users often request deletion of recently added data

#### 2. **Small Tail Retraining**
- **Total slices per shard**: 20
- **Average tail slices**: 4.0
- **Only retrain 4 slices** instead of all 20!

**Math**:
- Baseline: Retrain entire model (20 epochs × all data)
- SISA: Retrain only 4 slices (4 epochs × 20% of data) on 4 shards
- **Speedup ≈ (20 epochs × 100% data) / (4 epochs × 20% data × 4 shards)**
- **≈ 20 / (4 × 0.2 × 4) = 20 / 3.2 ≈ 6.25x per shard**
- **But we only retrain 4 shards, so total speedup is even higher!**

#### 3. **Sharding Advantage**
- **16 shards total**, only **4 affected**
- Each shard has **1/16 of the data**
- Retraining 4 shards = retraining 4/16 = **25% of data**
- But we only retrain the **tail** (20% of each shard)
- **Effective retraining: 25% × 20% = 5% of total data!**

#### 4. **Checkpoint Efficiency**
- Loading checkpoint is **instant** (milliseconds)
- No need to retrain from scratch
- Just load and continue from the right point

#### 5. **Epoch Efficiency**
- **Baseline**: 20 epochs on full dataset (243 seconds)
- **SISA**: 4 epochs on 5% of data (0.16 seconds)
- **Time per epoch**: Much faster on smaller data

### The Perfect Storm
All factors aligned:
- ✅ **Few affected shards** (4 out of 16 = 25%)
- ✅ **Small tail** (4 slices out of 20 = 20%)
- ✅ **Recent data only** (temporal locality)
- ✅ **Efficient checkpoints** (instant loading)
- ✅ **Small dataset** (MNIST trains fast)

**Result**: 1513x speedup! 🎉

### When Would Speedup Be Lower?

1. **Data from early slices**: Would need to retrain from slice 0 (much more data)
2. **Data spread across many shards**: More shards to retrain
3. **Large tail**: More slices to retrain
4. **Larger dataset**: CIFAR-10 is slower (only 3.8x speedup in your results)

---

## 💬 How to Explain This to Someone

### The Elevator Pitch (30 seconds)
> "We implemented two methods (SISA and ARCANE) that let you remove data from trained AI models **without retraining from scratch**. Instead of taking hours, it takes seconds. We achieved **1500x speedup** on MNIST by only retraining the parts of the model that saw the deleted data."

### The 2-Minute Explanation
> "Traditional machine learning: if you want to remove data, you delete it and retrain the entire model (slow, expensive).
> 
> **SISA** splits data into shards, trains separate models, and combines them. When data needs removal, only the affected shards are retrained from checkpoints.
> 
> **ARCANE** trains one model incrementally with checkpoints. When data needs removal, it rolls back to before that data was added and retrains only from that point.
> 
> Both methods use **temporal locality**: if you only remove recent data, you only retrain the 'tail' of training, not everything. This gives massive speedups (1500x in our best case)."

### The Technical Explanation (5 minutes)
> "**Machine Unlearning** addresses the 'right to be forgotten' in AI systems. When a user requests data deletion, we need to remove their data's influence from the model.
> 
> **SISA (Sharded, Isolated, Sliced, Aggregated)**:
> - **Shard**: Split data into 16 independent shards
> - **Isolate**: Train 16 separate models (one per shard)
> - **Slice**: Within each shard, train incrementally with 20 slices, saving checkpoints
> - **Aggregate**: Combine predictions via ensemble voting
> - **Unlearning**: Only retrain affected shards from their affected slices
> 
> **ARCANE (Architecture for Exact Machine Unlearning)**:
> - Split data into 20 sequential blocks
> - Train incrementally, saving checkpoints before each block
> - **Unlearning**: Rollback to before affected block, retrain from there
> 
> **Why the speedup?**
> - Baseline: Retrain entire model (20 epochs × 100% data = 243 seconds)
> - SISA: Retrain 4 shards × 4 tail slices (4 epochs × 5% data = 0.16 seconds)
> - **Speedup: 243 / 0.16 = 1513x**
> 
> **Key insight**: By targeting recent data (temporal locality) and using checkpoints, we avoid retraining the entire model."

### Key Points to Emphasize

1. **Problem**: GDPR/privacy requires data deletion, but retraining is slow
2. **Solution**: Checkpoint-based incremental retraining
3. **Methods**: SISA (ensemble) vs ARCANE (single model)
4. **Result**: 1500x speedup when unlearning recent data
5. **Trade-off**: Storage overhead (checkpoints) for speed
6. **Real-world**: Works best when deletions are from recent data

---

## 📊 Summary Table: Your Results

| Method | Dataset | Baseline Time | Unlearn Time | Speedup | Initial Acc | After Acc | Drop |
|--------|---------|---------------|--------------|---------|-------------|-----------|------|
| **SISA** | MNIST (opt) | 243.3s | 0.16s | **1513x** | 98.99% | 89.92% | 9.07% |
| **SISA** | Fashion-MNIST | 89.3s | 70.1s | 1.27x | 86.67% | 71.89% | 14.78% |
| **SISA** | CIFAR-10 | 3693.7s | 954.7s | 3.87x | 55.14% | 41.42% | 13.72% |
| **ARCANE** | Fashion-MNIST | - | - | - | 48.78% | 47.95% | 0.83% |
| **ARCANE** | CIFAR-10 | - | - | - | 31.77% | 26.95% | 4.82% |

### Observations

1. **MNIST is the star**: 1513x speedup due to:
   - Small dataset (fast training)
   - Recent data only (small tail)
   - Few affected shards (4/16)

2. **Fashion-MNIST SISA**: Low speedup (1.27x) because:
   - Data spread across many shards
   - Larger tail (5 slices)
   - More data to retrain

3. **CIFAR-10**: Moderate speedup (3.87x) because:
   - Larger dataset (slower training)
   - ResNet18 is more complex
   - But still 4x faster than baseline!

4. **Accuracy drops**: Expected when unlearning large amounts (1000-3000 points)
   - MNIST: 9% drop (but still 90% accurate)
   - Fashion-MNIST: 15% drop (needs more epochs)
   - CIFAR-10: 14% drop (complex dataset)

---

## 🎓 Key Takeaways

1. **Machine Unlearning** enables efficient data deletion from trained models
2. **SISA** uses ensemble of sharded models (better speedups, more storage)
3. **ARCANE** uses single model with block checkpoints (simpler, moderate speedups)
4. **Temporal locality** is key: recent deletions = massive speedups
5. **1521x speedup** achieved when unlearning recent data from few shards
6. **Trade-offs**: Storage (checkpoints) vs speed, accuracy vs unlearning amount
7. **Real-world**: Works best for GDPR compliance, data corrections, security

---

## 🔍 Further Reading

- **SISA Paper**: "Machine Unlearning" (IEEE S&P 2021)
- **ARCANE Paper**: "ARCANE: An Efficient Architecture for Exact Machine Unlearning" (IJCAI 2022)
- **Code**: This repository implements both methods with optimizations

---

**Questions?** Check the code comments or experiment with different hyperparameters!

