# Line-by-Line Code Analysis

## 🔬 Deep Dive into Critical Functions

### 1. LeNet Forward Pass (`common/models.py`)

```python
def forward(self, x):
    # Line 17: First convolution layer
    # Input shape: [batch_size, 1, 28, 28]
    # Conv1: 1 input channel → 6 output channels, kernel size 5×5
    # Output after conv1: [batch, 6, 24, 24] (28-5+1 = 24)
    x = self.pool(F.relu(self.conv1(x)))
    # After ReLU: [batch, 6, 24, 24] (same size)
    # After MaxPool(2,2): [batch, 6, 12, 12] (24/2 = 12)
    # MaxPool reduces spatial dimensions, adds translation invariance
    
    # Line 18: Second convolution layer
    # Input: [batch, 6, 12, 12]
    # Conv2: 6 input channels → 16 output channels, kernel size 5×5
    # Output after conv2: [batch, 16, 8, 8] (12-5+1 = 8)
    x = self.pool(F.relu(self.conv2(x)))
    # After ReLU: [batch, 16, 8, 8]
    # After MaxPool(2,2): [batch, 16, 4, 4] (8/2 = 4)
    
    # Line 19: Flatten for fully connected layers
    # Reshape from [batch, 16, 4, 4] → [batch, 256]
    # 16 × 4 × 4 = 256 features
    x = x.view(x.size(0), -1)
    # x.size(0) = batch_size, -1 = auto-calculate remaining dimensions
    
    # Line 20: First fully connected layer
    # Input: [batch, 256] → Output: [batch, 120]
    # ReLU activation introduces non-linearity
    x = F.relu(self.fc1(x))
    # ReLU(x) = max(0, x) - enables learning non-linear patterns
    
    # Line 21: Second fully connected layer
    # Input: [batch, 120] → Output: [batch, 84]
    feat = F.relu(self.fc2(x))
    # 'feat' stores the feature representation (penultimate layer)
    # This is useful for feature analysis or transfer learning
    
    # Line 22: Output layer (no activation - raw logits)
    # Input: [batch, 84] → Output: [batch, 10]
    # 10 classes for MNIST (digits 0-9)
    logits = self.fc3(feat)
    # Logits are unnormalized scores for each class
    # Softmax will be applied during loss computation or inference
    
    # Lines 23-25: Return logic
    if self.out_feat:
        return logits, feat  # Return both predictions and features
    return logits  # Standard case: return only predictions
```

**Why Each Layer Matters**:
- **Conv1**: Learns low-level features (edges, curves)
- **Pool1**: Reduces computation, adds invariance
- **Conv2**: Learns high-level features (digit parts)
- **Pool2**: Further reduces spatial dimensions
- **FC1**: Combines features into 120-dimensional space
- **FC2**: Refines to 84-dimensional representation
- **FC3**: Maps to 10 class scores

---

### 2. Training Loop (`common/train.py`)

```python
def train_classifier(model, train_ds, test_ds, epochs=5, batch_size=128, 
                     lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Line 9: Move model to GPU/CPU
    model = model.to(device)
    # CUDA enables GPU acceleration (10-100x faster on large models)
    # Falls back to CPU if GPU unavailable
    
    # Line 10: Initialize Adam optimizer
    opt = optim.Adam(model.parameters(), lr=lr)
    # Adam = Adaptive Moment Estimation
    # - Maintains per-parameter learning rates
    # - Uses moving averages of gradients (momentum)
    # - lr=1e-3 is well-tuned for MNIST (not too high, not too low)
    # - Better than SGD for this task (faster convergence)
    
    # Line 11: Loss function for classification
    loss_fn = nn.CrossEntropyLoss()
    # CrossEntropyLoss = softmax + negative log-likelihood
    # - Softmax: converts logits to probabilities
    # - NLL: penalizes wrong predictions
    # - Single function = more numerically stable than separate softmax+NLL
    
    # Line 12: Create data loader for training
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    # batch_size=128: processes 128 samples at once
    #   - Larger batches: more stable gradients, but slower
    #   - Smaller batches: faster updates, but noisier
    #   - 128 is good balance for MNIST
    # shuffle=True: randomizes order each epoch
    #   - Prevents model from memorizing order
    #   - Improves generalization
    # num_workers=2: parallel data loading (speeds up training)
    
    # Line 13: Create data loader for testing
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    # batch_size=256: larger for testing (no gradients computed)
    # shuffle=False: deterministic order for evaluation
    
    # Line 14: Start timing
    t0 = time.time()
    # Records training duration
    
    # Line 15: Epoch loop (one epoch = one pass through all training data)
    for _ in range(epochs):
        # Line 16: Set model to training mode
        model.train()
        # Enables dropout, batch normalization updates
        # Critical for proper training behavior
        
        # Line 17: Iterate through batches
        for x,y in train_loader:
            # x: batch of images [batch_size, channels, height, width]
            # y: batch of labels [batch_size]
            
            # Line 18: Move data to device (GPU/CPU)
            x,y = x.to(device), y.to(device)
            # GPU tensors must be on same device as model
            
            # Line 19: Zero out gradients from previous iteration
            opt.zero_grad()
            # PyTorch accumulates gradients by default
            # Must clear before each backward pass
            # Without this: gradients would accumulate incorrectly
            
            # Line 20: Forward pass through network
            logits = model(x)
            # Computes predictions for batch
            # Shape: [batch_size, num_classes]
            
            # Line 21: Compute loss
            loss = loss_fn(logits, y)
            # Compares predictions (logits) to true labels (y)
            # Returns scalar loss value
            # Higher loss = worse predictions
            
            # Line 22: Backward pass (backpropagation)
            loss.backward()
            # Computes gradients of loss w.r.t. all parameters
            # Stores gradients in parameter.grad
            # This is where the "learning" happens!
            
            # Line 23: Update model parameters
            opt.step()
            # Uses gradients to update weights
            # Adam: w = w - lr * m_t / (sqrt(v_t) + eps)
            #   m_t: momentum-corrected gradient
            #   v_t: variance-corrected gradient
            # This step actually changes the model
    
    # Line 24: Calculate total training time
    train_time = time.time() - t0
    # Returns seconds elapsed
    
    # Line 25: Evaluate on test set
    acc = evaluate(model, test_loader, device)
    # Computes accuracy: correct_predictions / total_predictions
    
    # Line 26: Return trained model, accuracy, and time
    return model, acc, train_time
```

**Key Training Principles**:
1. **Batch Processing**: Processes multiple samples together (efficient)
2. **Gradient Accumulation**: Gradients computed per batch, averaged
3. **Iterative Updates**: Many small steps toward optimal solution
4. **Adam Optimization**: Adapts learning rate per parameter

---

### 3. SISA Ensemble Prediction (`sisa/train_sisa.py`)

```python
@torch.no_grad()  # Decorator: disables gradient computation
def evaluate_ensemble(args, make_model, test_ds):
    # @torch.no_grad() speeds up inference:
    # - No need to track gradients during evaluation
    # - Reduces memory usage
    # - Faster computation
    
    # Line 71: Detect device (GPU/CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Line 72: Create test data loader
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    
    # Line 73: List to store all shard models
    all_models = []
    
    # Lines 74-82: Load models from all shards
    for si in range(args.shards):
        shard_dir = os.path.join(args.out,"checkpoints", f"shard_{si}")
        model = make_model().to(device)  # Create fresh model instance
        
        # Line 79: Find final checkpoint (last slice post-state)
        posts = sorted([f for f in os.listdir(shard_dir) 
                       if f.startswith("slice_") and f.endswith("_post.pt")])
        # Sorts files: slice_0_post.pt, slice_1_post.pt, ..., slice_N_post.pt
        # Takes last one (fully trained model)
        
        # Line 80: Load model weights
        model.load_state_dict(torch.load(os.path.join(shard_dir, posts[-1]), 
                                        map_location=device))
        # Loads saved weights into model
        # map_location: ensures tensors on correct device
        
        # Line 81: Set to evaluation mode
        model.eval()
        # Disables dropout, fixes batch norm statistics
        # Critical for consistent inference
        
        # Line 82: Add to ensemble
        all_models.append(model)
    
    # Lines 83-89: Evaluate ensemble
    total, correct = 0, 0
    
    for x,y in test_loader:  # Iterate through test batches
        x,y = x.to(device), y.to(device)
        
        # Line 86: Get predictions from all models
        logits_stack = torch.stack([m(x) for m in all_models], dim=0)
        # Input x: [batch_size, channels, height, width]
        # Each m(x): [batch_size, num_classes]
        # Stack result: [num_shards, batch_size, num_classes]
        # Example: 5 shards → [5, 256, 10]
        
        # Line 87: Ensemble prediction (KEY TO HIGH ACCURACY!)
        pred = logits_stack.softmax(-1).mean(0).argmax(-1)
        # Step-by-step:
        # 1. softmax(-1): Apply softmax along class dimension
        #    Converts logits to probabilities
        #    [5, 256, 10] → [5, 256, 10] (probabilities sum to 1 per model)
        #    Each [256, 10] has rows summing to 1.0
        #
        # 2. .mean(0): Average across shards (dimension 0)
        #    [5, 256, 10] → [256, 10]
        #    Averages probability distributions from all models
        #    Example: Model 1 says [0.1, 0.8, 0.1] for class 1
        #             Model 2 says [0.2, 0.7, 0.1] for class 1
        #             Average: [0.15, 0.75, 0.1] (more confident!)
        #
        # 3. argmax(-1): Pick class with highest average probability
        #    [256, 10] → [256] (one prediction per sample)
        #    Returns index of highest probability
        
        # Lines 88-89: Count correct predictions
        total += y.size(0)  # Total samples in batch
        correct += (pred == y).sum().item()  # Count matches
        # pred == y: element-wise comparison → boolean tensor
        # .sum(): counts True values (correct predictions)
        # .item(): converts single-element tensor to Python number
    
    # Line 90: Return accuracy
    return correct/total
```

**Why Ensemble Works So Well**:
- **Diversity**: Each model trained on different data
- **Error Reduction**: If one model is wrong, others compensate
- **Probability Averaging**: More stable than hard voting
- **Theoretical Guarantee**: Ensemble of N models with accuracy p 
  → ensemble accuracy ≈ 1 - (1-p)^N

**Example Calculation**:
- Individual model accuracy: 85%
- Ensemble of 5 models: 1 - (1-0.85)^5 = 1 - 0.15^5 = 1 - 0.00008 = 99.99%
- In practice: ~87.5% (due to correlated errors)

---

### 4. SISA Unlearning (`sisa/unlearn_sisa.py`)

```python
def main():
    # ... argument parsing ...
    
    # Lines 29-33: Select points to unlearn
    all_idx = np.array([i for shard in shards for i in shard])  # Flatten all indices
    n_un = max(1, int(len(all_idx)*args.unlearn_frac))  # Calculate number to unlearn
    # Example: 10000 samples, 0.01 fraction → 100 samples
    
    rng = np.random.default_rng(123)  # Fixed seed for reproducibility
    unlearn_points = rng.choice(all_idx, size=n_un, replace=False).tolist()
    # Randomly select n_un indices without replacement
    # These are the points we want to "forget"
    
    # Lines 34-40: Map each point to (shard, slice)
    slice_maps = {}
    for si in range(len(shards)):
        slices = json.load(open(os.path.join(args.run_dir,"indices", 
                                            f"slices_shard_{si}.json")))
        for r, sl in enumerate(slices):  # r = slice index, sl = slice indices
            for idx in sl:
                slice_maps[idx] = (si, r)  # Store mapping: idx → (shard, slice)
    # This allows us to quickly find which shard/slice contains each unlearn point
    
    # Lines 46-51: Baseline retraining (from scratch)
    baseline_t0 = time.time()
    keep = [i for i in all_idx.tolist() if i not in set(unlearn_points)]
    # Remove unlearn points from training set
    # set() lookup is O(1), making this fast
    
    base_model = make_model()  # Fresh model
    base_model, base_acc, base_time = train_classifier(
        base_model, Subset(train_ds, keep), test_ds, 
        epochs=epochs, batch_size=batch_size, lr=lr)
    # Train from scratch on remaining data
    # This is the "gold standard" - what we compare against
    
    baseline_total_time = time.time() - baseline_t0
    
    # Lines 52-56: Identify affected shards
    affected_by_shard = {}
    for idx in unlearn_points:
        si, r = slice_maps[idx]  # Get shard/slice for this point
        affected_by_shard.setdefault(si, set()).add(r)
    # Result: {shard_0: {slice_2, slice_5}, shard_3: {slice_1}, ...}
    # Only these shards need retraining
    
    # Lines 57-75: SISA unlearning (only retrain affected shards)
    un_t0 = time.time()
    
    for si, slice_set in affected_by_shard.items():  # For each affected shard
        shard_dir = os.path.join(args.run_dir,"checkpoints", f"shard_{si}")
        slices = json.load(open(os.path.join(args.run_dir,"indices", 
                                            f"slices_shard_{si}.json")))
        
        # Line 62: Find earliest affected slice
        start_r = min(slice_set)
        # If slices 2, 5, 7 are affected → start from slice 2
        # Need to retrain from before slice 2 was added
        
        # Lines 64-66: Load model from BEFORE affected slice
        model = make_model()
        pre_path = os.path.join(shard_dir, f"slice_{start_r}_preadd.pt")
        model.load_state_dict(torch.load(pre_path, map_location="cpu"))
        # CRITICAL: Load checkpoint from BEFORE slice was added
        # This ensures unlearned data never influenced the model state
        
        # Lines 68-72: Retrain from affected slice onwards
        new_indices = []
        for r in range(start_r, len(slices)):  # From affected slice to end
            sl = [i for i in slices[r] if i not in set(unlearn_points)]
            # Filter out unlearn points from this slice
            # Only keep data points we want to retain
            
            new_indices += sl  # Accumulate indices seen so far
            
            # Retrain on accumulated data (excluding unlearn points)
            model, acc, _ = train_classifier(
                model, Subset(train_ds, new_indices), test_ds, 
                epochs=epochs, batch_size=batch_size, lr=lr)
            # Incremental training: each iteration adds more data
            # Mimics original training process, but without unlearn points
        
        # Line 74: Save retrained model
        torch.save(model.state_dict(), os.path.join(shard_dir, f"after_unlearn.pt"))
    
    sisa_time = time.time() - un_t0
    
    # Line 77: Evaluate ensemble after unlearning
    ensemble_acc = evaluate_ensemble(args.run_dir, len(shards), make_model, test_ds)
    # Uses retrained models for affected shards, original models for unaffected
    
    # Lines 78-86: Save results
    json.dump({
        "n_unlearn": n_un,
        "baseline_total_time_s": baseline_total_time,
        "baseline_train_time_s": base_time,
        "baseline_acc": base_acc,
        "sisa_total_time_s": sisa_time,
        "ensemble_acc_after_unlearn": ensemble_acc
    }, open(os.path.join(args.run_dir,"metrics","unlearn_summary.json"),"w"), indent=2)
```

**Why This Maintains Accuracy**:
1. **Exact State Recovery**: Loads checkpoint from before affected slice
2. **Minimal Retraining**: Only affected shards retrained
3. **Preserves Ensemble**: Unaffected shards keep original models
4. **Correct Process**: Retrains incrementally, maintaining training dynamics

**Time Savings**:
- Baseline: Retrain entire model from scratch
- SISA: Only retrain affected shards from affected slices
- Result: ~10x faster (though in your run, SISA was slower due to overhead)

---

### 5. Data Splitting (`common/data.py`)

```python
def split_indices_by_class(ds) -> Dict[int, List[int]]:
    # Line 38: Get labels from dataset
    targets = ds.targets if hasattr(ds, "targets") else ds.labels
    # Different torchvision datasets use different attribute names
    # MNIST: .targets, SVHN: .labels
    
    # Line 39-40: Convert to list if tensor
    if torch.is_tensor(targets):
        targets = targets.tolist()
    # PyTorch tensors need conversion for indexing
    
    # Line 41: Initialize dictionary mapping class → list of indices
    class_to_idx = {c: [] for c in set(targets)}
    # set(targets): gets unique class labels (e.g., {0,1,2,...,9} for MNIST)
    # Creates empty list for each class
    
    # Lines 42-43: Populate dictionary
    for i, y in enumerate(targets):
        class_to_idx[int(y)].append(i)
    # For each sample: add its index to its class's list
    # Result: {0: [0, 1, 5, ...], 1: [2, 7, 12, ...], ...}
    
    return class_to_idx
```

**Why This Matters**:
- **Class Balancing**: Ensures each class has equal representation
- **ARCANE Requirement**: ARCANE trains per-class models
- **Fair Evaluation**: Prevents bias toward majority classes

---

## 🎓 Key Takeaways

1. **Ensemble Prediction** (Line 87 in `sisa/train_sisa.py`):
   - Softmax averaging is more stable than hard voting
   - Key to 87.5% accuracy

2. **Checkpoint Recovery** (Line 65 in `sisa/unlearn_sisa.py`):
   - Loading pre-slice state ensures exact unlearning
   - Critical for correctness

3. **Incremental Training**:
   - Training on accumulated slices improves convergence
   - Better than training on all data at once

4. **Adam Optimizer**:
   - Adaptive learning rates per parameter
   - Faster convergence than SGD

5. **Batch Processing**:
   - Processes multiple samples efficiently
   - Enables GPU acceleration

Each line contributes to the final accuracy and correctness of the unlearning process!

