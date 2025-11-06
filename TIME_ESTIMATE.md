# Time Estimates for M1 Mac

## ✅ Good News: MPS (Metal) Acceleration Enabled!

Your code now automatically uses **MPS (Metal Performance Shaders)** on your M1 Mac, which provides GPU acceleration similar to CUDA on NVIDIA GPUs. This makes training **3-5x faster** than CPU-only.

## ⏱️ Estimated Runtime

### With MPS (Metal GPU) - Current Setup

| Dataset | Method | Epochs | Estimated Time | Notes |
|---------|--------|--------|----------------|-------|
| **Fashion-MNIST** | SISA | 15 | **30-45 min** | LeNet, 5 shards × 10 slices |
| **Fashion-MNIST** | ARCANE | 15 | **15-25 min** | LeNet, 6 blocks |
| **CIFAR-10** | SISA | 30 | **2-3 hours** | ResNet18, 5 shards, limit=1000 |
| **CIFAR-10** | ARCANE | 30 | **1-1.5 hours** | ResNet18, 6 blocks, limit=1000 |
| **SVHN** | SISA | 25 | **1.5-2 hours** | ResNet18, 5 shards |
| **SVHN** | ARCANE | 25 | **45-75 min** | ResNet18, 6 blocks |

**Total Estimated Time: 6-9 hours** for all experiments

### Breakdown by Component

#### Fashion-MNIST (LeNet - Small Model)
- **SISA Training**: ~5-8 min per shard × 5 shards = 25-40 min
- **SISA Unlearning**: ~2-5 min
- **ARCANE Training**: ~15-25 min
- **ARCANE Unlearning**: ~2-5 min
- **Total**: ~45-75 min

#### CIFAR-10 (ResNet18 - Large Model, Limited Dataset)
- **SISA Training**: ~20-30 min per shard × 5 shards = 1.5-2.5 hours
- **SISA Unlearning**: ~5-10 min
- **ARCANE Training**: ~1-1.5 hours
- **ARCANE Unlearning**: ~5-10 min
- **Total**: ~3-4 hours

#### SVHN (ResNet18 - Large Model)
- **SISA Training**: ~15-20 min per shard × 5 shards = 1.25-1.75 hours
- **SISA Unlearning**: ~5-10 min
- **ARCANE Training**: ~45-75 min
- **ARCANE Unlearning**: ~5-10 min
- **Total**: ~2-3 hours

## 🚀 Speed Comparison

### MPS (Metal) vs CPU on M1 Mac:
- **LeNet (Fashion-MNIST)**: ~3-4x faster
- **ResNet18 (CIFAR-10/SVHN)**: ~4-5x faster

### If Running on CPU Only:
- Fashion-MNIST: ~2-3 hours
- CIFAR-10: ~8-12 hours
- SVHN: ~6-9 hours
- **Total**: ~16-24 hours

## ⚡ Optimization Tips

1. **Run Overnight**: Start before bed, check in morning
2. **Run in Background**: Use `nohup` or `screen` to run in background
3. **Reduce Epochs for Testing**: Use fewer epochs for quick tests
4. **Use Limits**: CIFAR-10 already uses `limit_per_class=1000` for speed
5. **Run Sequentially**: Don't run multiple experiments simultaneously (memory)

## 📊 Factors Affecting Speed

### Faster:
- ✅ MPS (Metal) enabled (current setup)
- ✅ Smaller batch sizes (64 vs 128)
- ✅ Dataset limits (CIFAR-10 uses 1000 per class)
- ✅ Fewer epochs

### Slower:
- ❌ CPU-only mode
- ❌ Larger batch sizes
- ❌ Full dataset (no limits)
- ❌ More epochs
- ❌ More shards/slices

## 🔧 Potential Issues & Solutions

### MPS Limitations:
1. **num_workers > 0**: MPS sometimes has issues with multiprocessing
   - **Solution**: Already set to `num_workers=2` (should be fine)
   - If errors occur, change to `num_workers=0`

2. **Memory**: M1 Macs have unified memory (8-16GB typically)
   - **Solution**: Batch sizes are already optimized (64 for ResNet18)

3. **First Run**: MPS may be slower on first run (compilation)
   - **Solution**: Subsequent runs will be faster

### If MPS Fails:
The code will automatically fall back to CPU, but training will be slower.

## 📝 Recommended Approach

### Option 1: Full Run (Recommended)
```bash
# Start before bed or when you have 6-9 hours
python scripts/run_all_datasets.py
```

### Option 2: Quick Test (1-2 hours)
```bash
# Test with fewer epochs first
python sisa/train_sisa.py --dataset fashionmnist --epochs 5 --shards 3 --slices 5 --out runs/test
```

### Option 3: One Dataset at a Time
```bash
# Run Fashion-MNIST first (fastest, ~1 hour)
python sisa/train_sisa.py --dataset fashionmnist --epochs 15 --shards 5 --slices 10 --out runs/fashionmnist_sisa
python sisa/unlearn_sisa.py --run-dir runs/fashionmnist_sisa --unlearn-frac 0.01
```

## 🎯 Realistic Expectations

Based on M1 Mac benchmarks:
- **Fashion-MNIST**: Should complete in **~1 hour** total
- **CIFAR-10**: Should complete in **~3-4 hours** total (with limits)
- **SVHN**: Should complete in **~2-3 hours** total

**Total**: **6-8 hours** for all experiments (with MPS acceleration)

## 💡 Pro Tips

1. **Monitor Progress**: Check `runs/all_datasets/*/summary.json` for completed runs
2. **Check GPU Usage**: Use Activity Monitor to see MPS usage
3. **Save Power**: Plug in your Mac (MPS uses more power)
4. **Close Other Apps**: Free up memory for training

Good luck! 🚀

