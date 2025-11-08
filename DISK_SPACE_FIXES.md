# Disk Space and Temp Directory Fixes

## Issues Fixed

### 1. **SISA CIFAR-10: Checkpoint Write Failures**
**Error**: `RuntimeError: PytorchStreamWriter failed writing file data/84: file write failed`

**Cause**: 
- Disk is 100% full (only 559MB free)
- With 10 shards × 40 slices = 800 checkpoints
- Each ResNet18 checkpoint is ~45MB = ~36GB total

**Fix Applied**:
- ✅ Added retry logic for checkpoint saving (3 retries with 1s delay)
- ✅ Set temp directory to output directory (avoids /tmp issues)
- ✅ Better error messages

### 2. **Arcane CIFAR-10: Temp Directory Not Found**
**Error**: `FileNotFoundError: No usable temporary directory found`

**Cause**: 
- Temp directory issues due to disk space
- System temp directories not accessible

**Fix Applied**:
- ✅ Set temp directory to output directory (`.tmp` subfolder)
- ✅ Sets `TMPDIR`, `TMP`, and `TEMP` environment variables

## Changes Made

### `sisa/train_sisa.py`:
1. Added `setup_temp_dir()` function to set temp directory
2. Added `save_checkpoint()` function with retry logic
3. Replaced `torch.save()` calls with `save_checkpoint()`

### `arcane/train_arcane.py`:
1. Added `setup_temp_dir()` function to set temp directory
2. Added `save_checkpoint()` function with retry logic
3. Replaced `torch.save()` calls with `save_checkpoint()`

## Important: Disk Space Warning

⚠️ **Your disk is 100% full (only 559MB free)!**

**Recommendations**:
1. **Free up disk space** before running CIFAR-10 experiments
   - CIFAR-10 will need ~36GB for checkpoints (10 shards × 40 slices)
   - Fashion-MNIST needs ~2GB (16 shards × 25 slices)

2. **Clean up old runs**:
   ```bash
   # Remove old experiment results
   rm -rf runs/all_datasets/*
   rm -rf runs/fast/*
   ```

3. **Monitor disk space**:
   ```bash
   df -h .
   ```

4. **Consider reducing checkpoints** (if needed):
   - Could save every N slices instead of every slice
   - But this affects unlearning accuracy

## How the Fixes Help

1. **Retry Logic**: Handles transient disk I/O errors
2. **Temp Directory**: Avoids system temp directory issues
3. **Better Errors**: Clearer error messages if disk is full

## Testing

The fixes are now in place. To test:

```bash
# Run CIFAR-10 SISA (will now handle disk errors better)
python sisa/train_sisa.py \
    --dataset cifar10 \
    --epochs 40 \
    --shards 10 \
    --slices 40 \
    --batch-size 128 \
    --lr 0.001 \
    --out runs/full_datasets_optimized/sisa_cifar10_s10x40_e40
```

**Note**: If disk space is still an issue, you may need to:
- Free up more space
- Reduce number of shards/slices
- Use a different output directory with more space

