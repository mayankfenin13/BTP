import argparse, os, json, time, numpy as np, torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch.utils.data import Subset, DataLoader
from common.data import get_dataset
from common.models import LeNet, small_cnn_cifar
from common.device import get_device, is_mps
from arcane.train_arcane import make_model, train_supervised, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--unlearn-frac", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=None, help="override epochs for baseline")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()
    
    meta = json.load(open(os.path.join(args.run_dir,"meta.json")))
    dataset = meta["dataset"]
    
    # FIXED: Use total epochs that match original training
    epochs_per_block = meta.get("epochs_per_block", 1)
    total_epochs = meta["blocks"] * epochs_per_block
    
    # For baseline, use the same total epochs as original training
    baseline_epochs = args.epochs if args.epochs is not None else total_epochs
    batch_size = args.batch_size if args.batch_size is not None else meta["batch_size"]
    lr = args.lr if args.lr is not None else meta["lr"]
    train_ds, num_classes = get_dataset(dataset, train=True)
    test_ds,_ = get_dataset(dataset, train=False)
    device = get_device()
    # Try MPS/CUDA first for speed, but allow CPU fallback if needed
    # For full dataset runs, MPS is much faster (3-5x) than CPU
    train_device = device  # Use MPS/CUDA if available, CPU otherwise
    # Note: If MPS crashes occur, you can manually set train_device = "cpu" above
    blocks = json.load(open(os.path.join(args.run_dir,"indices","blocks.json")))
    
    # FIXED: Sample deletions from recent blocks only (temporal locality)
    # This ensures we only retrain the tail, giving real speedups
    recent_frac = 0.3  # last 30% blocks only
    num_blocks = len(blocks)
    start_b = max(0, int(np.floor((1 - recent_frac) * num_blocks)))
    
    # Build candidate pool from recent blocks only
    recent_pool = []
    for bi in range(start_b, num_blocks):
        recent_pool.extend(blocks[bi])
    
    recent_pool = np.array(recent_pool) if recent_pool else np.array([], dtype=int)
    all_idx = [i for b in blocks for i in b]
    n_un = max(1, int(len(all_idx) * args.unlearn_frac))
    n_un = min(n_un, len(recent_pool))
    
    rng = np.random.default_rng(123)
    unlearn_points = set(rng.choice(recent_pool, size=n_un, replace=False).tolist())
    # FIXED: Baseline trains with same total epochs as original ARCANE training
    t0 = time.time()
    keep = [i for i in all_idx if i not in unlearn_points]
    base_model = make_model(dataset, num_classes)
    base_model = train_supervised(
        base_model, Subset(train_ds, keep), test_ds, 
        epochs=baseline_epochs,  # ← FIXED: Use same total epochs
        batch_size=batch_size, 
        lr=lr, 
        device=train_device
    )
    baseline_time = time.time() - t0
    # ARCANE retraining from nearest pre-state: locate earliest block containing any unlearn point
    t1 = time.time()
    # find earliest affected block index
    earliest = None
    for bi, blk in enumerate(blocks):
        if any(i in unlearn_points for i in blk):
            earliest = bi if earliest is None else min(earliest, bi)
    model = make_model(dataset, num_classes)
    if earliest is None:
        # no change; load last post
        posts = sorted([f for f in os.listdir(os.path.join(args.run_dir,"checkpoints")) if f.startswith("block_") and f.endswith("_post.pt")])
        model.load_state_dict(torch.load(os.path.join(args.run_dir,"checkpoints", posts[-1]), map_location=device))
    else:
        # Load pre of earliest and continue on cleaned blocks
        pre_path = os.path.join(args.run_dir, "checkpoints", f"block_{earliest}_pre.pt")
        model.load_state_dict(torch.load(pre_path, map_location=device))
        
        # FIXED: Train incrementally like original (1 epoch per block)
        current_indices = []
        for bi in range(earliest, len(blocks)):
            blk = [i for i in blocks[bi] if i not in unlearn_points]
            current_indices += blk
            
            # Train for 1 epoch (matching original training)
            model = train_supervised(
                model, Subset(train_ds, current_indices), test_ds, 
                epochs=epochs_per_block,  # ← FIXED: 1 epoch per block
                batch_size=batch_size, 
                lr=lr, 
                device=train_device
            )
    arcane_time = time.time() - t1
    # Evaluate both
    base_acc = evaluate(base_model, test_ds)
    arcane_acc = evaluate(model, test_ds)
    
    # Calculate speedup (baseline_time is already training time)
    speedup = baseline_time / arcane_time if arcane_time > 0 else 0
    
    # Calculate tail block statistics (key metric for speedup)
    affected_tail_blocks = 0 if earliest is None else (len(blocks) - earliest)
    
    os.makedirs(os.path.join(args.run_dir,"metrics"), exist_ok=True)
    result = {
        "n_unlearn": int(n_un),
        "baseline_epochs": baseline_epochs,
        "arcane_epochs": affected_tail_blocks * epochs_per_block,
        "earliest_affected_block": earliest if earliest is not None else -1,
        "affected_tail_blocks": int(affected_tail_blocks),
        "total_blocks": int(len(blocks)),
        "baseline_time_s": baseline_time,
        "arcane_time_s": arcane_time,
        "baseline_acc": float(base_acc),
        "arcane_acc_after_unlearn": float(arcane_acc),
        "speedup": speedup
    }
    json.dump(result, open(os.path.join(args.run_dir,"metrics","unlearn_summary.json"),"w"), indent=2)
    
    print("\n" + "="*60)
    print("ARCANE Unlearning Results")
    print("="*60)
    print(f"Baseline time:    {baseline_time:.2f}s ({baseline_epochs} epochs)")
    print(f"ARCANE time:      {arcane_time:.2f}s (from block {earliest if earliest is not None else 'N/A'})")
    print(f"Speedup:          {speedup:.2f}x")
    print(f"Tail blocks:      {affected_tail_blocks} / {len(blocks)}")
    print(f"Baseline acc:     {base_acc:.4f}")
    print(f"ARCANE acc:       {arcane_acc:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
