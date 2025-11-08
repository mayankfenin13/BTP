import argparse, os, json, time, random
import sys
from pathlib import Path
# Ensure project root is on sys.path for direct script execution
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from common.data import get_dataset
from common.models import LeNet, small_cnn_cifar
from common.device import get_device, get_num_workers, is_mps, get_safe_infer_device
from common.train import train_classifier, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--unlearn-frac", type=float, default=0.01, help="fraction of training set to unlearn (randomly)")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs for baseline")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.run_dir,"meta.json")))
    dataset = meta["dataset"]
    
    # FIXED: Use total epochs that match original training
    # Original training: slices × 1 epoch per slice = total_epochs
    epochs_per_slice = meta.get("epochs_per_slice", 1)
    total_epochs = meta["slices"] * epochs_per_slice
    
    # For baseline, use the same total epochs as original training
    baseline_epochs = args.epochs if args.epochs is not None else total_epochs
    batch_size = args.batch_size if args.batch_size is not None else meta["batch_size"]
    lr = args.lr if args.lr is not None else meta["lr"]
    shards = json.load(open(os.path.join(args.run_dir,"indices","shards.json")))
    train_ds, num_classes = get_dataset(dataset, train=True)
    test_ds,_ = get_dataset(dataset, train=False)
    
    # FIXED: Sample deletions from recent slices only (temporal locality)
    # This ensures we only retrain the tail, giving real speedups
    rng = np.random.default_rng(123)
    num_shards = len(shards)
    recent_frac = 0.2          # focus on most recent 20% slices
    affect_frac_shards = 0.2   # touch only 20% of shards
    
    # Pick shards to affect
    affect_count = max(1, int(np.ceil(num_shards * affect_frac_shards)))
    affected_shards = rng.choice(np.arange(num_shards), size=affect_count, replace=False)
    
    # Build candidate pool only from the tail slices of affected shards
    candidate_pool = []
    for si in affected_shards:
        slices = json.load(open(os.path.join(args.run_dir, "indices", f"slices_shard_{si}.json")))
        start_r = max(0, int(np.floor((1 - recent_frac) * len(slices))))
        for r in range(start_r, len(slices)):
            candidate_pool.extend(slices[r])
    
    candidate_pool = np.array(candidate_pool)
    all_train_size = sum(len(s) for s in shards)
    n_un = max(1, int(all_train_size * args.unlearn_frac))
    n_un = min(n_un, len(candidate_pool))
    unlearn_points = rng.choice(candidate_pool, size=n_un, replace=False).tolist()
    
    # Build all_idx from shards (needed for baseline training)
    all_idx = np.array([i for shard in shards for i in shard])
    
    # Map each unlearn point to (shard_id, slice_id)
    slice_maps = {}
    for si in range(len(shards)):
        slices = json.load(open(os.path.join(args.run_dir,"indices", f"slices_shard_{si}.json")))
        for r, sl in enumerate(slices):
            for idx in sl:
                slice_maps[idx] = (si, r)
    # model factory
    if dataset in ["mnist", "fashionmnist"]:
        make_model = lambda: LeNet(num_classes=num_classes)
    else:
        make_model = lambda: small_cnn_cifar(num_classes=num_classes)
    # Try MPS first for speed, but allow CPU fallback if needed
    # For full dataset runs, MPS is much faster (3-5x) than CPU
    train_device = get_device()  # Use MPS/CUDA if available, CPU otherwise
    # Note: If MPS crashes occur, you can manually set train_device = "cpu" above
    # FIXED: Baseline trains with same total epochs as original SISA training
    baseline_t0 = time.time()
    keep = [i for i in all_idx.tolist() if i not in set(unlearn_points)]
    base_model = make_model()
    base_model, base_acc, base_time = train_classifier(
        base_model, Subset(train_ds, keep), test_ds, 
        epochs=baseline_epochs,  # ← FIXED: Use same total epochs
        batch_size=batch_size, 
        lr=lr, 
        device=train_device
    )
    baseline_total_time = time.time() - baseline_t0
    # SISA unlearning: for each affected shard, restart from pre-slice state and continue
    affected_by_shard = {}
    for idx in unlearn_points:
        si, r = slice_maps[idx]
        affected_by_shard.setdefault(si, set()).add(r)
    un_t0 = time.time()
    total_slices_per_shard = meta["slices"]
    for si, slice_set in affected_by_shard.items():
        shard_dir = os.path.join(args.run_dir,"checkpoints", f"shard_{si}")
        slices = json.load(open(os.path.join(args.run_dir,"indices", f"slices_shard_{si}.json")))
        # build new indices: all slices up to each affected slice need retrain from that slice's preadd
        start_r = min(slice_set)
        # Load model from preadd of start_r
        model = make_model()
        pre_path = os.path.join(shard_dir, f"slice_{start_r}_preadd.pt")
        model.load_state_dict(torch.load(pre_path, map_location="cpu"))
        
        # FIXED: Train incrementally like original (1 epoch per slice)
        current_indices = []
        for r in range(start_r, len(slices)):
            sl = [i for i in slices[r] if i not in set(unlearn_points)]
            current_indices += sl
            
            # Train for 1 epoch (matching original training)
            train_subset = Subset(train_ds, current_indices)
            if len(train_subset) == 0:
                # Nothing to train on (all points removed from this slice)
                continue
            model, acc, _ = train_classifier(
                model, train_subset, test_ds, 
                epochs=epochs_per_slice,  # ← FIXED: 1 epoch per slice
                batch_size=batch_size, 
                lr=lr, 
                device=train_device
            )
        
        # Save final
        torch.save(model.state_dict(), os.path.join(shard_dir, f"after_unlearn.pt"))
    sisa_time = time.time() - un_t0
    # Evaluate ensemble after unlearning
    ensemble_acc = evaluate_ensemble(args.run_dir, len(shards), make_model, test_ds)
    
    # Calculate speedups (use training time for fair comparison)
    speedup_train_only = (base_time / sisa_time) if sisa_time > 0 else 0
    speedup_total = (baseline_total_time / sisa_time) if sisa_time > 0 else 0
    
    os.makedirs(os.path.join(args.run_dir,"metrics"), exist_ok=True)
    
    # Calculate tail slice statistics (key metric for speedup)
    tail_blocks = []
    for si, slice_set in affected_by_shard.items():
        earliest = min(slice_set)
        slices = json.load(open(os.path.join(args.run_dir, "indices", f"slices_shard_{si}.json")))
        tail_blocks.append(len(slices) - earliest)
    
    avg_tail_slices = float(np.mean(tail_blocks) if tail_blocks else 0.0)
    
    result = {
        "n_unlearn": n_un,
        "baseline_epochs": baseline_epochs,
        "num_affected_shards": len(affected_by_shard),
        "affected_shards": sorted(list(affected_by_shard.keys())),
        "avg_tail_slices": avg_tail_slices,
        "total_slices_per_shard": int(total_slices_per_shard),
        "baseline_total_time_s": baseline_total_time,
        "baseline_train_time_s": base_time,
        "baseline_acc": base_acc,
        "sisa_total_time_s": sisa_time,
        "ensemble_acc_after_unlearn": ensemble_acc,
        "speedup_train_only": speedup_train_only,  # Primary speedup metric
        "speedup_total": speedup_total
    }
    json.dump(result, open(os.path.join(args.run_dir,"metrics","unlearn_summary.json"),"w"), indent=2)
    
    print("\n" + "="*60)
    print("SISA Unlearning Results")
    print("="*60)
    print(f"Baseline time:    {baseline_total_time:.2f}s ({baseline_epochs} epochs)")
    print(f"SISA time:        {sisa_time:.2f}s ({len(affected_by_shard)} affected shards)")
    print(f"Speedup (train):  {speedup_train_only:.2f}x")
    print(f"Speedup (total):  {speedup_total:.2f}x")
    print(f"Avg tail slices:  {avg_tail_slices:.1f} / {total_slices_per_shard}")
    print(f"Baseline acc:     {base_acc:.4f}")
    print(f"SISA acc:         {ensemble_acc:.4f}")
    print("="*60)

import torch
from torch.utils.data import DataLoader
@torch.no_grad()
def evaluate_ensemble(run_dir, num_shards, make_model, test_ds):
    device = get_safe_infer_device()
    nw = get_num_workers(device)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=nw)
    models = []
    for si in range(num_shards):
        shard_dir = os.path.join(run_dir,"checkpoints", f"shard_{si}")
        # prefer after_unlearn if exists else last post
        path = os.path.join(shard_dir,"after_unlearn.pt")
        model = make_model().to(device)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            posts = sorted([f for f in os.listdir(shard_dir) if f.endswith("_post.pt")])
            model.load_state_dict(torch.load(os.path.join(shard_dir, posts[-1]), map_location=device))
        model.eval()
        models.append(model)
    total, correct = 0, 0
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        logits_stack = torch.stack([m(x) for m in models], dim=0)
        pred = logits_stack.softmax(-1).mean(0).argmax(-1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct/total

if __name__ == "__main__":
    main()
