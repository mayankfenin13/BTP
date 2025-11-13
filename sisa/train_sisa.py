import argparse, os, json, random, time
import sys
from pathlib import Path
# Ensure project root is on sys.path for direct script execution
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from common.data import get_dataset, split_indices_by_class, random_shards, cap_per_class, merge_class_indices
from common.models import LeNet, small_cnn_cifar
from common.device import get_device, get_num_workers, get_safe_infer_device
from common.train import train_classifier, evaluate
import tempfile

# Set temp directory to output directory to avoid disk space issues
def setup_temp_dir(out_dir):
    """Set temp directory to output directory to avoid disk space issues"""
    temp_dir = os.path.join(out_dir, ".tmp")
    os.makedirs(temp_dir, exist_ok=True)
    os.environ['TMPDIR'] = temp_dir
    os.environ['TMP'] = temp_dir
    os.environ['TEMP'] = temp_dir
    return temp_dir

def save_checkpoint(state_dict, path, max_retries=3):
    """Save checkpoint with retry logic and compression"""
    for attempt in range(max_retries):
        try:
            # Use compression to save disk space
            torch.save(state_dict, path, _use_new_zipfile_serialization=False)
            return True
        except (RuntimeError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait 1 second before retry
                continue
            else:
                print(f"ERROR: Failed to save checkpoint after {max_retries} attempts: {path}")
                print(f"Error: {e}")
                raise
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist","fashionmnist","cifar10","svhn"])
    ap.add_argument("--epochs", type=int, default=5, help="Total epoch budget (will use 1 epoch per slice)")
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--slices", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # Set temp directory to avoid disk space issues
    setup_temp_dir(args.out)
    rng = np.random.default_rng(args.seed)
    train_ds, num_classes = get_dataset(args.dataset, train=True)
    test_ds, _ = get_dataset(args.dataset, train=False)
    # Optional cap per class for fast runs
    byc = split_indices_by_class(train_ds)
    byc = cap_per_class(byc, args.limit_per_class) if args.limit_per_class is not None else byc
    indices = merge_class_indices(byc)
    shards = random_shards(indices, args.shards, rng)
    # model choice
    if args.dataset in ["mnist", "fashionmnist"]:
        make_model = lambda: LeNet(num_classes=num_classes)
    else:
        make_model = lambda: small_cnn_cifar(num_classes=num_classes)
    # Train a constituent per shard with slicing checkpoints
    meta = {
        "dataset": args.dataset, 
        "epochs": args.epochs,  # This is now just metadata
        "epochs_per_slice": 1,  # FIXED: Use 1 epoch per slice
        "shards": args.shards, 
        "slices": args.slices, 
        "batch_size": args.batch_size, 
        "lr": args.lr, 
        "seed": args.seed
    }
    json.dump(meta, open(os.path.join(args.out,"meta.json"),"w"), indent=2)
    os.makedirs(os.path.join(args.out,"checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.out,"indices"), exist_ok=True)
    json.dump([list(map(int,s)) for s in shards], open(os.path.join(args.out,"indices","shards.json"),"w"))
    
    accuracies = []
    # FIXED: Use 1 epoch per slice for fair comparison
    epochs_per_slice = 1
    for si, shard_idx in enumerate(shards):
        shard_dir = os.path.join(args.out,"checkpoints", f"shard_{si}")
        os.makedirs(shard_dir, exist_ok=True)
        # slice indices
        slices = np.array_split(np.array(shard_idx), args.slices)
        json.dump([s.tolist() for s in slices], open(os.path.join(args.out,"indices", f"slices_shard_{si}.json"),"w"))
        # FIXED: incremental training with 1 epoch per slice
        current_indices = []
        model = make_model()
        for r, sl in enumerate(slices):
            current_indices += sl.tolist()
            # save state BEFORE adding next slice (slicing idea)
            preadd_path = os.path.join(shard_dir, f"slice_{r}_preadd.pt")
            save_checkpoint(model.state_dict(), preadd_path)
            train_subset = Subset(train_ds, current_indices)
            # FIXED: Train for 1 epoch per slice (not args.epochs!)
            model, acc, tr_time = train_classifier(
                model, train_subset, test_ds, 
                epochs=epochs_per_slice,  # ← FIXED: 1 epoch per slice
                batch_size=args.batch_size, 
                lr=args.lr
            )
            post_path = os.path.join(shard_dir, f"slice_{r}_post.pt")
            save_checkpoint(model.state_dict(), post_path)
        # record shard acc
        accuracies.append(acc)
    # Simple aggregation evaluation: majority vote among constituents
    agg_acc = evaluate_ensemble(args, make_model, test_ds)
    json.dump({
        "per_shard_acc": accuracies, 
        "ensemble_acc": agg_acc,
        "total_epochs_per_shard": args.slices * epochs_per_slice  # Track actual epochs used
    }, open(os.path.join(args.out,"summary.json"),"w"), indent=2)
    print(f"Ensemble accuracy: {agg_acc:.4f} (trained with {args.slices} epochs per shard)")

@torch.no_grad()
def evaluate_ensemble(args, make_model, test_ds):
    device = get_safe_infer_device()
    # Use num_workers=0 to avoid multiprocessing shared memory issues
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    all_models = []
    for si in range(args.shards):
        shard_dir = os.path.join(args.out,"checkpoints", f"shard_{si}")
        model = make_model().to(device)
        # load final post model
        # find last slice file
        posts = sorted([f for f in os.listdir(shard_dir) if f.startswith("slice_") and f.endswith("_post.pt")])
        model.load_state_dict(torch.load(os.path.join(shard_dir, posts[-1]), map_location=device))
        model.eval()
        all_models.append(model)
    total, correct = 0, 0
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        logits_stack = torch.stack([m(x) for m in all_models], dim=0)  # [S,B,C]
        pred = logits_stack.softmax(-1).mean(0).argmax(-1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct/total

if __name__ == "__main__":
    main()
