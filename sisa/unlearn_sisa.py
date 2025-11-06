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
    ap.add_argument("--epochs", type=int, default=None, help="override epochs for retraining")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.run_dir,"meta.json")))
    dataset = meta["dataset"]
    epochs = args.epochs if args.epochs is not None else meta["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else meta["batch_size"]
    lr = args.lr if args.lr is not None else meta["lr"]
    shards = json.load(open(os.path.join(args.run_dir,"indices","shards.json")))
    train_ds, num_classes = get_dataset(dataset, train=True)
    test_ds,_ = get_dataset(dataset, train=False)
    # pick points to unlearn uniformly across the full training set
    all_idx = np.array([i for shard in shards for i in shard])
    n_un = max(1, int(len(all_idx)*args.unlearn_frac))
    rng = np.random.default_rng(123)
    unlearn_points = rng.choice(all_idx, size=n_un, replace=False).tolist()
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
    # Choose a safer device for training if on MPS (use CPU to avoid crashes)
    train_device = "cpu" if is_mps() else get_device()
    # Baseline: retrain from scratch without unlearned points (time it)
    baseline_t0 = time.time()
    keep = [i for i in all_idx.tolist() if i not in set(unlearn_points)]
    base_model = make_model()
    base_model, base_acc, base_time = train_classifier(base_model, Subset(train_ds, keep), test_ds, epochs=epochs, batch_size=batch_size, lr=lr, device=train_device)
    baseline_total_time = time.time() - baseline_t0
    # SISA unlearning: for each affected shard, restart from pre-slice state and continue
    affected_by_shard = {}
    for idx in unlearn_points:
        si, r = slice_maps[idx]
        affected_by_shard.setdefault(si, set()).add(r)
    un_t0 = time.time()
    for si, slice_set in affected_by_shard.items():
        shard_dir = os.path.join(args.run_dir,"checkpoints", f"shard_{si}")
        slices = json.load(open(os.path.join(args.run_dir,"indices", f"slices_shard_{si}.json")))
        # build new indices: all slices up to each affected slice need retrain from that slice's preadd
        start_r = min(slice_set)
        # load model from preadd of start_r
        model = make_model()
        pre_path = os.path.join(shard_dir, f"slice_{start_r}_preadd.pt")
        model.load_state_dict(torch.load(pre_path, map_location="cpu"))
        # Collect all remaining data from start_r onwards (excluding unlearn points)
        new_indices = []
        for r in range(start_r, len(slices)):
            sl = [i for i in slices[r] if i not in set(unlearn_points)]
            new_indices += sl
        # Train ONCE on all remaining data (not per-slice!)
        model, acc, _ = train_classifier(model, Subset(train_ds, new_indices), test_ds, epochs=epochs, batch_size=batch_size, lr=lr, device=train_device)
        # save final
        torch.save(model.state_dict(), os.path.join(shard_dir, f"after_unlearn.pt"))
    sisa_time = time.time() - un_t0
    # Evaluate ensemble after unlearning (reuse existing final models where unaffected)
    ensemble_acc = evaluate_ensemble(args.run_dir, len(shards), make_model, test_ds)
    os.makedirs(os.path.join(args.run_dir,"metrics"), exist_ok=True)
    json.dump({
        "n_unlearn": n_un,
        "baseline_total_time_s": baseline_total_time,
        "baseline_train_time_s": base_time,
        "baseline_acc": base_acc,
        "sisa_total_time_s": sisa_time,
        "ensemble_acc_after_unlearn": ensemble_acc
    }, open(os.path.join(args.run_dir,"metrics","unlearn_summary.json"),"w"), indent=2)
    print(json.dumps({"baseline_time_s": baseline_total_time, "sisa_time_s": sisa_time, "baseline_acc": base_acc, "ensemble_acc_after_unlearn": ensemble_acc}, indent=2))

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
