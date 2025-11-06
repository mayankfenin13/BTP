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
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.run_dir,"meta.json")))
    dataset = meta["dataset"]
    epochs = args.epochs if args.epochs is not None else meta["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else meta["batch_size"]
    lr = args.lr if args.lr is not None else meta["lr"]
    train_ds, num_classes = get_dataset(dataset, train=True)
    test_ds,_ = get_dataset(dataset, train=False)
    device = get_device()
    # Use same device logic as SISA for fair comparison
    train_device = "cpu" if is_mps() else device
    blocks = json.load(open(os.path.join(args.run_dir,"indices","blocks.json")))
    # Build flat list of training indices
    all_idx = [i for b in blocks for i in b]
    n_un = max(1, int(len(all_idx)*args.unlearn_frac))
    rng = np.random.default_rng(123)
    unlearn_points = set(rng.choice(np.array(all_idx), size=n_un, replace=False).tolist())
    # Baseline retraining: rebuild model from scratch without those points
    t0 = time.time()
    keep = [i for i in all_idx if i not in unlearn_points]
    base_model = make_model(dataset, num_classes)
    base_model = train_supervised(base_model, Subset(train_ds, keep), test_ds, epochs=epochs, batch_size=batch_size, lr=lr, device=train_device)
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
        # load pre of earliest and continue on cleaned blocks
        pre_path = os.path.join(args.run_dir, "checkpoints", f"block_{earliest}_pre.pt")
        model.load_state_dict(torch.load(pre_path, map_location=device))
        # Collect all remaining data from earliest block onwards (excluding unlearn points)
        used = []
        for bi in range(earliest, len(blocks)):
            blk = [i for i in blocks[bi] if i not in unlearn_points]
            used += blk
        # Train ONCE on all remaining data (not per-block!)
        model = train_supervised(model, Subset(train_ds, used), test_ds, epochs=epochs, batch_size=batch_size, lr=lr, device=train_device)
    arcane_time = time.time() - t1
    # Evaluate both
    base_acc = evaluate(base_model, test_ds)
    arcane_acc = evaluate(model, test_ds)
    os.makedirs(os.path.join(args.run_dir,"metrics"), exist_ok=True)
    json.dump({
        "n_unlearn": int(n_un),
        "baseline_time_s": baseline_time,
        "arcane_time_s": arcane_time,
        "baseline_acc": float(base_acc),
        "arcane_acc_after_unlearn": float(arcane_acc)
    }, open(os.path.join(args.run_dir,"metrics","unlearn_summary.json"),"w"), indent=2)
    print(json.dumps({"baseline_time_s": baseline_time, "arcane_time_s": arcane_time, "baseline_acc": base_acc, "arcane_acc_after_unlearn": arcane_acc}, indent=2))

if __name__ == "__main__":
    main()
