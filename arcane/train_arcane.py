import argparse, os, json, time, numpy as np, torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch.utils.data import Subset, DataLoader
from common.data import get_dataset, split_indices_by_class, cap_per_class, merge_class_indices
from common.models import LeNet, small_cnn_cifar
from common.device import get_device, get_num_workers
import torch.nn as nn
import torch.optim as optim

def make_model(dataset, num_classes):
    if dataset in ["mnist", "fashionmnist"]:
        return LeNet(num_classes=num_classes)
    else:
        return small_cnn_cifar(num_classes=num_classes)

def train_supervised(model, train_subset, test_ds, epochs=5, batch_size=128, lr=1e-3, device=None):
    if device is None:
        device = get_device()
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    nw = get_num_workers(device)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=nw)
    for _ in range(epochs):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
    return model

@torch.no_grad()
def evaluate(model, test_ds):
    device = get_device()
    model = model.to(device).eval()
    nw = get_num_workers(device)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=nw)
    total, correct = 0, 0
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist","fashionmnist","cifar10","svhn"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--blocks", type=int, default=5, help="number of training blocks (save pre/post state)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    train_ds, num_classes = get_dataset(args.dataset, train=True)
    test_ds, _ = get_dataset(args.dataset, train=False)

    # Build capped, shuffled training index list
    byc = split_indices_by_class(train_ds)
    if args.limit_per_class is not None:
        byc = cap_per_class(byc, args.limit_per_class)
    all_idx = merge_class_indices(byc)
    all_idx = rng.permutation(np.array(all_idx)).tolist()
    blocks = np.array_split(np.array(all_idx), args.blocks)

    json.dump({
        "dataset": args.dataset,
        "epochs": args.epochs,
        "blocks": args.blocks,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed
    }, open(os.path.join(args.out,"meta.json"),"w"), indent=2)
    os.makedirs(os.path.join(args.out,"indices"), exist_ok=True)
    json.dump([b.tolist() for b in blocks], open(os.path.join(args.out,"indices","blocks.json"),"w"))

    os.makedirs(os.path.join(args.out,"checkpoints"), exist_ok=True)
    model = make_model(args.dataset, num_classes)
    used = []
    for bi, blk in enumerate(blocks):
        # save pre state
        torch.save(model.state_dict(), os.path.join(args.out, "checkpoints", f"block_{bi}_pre.pt"))
        used += blk.tolist()
        model = train_supervised(model, Subset(train_ds, used), test_ds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        torch.save(model.state_dict(), os.path.join(args.out, "checkpoints", f"block_{bi}_post.pt"))

    acc = evaluate(model, test_ds)
    json.dump({"arcane_acc": float(acc)}, open(os.path.join(args.out,"summary.json"),"w"), indent=2)
    print(f"ARCANE accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
