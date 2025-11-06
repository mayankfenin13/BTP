import torch, time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from common.device import get_device, get_num_workers

def train_classifier(model, train_ds, test_ds, epochs=5, batch_size=128, lr=1e-3, device=None):
    if device is None:
        device = get_device()
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    nw = get_num_workers(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=nw)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
    train_time = time.time() - t0
    acc = evaluate(model, test_loader, device)
    return model, acc, train_time

@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    total, correct = 0, 0
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct/total
