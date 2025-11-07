import os
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, List, Dict

def get_dataset(name: str, train: bool=True, data_dir: str="./data"):
    name = name.lower()
    if name == "mnist":
        tfm = transforms.Compose([transforms.ToTensor()])
        ds = datasets.MNIST(root=data_dir, train=train, download=True, transform=tfm)
        num_classes = 10
    elif name == "cifar10":
        tfm = transforms.Compose([transforms.ToTensor()])
        ds = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfm)
        num_classes = 10

    elif name == "fashionmnist":
        tfm = transforms.Compose([transforms.ToTensor()])
        ds = datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=tfm)
        num_classes = 10
    elif name == "svhn":
        split = "train" if train else "test"
        tfm = transforms.Compose([transforms.ToTensor()])
        ds = datasets.SVHN(root=data_dir, split=split, download=True, transform=tfm)
        num_classes = 10
    elif name == "purchase":
        # Placeholder: expects a CSV dropped locally with shape [N, d] features and last column as label.
        # Implement your own loader here and return a torch Dataset.
        raise NotImplementedError("Provide local PURCHASE100 loader in common/data.py")
    else:
        raise ValueError(f"Unknown dataset {name}")
    return ds, num_classes

def split_indices_by_class(ds) -> Dict[int, List[int]]:
    # Works for torchvision datasets returning (image, label)
    targets = ds.targets if hasattr(ds, "targets") else ds.labels
    if torch.is_tensor(targets):
        targets = targets.tolist()
    class_to_idx = {c: [] for c in set(targets)}
    for i, y in enumerate(targets):
        class_to_idx[int(y)].append(i)
    return class_to_idx

def random_shards(indices: List[int], num_shards: int, rng: np.random.Generator) -> List[List[int]]:
    idx = np.array(indices)
    rng.shuffle(idx)
    shards = np.array_split(idx, num_shards)
    return [s.tolist() for s in shards]


def cap_per_class(indices_by_class: Dict[int, list], limit: int) -> Dict[int, list]:
    """Cap indices per class using random sampling (not first N)"""
    import random
    capped = {}
    for c, idxs in indices_by_class.items():
        if limit is None or limit <= 0:
            capped[c] = list(idxs)
        else:
            idxs_list = list(idxs)
            if len(idxs_list) <= limit:
                capped[c] = idxs_list
            else:
                capped[c] = random.sample(idxs_list, limit)
    return capped

def merge_class_indices(indices_by_class: Dict[int, list]) -> list:
    merged = []
    for c in sorted(indices_by_class.keys()):
        merged.extend(indices_by_class[c])
    return merged
