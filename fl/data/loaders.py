# fl/data/loaders.py

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, Tuple


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def get_cifar100_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def build_client_loaders(
    partition: Dict[int, np.ndarray],
    data_root: str,
    batch_size: int,
    num_workers: int = 2,
) -> Dict[int, DataLoader]:
    """
    Build a DataLoader for each client from their partition indices.
    """
    dataset = datasets.CIFAR100(
        root=data_root, train=True, download=True,
        transform=get_cifar100_transforms(train=True),
    )
    loaders = {}
    for cid, indices in partition.items():
        subset = Subset(dataset, indices.tolist())
        loaders[cid] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def build_global_test_loader(
    data_root: str,
    batch_size: int,
    num_workers: int = 2,
) -> DataLoader:
    dataset = datasets.CIFAR100(
        root=data_root, train=False, download=True,
        transform=get_cifar100_transforms(train=False),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )