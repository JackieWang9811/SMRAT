"""Sequential and permuted MNIST data loaders."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class PermutedMNIST(Dataset):
    """Wraps MNIST and applies a fixed pixel permutation to each sequence."""

    def __init__(self, mnist: Dataset, permutation: torch.Tensor):
        self.mnist = mnist
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, index: int):
        x, y = self.mnist[index]
        x = x.view(-1)[self.permutation]
        return x, y


def build_mnist_sequence_dataloaders(
    root: str,
    batch_size: int,
    permuted: bool = False,
    seed: int = 0,
    num_workers: int = 4,
):
    transform = transforms.Compose([transforms.ToTensor()])
    development_base = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_base = datasets.MNIST(root, train=False, download=True, transform=transform)
    split_generator = torch.Generator().manual_seed(seed)
    train_base, val_base = random_split(
        development_base, [50_000, 10_000], generator=split_generator
    )

    if permuted:
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(28 * 28, generator=generator)
        train_set = PermutedMNIST(train_base, permutation)
        val_set = PermutedMNIST(val_base, permutation)
        test_set = PermutedMNIST(test_base, permutation)
    else:
        train_set = train_base
        val_set = val_base
        test_set = test_base

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
