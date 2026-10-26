"""DVS dataset factories for SMRAT vision benchmarks."""

from __future__ import annotations

import torch


def _class_balanced_split(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    label_to_indices = [[] for _ in range(num_classes)]
    for index, item in enumerate(origin_dataset):
        label_to_indices[item[1]].append(index)

    train_indices, test_indices = [], []
    for indices in label_to_indices:
        if random_split:
            permuted = torch.randperm(len(indices)).tolist()
            indices = [indices[i] for i in permuted]
        split = int(train_ratio * len(indices))
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])

    return torch.utils.data.Subset(origin_dataset, train_indices), torch.utils.data.Subset(origin_dataset, test_indices)


def _samplers(dataset_train, dataset_test, distributed: bool):
    if distributed:
        return (
            torch.utils.data.distributed.DistributedSampler(dataset_train),
            torch.utils.data.distributed.DistributedSampler(dataset_test),
        )
    return torch.utils.data.RandomSampler(dataset_train), torch.utils.data.SequentialSampler(dataset_test)


def build_dvs128_dataloaders(dataset_dir: str, batch_size: int, time_step: int, distributed: bool = False, num_workers: int = 4):
    from spikingjelly.datasets import dvs128_gesture

    train_set = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=True, data_type="frame", frames_number=time_step, split_by="number")
    val_set = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=False, data_type="frame", frames_number=time_step, split_by="number")
    train_sampler, val_sampler = _samplers(train_set, val_set, distributed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=not distributed, drop_last=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_sampler, val_sampler


def build_cifar10dvs_dataloaders(dataset_dir: str, batch_size: int, time_step: int, distributed: bool = False, num_workers: int = 4):
    from spikingjelly.datasets import cifar10_dvs

    origin_set = cifar10_dvs.CIFAR10DVS(root=dataset_dir, data_type="frame", frames_number=time_step, split_by="number")
    train_set, val_set = _class_balanced_split(0.9, origin_set, 10)
    train_sampler, val_sampler = _samplers(train_set, val_set, distributed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=not distributed, drop_last=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_sampler, val_sampler
