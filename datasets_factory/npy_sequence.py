"""Numpy-backed sequence datasets used by non-DVS SMRAT benchmarks."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class EEGNpyDataset(Dataset):
    """Loads an EEG-style tensor file and a matching label file."""

    def __init__(self, data_path: str, label_path: str, scale_factor: float = 100.0):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        self.scale_factor = scale_factor

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        x = torch.as_tensor(self.data[index], dtype=torch.float32) / self.scale_factor
        y = torch.as_tensor(self.labels[index], dtype=torch.long)
        return x, y


class CHBMITNpyDataset(EEGNpyDataset):
    """Alias kept for CHB-MIT scripts and paper benchmark naming."""


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _build_eeg_split_loaders(
    dataset_root: str,
    train_data: str,
    train_label: str,
    val_data: str,
    val_label: str,
    test_data: str,
    test_label: str,
    batch_size: int = 64,
    num_workers: int = 4,
    scale_factor: float = 100.0,
    dataset_cls=EEGNpyDataset,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_set = dataset_cls(os.path.join(dataset_root, train_data), os.path.join(dataset_root, train_label), scale_factor)
    val_set = dataset_cls(os.path.join(dataset_root, val_data), os.path.join(dataset_root, val_label), scale_factor)
    test_set = dataset_cls(os.path.join(dataset_root, test_data), os.path.join(dataset_root, test_label), scale_factor)
    return (
        _loader(train_set, batch_size, True, num_workers),
        _loader(val_set, batch_size, False, num_workers),
        _loader(test_set, batch_size, False, num_workers),
    )


def build_bcic2020_dataloaders(dataset_root: str, batch_size: int = 64, num_workers: int = 4, scale_factor: float = 100.0):
    return _build_eeg_split_loaders(
        dataset_root,
        "eeg_train.npy",
        "labels_train.npy",
        "eeg_val.npy",
        "labels_val.npy",
        "eeg_test.npy",
        "labels_test.npy",
        batch_size,
        num_workers,
        scale_factor,
    )


def build_seedv_dataloaders(dataset_root: str, batch_size: int = 64, num_workers: int = 4, scale_factor: float = 100.0):
    return build_bcic2020_dataloaders(dataset_root, batch_size, num_workers, scale_factor)


def build_chbmit_dataloaders(dataset_root: str, batch_size: int = 64, num_workers: int = 4, scale_factor: float = 100.0):
    return _build_eeg_split_loaders(
        dataset_root,
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
        batch_size,
        num_workers,
        scale_factor,
        dataset_cls=CHBMITNpyDataset,
    )


def _tensor_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    labels = np.argmax(y, axis=-1) if y.ndim > 1 else y
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _minmax_normalize(data: np.ndarray, vmax: np.ndarray, vmin: np.ndarray) -> np.ndarray:
    return (data - vmin) / (vmax - vmin + 1e-8)


def build_timit_dataloaders(dataset_root: str, time_step: int, batch_size: int = 200, num_channels: int = 39):
    """Build TIMIT loaders from preprocessed ``*_f100_t{time_step}.npy`` files."""

    train = np.load(os.path.join(dataset_root, f"train_f100_t{time_step}.npy"))
    valid = np.load(os.path.join(dataset_root, f"valid_f100_t{time_step}.npy"))
    test = np.load(os.path.join(dataset_root, f"test_f100_t{time_step}.npy"))

    vmax = np.max(train[:, :, :num_channels], axis=(0, 1))
    vmin = np.min(train[:, :, :num_channels], axis=(0, 1))
    train_x = _minmax_normalize(train[:, :, :num_channels], vmax, vmin)
    valid_x = _minmax_normalize(valid[:, :, :num_channels], vmax, vmin)
    test_x = _minmax_normalize(test[:, :, :num_channels], vmax, vmin)
    train_y = train[:, :, num_channels:]
    valid_y = valid[:, :, num_channels:]
    test_y = test[:, :, num_channels:]

    meta = {"sequence_length": train_x.shape[1], "input_dim": train_x.shape[2], "output_dim": train_y.shape[2]}
    return (
        _tensor_loader(train_x, train_y, batch_size, True),
        _tensor_loader(valid_x, valid_y, batch_size, False),
        _tensor_loader(test_x, test_y, batch_size, False),
        meta,
    )
