"""ISRUC data factory for subject-level sleep-stage splits."""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()


class CustomDataset(Dataset):
    """Loads one preprocessed ISRUC sequence file and one label file per item."""

    def __init__(self, seq_label_pairs: Sequence[Tuple[str, str]], scale_factor: float = 100.0):
        self.seq_label_pairs = list(seq_label_pairs)
        self.scale_factor = scale_factor

    def __len__(self) -> int:
        return len(self.seq_label_pairs)

    def __getitem__(self, index: int):
        seq_path, label_path = self.seq_label_pairs[index]
        seq = np.load(seq_path) / self.scale_factor
        label = np.load(label_path).astype(np.int64)
        return seq, label

    def collate(self, batch):
        x_seq = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return _to_tensor(x_seq), _to_tensor(y_label).long()


class LoadDataset:
    """Legacy-compatible ISRUC loader used by the original training scripts."""

    def __init__(self, params):
        self.params = params
        self.seqs_dir = os.path.join(params.datasets_dir, "seq")
        self.labels_dir = os.path.join(params.datasets_dir, "labels")
        self.seq_label_groups = self.load_path()

    def get_data_loader(self):
        train_pairs, val_pairs, test_pairs = self.split_dataset(self.seq_label_groups)
        train_set = CustomDataset(train_pairs)
        val_set = CustomDataset(val_pairs)
        test_set = CustomDataset(test_pairs)
        return {
            "train": DataLoader(train_set, batch_size=self.params.batch_size, collate_fn=train_set.collate, shuffle=True, num_workers=4),
            "val": DataLoader(val_set, batch_size=1, collate_fn=val_set.collate, shuffle=False, num_workers=4),
            "test": DataLoader(test_set, batch_size=1, collate_fn=test_set.collate, shuffle=False, num_workers=4),
        }

    def load_path(self) -> List[Tuple[int, List[Tuple[str, str]]]]:
        seq_label_groups = []
        for subject_id in range(1, 101):
            seq_dir = os.path.join(self.seqs_dir, f"ISRUC-group1-{subject_id}")
            label_dir = os.path.join(self.labels_dir, f"ISRUC-group1-{subject_id}")
            if not os.path.exists(seq_dir) or not os.path.exists(label_dir):
                continue
            seq_files = sorted(os.listdir(seq_dir))
            label_files = sorted(os.listdir(label_dir))
            if len(seq_files) != len(label_files):
                raise ValueError(f"Mismatch in seq/label count for subject {subject_id}")
            pairs = [
                (os.path.join(seq_dir, seq_name), os.path.join(label_dir, label_name))
                for seq_name, label_name in zip(seq_files, label_files)
            ]
            seq_label_groups.append((subject_id, pairs))
        return seq_label_groups

    @staticmethod
    def split_dataset(seq_label_groups):
        """Subject split: 1-80 train, 81-90 validation, 91-100 test."""

        train_pairs, val_pairs, test_pairs = [], [], []
        for subject_id, subject_pairs in seq_label_groups:
            if subject_id <= 80:
                train_pairs.extend(subject_pairs)
            elif subject_id <= 90:
                val_pairs.extend(subject_pairs)
            else:
                test_pairs.extend(subject_pairs)
        return train_pairs, val_pairs, test_pairs


def build_isruc_dataloaders(config):
    params = type("ISRUCParams", (), {"datasets_dir": config.datasets_path, "batch_size": config.batch_size})()
    loaders = LoadDataset(params).get_data_loader()
    return loaders["train"], loaders["val"], loaders["test"]
