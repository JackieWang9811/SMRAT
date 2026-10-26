"""Speech and spiking-speech data factories for SMRAT benchmarks."""

from __future__ import annotations

import random
from functools import partial
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from torchaudio.transforms import AmplitudeToDB, MelScale, Resample, Spectrogram
from torchvision import transforms

from . import augmentations


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fixed_length_spike_collate(batch, time_steps: int):
    """Pad/truncate SHD/SSC samples to the shared T=100 protocol."""

    from spikingjelly.datasets import pad_sequence_collate

    frames, targets, lengths = pad_sequence_collate(batch)
    frames = frames[:, :time_steps]
    if frames.shape[1] < time_steps:
        frames = F.pad(frames, (0, 0, 0, time_steps - frames.shape[1]))
    lengths = torch.as_tensor(lengths).clamp(max=time_steps)
    return frames, targets, lengths


class TimeNeuronsMaskAug:
    """Time-bin and neuron-channel masking used for SHD/SSC augmentation."""

    def __init__(self, config):
        self.config = config

    def __call__(self, x, y):
        if np.random.uniform() < self.config.TN_mask_aug_proba:
            mask_size = int(self.config.time_mask_proportion * x.shape[0])
            index = np.random.randint(0, max(x.shape[0] - mask_size, 1))
            x[index : index + mask_size, :] = 0
        if np.random.uniform() < self.config.TN_mask_aug_proba:
            mask_size = np.random.randint(0, self.config.neuron_mask_size)
            index = np.random.randint(0, max(x.shape[1] - self.config.neuron_mask_size, 1))
            x[:, index : index + mask_size] = 0
        return x, y


class Augs:
    def __init__(self, config):
        self.augs = [TimeNeuronsMaskAug(config)]

    def __call__(self, x, y):
        for aug in self.augs:
            x, y = aug(x, y)
        return x, y

    def list_augs(self):
        return [aug.__class__.__name__ for aug in self.augs]


class BinnedSpikingHeidelbergDigits:
    """Lazy wrapper around SpikingJelly SHD with frame binning."""

    def __new__(cls, *args, **kwargs):
        from spikingjelly.datasets.shd import SpikingHeidelbergDigits

        class _BinnedSHD(SpikingHeidelbergDigits):
            def __init__(
                self,
                root: str,
                n_bins: int,
                train: bool = None,
                data_type: str = "event",
                frames_number: int = None,
                split_by: str = None,
                duration: int = None,
                custom_integrate_function: Callable = None,
                custom_integrated_frames_dir_name: str = None,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
            ):
                super().__init__(
                    root,
                    train,
                    data_type,
                    frames_number,
                    split_by,
                    duration,
                    custom_integrate_function,
                    custom_integrated_frames_dir_name,
                    transform,
                    target_transform,
                )
                self.n_bins = n_bins

            def __getitem__(self, index: int):
                if self.data_type == "event":
                    events = {"t": self.h5_file["spikes"]["times"][index], "x": self.h5_file["spikes"]["units"][index]}
                    label = self.h5_file["labels"][index]
                    if self.transform is not None:
                        events, label = self.transform(events, label)
                    return events, label
                frames = np.load(self.frames_path[index], allow_pickle=True)["frames"].astype(np.float32)
                label = self.frames_label[index]
                binned_len = frames.shape[1] // self.n_bins
                binned = np.zeros((frames.shape[0], binned_len), dtype=np.float32)
                for bin_index in range(binned_len):
                    binned[:, bin_index] = frames[:, self.n_bins * bin_index : self.n_bins * (bin_index + 1)].sum(axis=1)
                if self.transform is not None:
                    binned, label = self.transform(binned, label)
                return binned, label

        return _BinnedSHD(*args, **kwargs)


class BinnedSpikingSpeechCommands:
    """Lazy wrapper around SpikingJelly SSC with frame binning."""

    def __new__(cls, *args, **kwargs):
        from spikingjelly.datasets.shd import SpikingSpeechCommands

        class _BinnedSSC(SpikingSpeechCommands):
            def __init__(
                self,
                root: str,
                n_bins: int,
                split: str = "train",
                data_type: str = "event",
                frames_number: int = None,
                split_by: str = None,
                duration: int = None,
                custom_integrate_function: Callable = None,
                custom_integrated_frames_dir_name: str = None,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
            ):
                super().__init__(
                    root,
                    split,
                    data_type,
                    frames_number,
                    split_by,
                    duration,
                    custom_integrate_function,
                    custom_integrated_frames_dir_name,
                    transform,
                    target_transform,
                )
                self.n_bins = n_bins

            def __getitem__(self, index: int):
                if self.data_type == "event":
                    events = {"t": self.h5_file["spikes"]["times"][index], "x": self.h5_file["spikes"]["units"][index]}
                    label = self.h5_file["labels"][index]
                    if self.transform is not None:
                        events, label = self.transform(events, label)
                    return events, label
                frames = np.load(self.frames_path[index], allow_pickle=True)["frames"].astype(np.float32)
                label = self.frames_label[index]
                binned_len = frames.shape[1] // self.n_bins
                binned = np.zeros((frames.shape[0], binned_len), dtype=np.float32)
                for bin_index in range(binned_len):
                    binned[:, bin_index] = frames[:, self.n_bins * bin_index : self.n_bins * (bin_index + 1)].sum(axis=1)
                if self.transform is not None:
                    binned, label = self.transform(binned, label)
                return binned, label

        return _BinnedSSC(*args, **kwargs)


def SHD_dataloaders(config):
    set_seed(config.seed)
    transform = Augs(config) if getattr(config, "use_aug", False) else None
    train_set = BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=True, data_type="frame", duration=config.time_step, transform=transform)
    val_set = BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=False, data_type="frame", duration=config.time_step)
    collate = partial(fixed_length_spike_collate, time_steps=config.max_length)
    train_loader = DataLoader(train_set, collate_fn=collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, collate_fn=collate, batch_size=config.batch_size, num_workers=4)
    return train_loader, val_loader


def SSC_dataloaders(config):
    set_seed(config.seed)
    transform = Augs(config) if getattr(config, "use_aug", False) else None
    train_set = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split="train", data_type="frame", duration=config.time_step, transform=transform)
    val_set = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split="valid", data_type="frame", duration=config.time_step)
    test_set = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split="test", data_type="frame", duration=config.time_step)
    collate = partial(fixed_length_spike_collate, time_steps=config.max_length)
    train_loader = DataLoader(train_set, collate_fn=collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, collate_fn=collate, batch_size=config.batch_size, num_workers=4)
    test_loader = DataLoader(test_set, collate_fn=collate, batch_size=config.batch_size, num_workers=4)
    return train_loader, val_loader, test_loader


class SpecAugment(nn.Module):
    """SpecAugment for mel-spectrogram batches."""

    def __init__(self, config):
        super().__init__()
        self.mF = config.mF
        self.F = config.F
        self.mT = config.mT
        self.pS = config.pS

    def forward(self, x, x_len):
        x = x.transpose(1, 2)
        for _ in range(self.mF):
            x = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.F, iid_masks=False)(x)
        for batch_index in range(x.size(0)):
            time_mask = int(self.pS * x_len[batch_index])
            for _ in range(self.mT):
                x[batch_index : batch_index + 1, :, : x_len[batch_index]] = torchaudio.transforms.TimeMasking(time_mask_param=time_mask)(
                    x[batch_index : batch_index + 1, :, : x_len[batch_index]]
                )
        return x.transpose(1, 2)


class FixedTimeSteps(nn.Module):
    """Resize a spectrogram to the paper's fixed temporal length."""

    def __init__(self, time_steps: int):
        super().__init__()
        self.time_steps = int(time_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.time_steps:
            return x
        return F.interpolate(
            x.unsqueeze(0),
            size=(x.shape[-2], self.time_steps),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)


GSC_LABELS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward",
    "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off",
    "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two",
    "up", "visual", "wow", "yes", "zero",
]


def _target_transform(word: str) -> torch.Tensor:
    return torch.tensor(GSC_LABELS.index(word))


class GSpeechCommands(Dataset):
    def __init__(self, root: str, split_name: str, transform=None, target_transform=None, download: bool = True):
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = SPEECHCOMMANDS(root, download=download, subset=split_name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        waveform, _, label, _, _ = self.dataset[index]
        if self.transform is not None:
            waveform = self.transform(waveform).squeeze().t()
        target = self.target_transform(label) if self.target_transform is not None else label
        valid_rows = waveform.ne(-100.0).all(dim=1)
        return waveform, target, len(valid_rows)


def build_transform(config, is_train: bool):
    sample_rate = 16000
    transform_steps = [
        augmentations.PadOrTruncate(sample_rate),
        Resample(sample_rate, sample_rate // 2),
    ]
    if is_train:
        transform_steps.extend(
            [
                augmentations.RandomRoll(dims=(1,)),
                augmentations.SpeedPerturbation(rates=(0.5, 1.5), p=0.5),
            ]
        )
    transform_steps.extend(
        [
            Spectrogram(n_fft=config.window_size, hop_length=config.hop_length, power=2),
            MelScale(n_mels=config.n_mels, sample_rate=sample_rate // 2, f_min=50, f_max=14000, n_stft=config.window_size // 2 + 1),
            AmplitudeToDB(),
            FixedTimeSteps(config.max_length),
        ]
    )
    return transforms.Compose(transform_steps)


def GSC_dataloaders(config, is_transform: bool = False):
    set_seed(config.seed)
    train_set = GSpeechCommands(config.datasets_path, "training", transform=build_transform(config, is_transform), target_transform=_target_transform)
    val_set = GSpeechCommands(config.datasets_path, "validation", transform=build_transform(config, False), target_transform=_target_transform)
    test_set = GSpeechCommands(config.datasets_path, "testing", transform=build_transform(config, False), target_transform=_target_transform)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, num_workers=4)
    return train_loader, val_loader, test_loader
