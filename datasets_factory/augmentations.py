"""Small audio augmentations used by the GSC data factory."""

from __future__ import annotations

import random
from typing import Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PadOrTruncate:
    def __init__(self, audio_length: int):
        self.audio_length = audio_length

    def __call__(self, sample: Tensor) -> Tensor:
        if sample.size(-1) <= self.audio_length:
            return F.pad(sample, (0, self.audio_length - sample.size(-1)))
        return sample[..., : self.audio_length]


class RandomRoll:
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, sample: Tensor) -> Tensor:
        shifts = [torch.randint(-sample.size(dim), sample.size(dim), size=(1,)).item() for dim in self.dims]
        return torch.roll(sample, shifts, self.dims)


class SpeedPerturbation(nn.Module):
    """Nearest-neighbor speed perturbation followed by pad/crop to original length."""

    def __init__(self, rates: Tuple[float, float] = (0.9, 1.1), target_length: Union[int, str] = "same", dim: int = -1, p: float = 1.0):
        super().__init__()
        self.rates = rates
        self.target_length = target_length
        self.dim = dim
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p < 1.0 and random.random() > self.p:
            return x
        length = x.shape[self.dim]
        rate = self.rates[0] if self.rates[0] == self.rates[1] else torch.empty(()).uniform_(*self.rates).item()
        indices = torch.arange(0, length, 1.0 / rate, device=x.device).round().long().clamp(max=length - 1)
        slices = [slice(None)] * x.ndim
        slices[self.dim] = indices
        out = x[tuple(slices)].contiguous()
        target = length if self.target_length == "same" else int(self.target_length)
        if out.shape[self.dim] < target:
            pad = [0 for _ in range(out.ndim * 2)]
            pad_index = out.ndim - (self.dim % out.ndim) - 1
            pad[pad_index * 2 + 1] = target - out.shape[self.dim]
            out = F.pad(out, pad)
        if out.shape[self.dim] > target:
            slices = [slice(None)] * out.ndim
            slices[self.dim] = slice(0, target)
            out = out[tuple(slices)].contiguous()
        return out
