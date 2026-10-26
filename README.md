# SMRAT: partial code snapshot

This repository is a public snapshot of the SMRAT code and data supplement prepared for manuscript review. It contains model definitions, benchmark configurations, data builders, and training entry points. **The exact `SCATR.forward` implementation is intentionally omitted in both model variants. The training scripts therefore stop at `NotImplementedError`, and this snapshot cannot reproduce the paper's numerical results.** Data files and checkpoints are also not included.

Run commands from this directory so that imports resolve consistently after the missing implementation is released.

## Code organization

- `models/smrat.py`: the original non-DVS SMRAT implementation used by the
  speech, sequential, and physiological-signal experiments.
- `models/smrat_dvs.py`: the original DVS SMRAT implementation, including its
  SPS encoder and original tensor path.
- `module/conv.py`: the three convolution/transpose wrappers required by the
  original non-DVS model.
- `datasets_factory/`: benchmark loading, deterministic splits, temporal
  normalization, and DVS frame integration.
- `<benchmark>/config_smrat.py`: the reported optimizer and model scale.
- `<benchmark>/train_smrat.py`: training, validation-based checkpointing, and
  metric computation.

The released configurations use the paper's defining settings:

- compact temporal kernel `K=2`;
- `M=ceil(log2((T-1)/(K_ref-1)+1))`, which is `ceil(log2(T))` for `K_ref=2`;
- `d_m^(l)=2^(m-1)+1+l*Delta` with `Delta=2`;
- `K_a=max(1, round(rho*N))` with `rho=0.5`;
- SpikingJelly `LIFNode` with `tau=2.0` and threshold `1.0`;
- ATan surrogate (`alpha=5`) except DVS, which uses Sigmoid (`alpha=4`);
- AdamW with warmup/cosine scheduling;
- seeds `{312, 42, 3112, 0, 10086}` for multi-seed experiments.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`SMRAT_BACKEND` defaults to `cupy`, matching the reported GPU experiments.
Set `SMRAT_BACKEND=torch` only when a CPU-compatible SpikingJelly backend is
needed for inspection.

## Data

Set `SMRAT_DATA_ROOT` to the root expected by the selected configuration, or
edit only that configuration's `datasets_path`. Data are not included.
`BENCHMARKS.md` lists the expected files and public dataset builders.

Temporal protocols are fixed in code: speech uses `T=100`, EEG patches use
`T=200`, PTB-XL uses `T=1000`, DVS uses `T=16`, and sequential MNIST uses all
`784` pixels. The MNIST loader uses 50k/10k from the official 60k training set
for train/validation and retains the official 10k test set. This avoids using
the test set for checkpoint selection.

## Training entry points (not yet executable)

These commands are provided for orientation; they require the omitted SCATR forward implementation. Examples:

```bash
SMRAT_DATA_ROOT=/path/to/SHD python SHD/train_smrat.py
SMRAT_DATA_ROOT=/path/to/SSC python SSC/train_smrat.py
SMRAT_DATA_ROOT=/path/to/GSC python GSC/train_smrat.py
SMRAT_DATA_ROOT=/path/to/TIMIT python TIMIT/train_smrat.py
SMRAT_DATA_ROOT=/path/to/ISRUC python ISRUC/train_smrat.py
SMRAT_DATA_ROOT=/path/to/SEED-V/processed_npy python SEEDV/train_smrat.py
SMRAT_DATA_ROOT=/path/to/BCIC2020/processed_npy python BCIC2020/train_smrat.py
SMRAT_DATA_ROOT=/path/to/CHB-MIT/processed_npy python CHBMIT/train_smrat.py
SMRAT_DATA_ROOT=/path/to/PTBXL python PTBXL/train_smrat.py
SMRAT_DATA_ROOT=/path/to/DVS128 python DVS128/train_smrat.py
SMRAT_DATA_ROOT=/path/to/CIFAR10DVS python CIFAR10DVS/train_smrat.py
SMRAT_DATA_ROOT=/path/to/MNIST python sMNIST/train_smrat.py
SMRAT_DATA_ROOT=/path/to/MNIST python psMNIST/train_smrat.py
```

Checkpoints, logs, and config snapshots are written beneath the current
directory. Test metrics are associated with the checkpoint selected by
validation performance; 

## Review-time implementation notice

This package provides the model architecture, module initialization, configurations, data pipelines, and
training interfaces. The exact forward implementation of SCATR is intentionally
omitted from the anonymous review-time release; consequently, this package is
not sufficient to exactly reproduce the reported numerical results. The authors
intend to release the complete implementation with the camera-ready version if
the paper is accepted.
