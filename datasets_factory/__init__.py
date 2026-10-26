"""Lazy public data-factory exports for the SMRAT reproducibility package."""

_EXPORTS = {
    "EEGNpyDataset": "npy_sequence",
    "CHBMITNpyDataset": "npy_sequence",
    "build_bcic2020_dataloaders": "npy_sequence",
    "build_seedv_dataloaders": "npy_sequence",
    "build_chbmit_dataloaders": "npy_sequence",
    "build_timit_dataloaders": "npy_sequence",
    "LoadDataset": "isruc",
    "build_isruc_dataloaders": "isruc",
    "PermutedMNIST": "mnist_sequence",
    "build_mnist_sequence_dataloaders": "mnist_sequence",
    "build_ptbxl_dataloaders": "ptbxl",
    "SHD_dataloaders": "speech",
    "SSC_dataloaders": "speech",
    "GSC_dataloaders": "speech",
    "SpecAugment": "speech",
    "build_dvs128_dataloaders": "vision_dvs",
    "build_cifar10dvs_dataloaders": "vision_dvs",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'datasets_factory' has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f"{__name__}.{_EXPORTS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value
