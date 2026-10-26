# SMRAT Benchmark Main Files

| Dataset | Training file | Config file | Model entry | Data factory entry |
| --- | --- | --- | --- | --- |
| SHD | `SHD/train_smrat.py` | `SHD/config_smrat.py` | `models/smrat.py` | `SHD_dataloaders` |
| SSC | `SSC/train_smrat.py` | `SSC/config_smrat.py` | `models/smrat.py` | `SSC_dataloaders` |
| GSC | `GSC/train_smrat.py` | `GSC/config_smrat.py` | `models/smrat.py` | `GSC_dataloaders` |
| TIMIT | `TIMIT/train_smrat.py` | `TIMIT/config_smrat.py` | `models/smrat.py` | `build_timit_dataloaders` |
| ISRUC | `ISRUC/train_smrat.py` | `ISRUC/config_smrat.py` | `models/smrat.py` | `LoadDataset` |
| SEED-V | `SEEDV/train_smrat.py` | `SEEDV/config_smrat.py` | `models/smrat.py` | `build_seedv_dataloaders` |
| BCIC2020-3 | `BCIC2020/train_smrat.py` | `BCIC2020/config_smrat.py` | `models/smrat.py` | `build_bcic2020_dataloaders` |
| CHB-MIT | `CHBMIT/train_smrat.py` | `CHBMIT/config_smrat.py` | `models/smrat.py` | `build_chbmit_dataloaders` |
| PTB-XL | `PTBXL/train_smrat.py` | `PTBXL/config_smrat.py` | `models/smrat.py` | `build_ptbxl_dataloaders` |
| DVS128 Gesture | `DVS128/train_smrat.py` | `DVS128/config_smrat.py` | `models/smrat_dvs.py` | `build_dvs128_dataloaders` |
| CIFAR10-DVS | `CIFAR10DVS/train_smrat.py` | `CIFAR10DVS/config_smrat.py` | `models/smrat_dvs.py` | `build_cifar10dvs_dataloaders` |
| sMNIST | `sMNIST/train_smrat.py` | `sMNIST/config_smrat.py` | `models/smrat.py` | `build_mnist_sequence_dataloaders` |
| psMNIST | `psMNIST/train_smrat.py` | `psMNIST/config_smrat.py` | `models/smrat.py` | `build_mnist_sequence_dataloaders` |

## Temporal protocol

| Group | Time steps | Resolution levels |
| --- | ---: | ---: |
| SHD, SSC, GSC, TIMIT | 100 | 7 |
| ISRUC, SEED-V, BCIC2020-3, CHB-MIT | 200 | 8 |
| PTB-XL | 1000 | 10 |
| DVS128 Gesture, CIFAR10-DVS | 16 | 4 |
| sMNIST, psMNIST | 784 | 10 |

## Expected local data

- SHD, SSC, DVS128 Gesture, CIFAR10-DVS, GSC, and MNIST are constructed by
  their public TorchVision/SpikingJelly dataset classes and may download their
  public archives when supported.
- TIMIT expects `train_f100_t20.npy`, `valid_f100_t20.npy`, and
  `test_f100_t20.npy`, each containing 39 input features followed by one-hot
  phoneme targets.
- BCIC2020-3 and SEED-V expect `eeg_{train,val,test}.npy` and
  `labels_{train,val,test}.npy`.
- CHB-MIT expects `X_{train,val,test}.npy` and `y_{train,val,test}.npy`.
- ISRUC expects `seq/ISRUC-group1-<subject>/` and the matching
  `labels/ISRUC-group1-<subject>/` directories. Subjects 1-80/81-90/91-100
  form train/validation/test splits.
- PTB-XL expects the standard WFDB records, `ptbxl_database.csv`, and
  `scp_statements.csv`.

The biomedical NumPy files are the benchmark-specific preprocessing outputs;
raw clinical recordings are not redistributed in this anonymous package.

