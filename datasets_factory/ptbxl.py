"""PTB-XL data factory used by the SMRAT ECG benchmark."""

from __future__ import annotations

import ast
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def _compute_label_aggregations(df_meta: pd.DataFrame, scp_csv_path: Path, task: str) -> pd.DataFrame:
    """Aggregate PTB-XL SCP statements according to the selected task."""

    df = df_meta.copy()
    aggregation = pd.read_csv(scp_csv_path, index_col=0)

    if task in ["diagnostic", "subdiagnostic", "superdiagnostic"]:
        diagnostic = aggregation[aggregation.diagnostic == 1.0]

        def diagnostic_codes(codes):
            return list({key for key in codes if key in diagnostic.index})

        def diagnostic_subclasses(codes):
            return list({diagnostic.loc[key].diagnostic_subclass for key in codes if key in diagnostic.index and str(diagnostic.loc[key].diagnostic_subclass) != "nan"})

        def diagnostic_superclasses(codes):
            return list({diagnostic.loc[key].diagnostic_class for key in codes if key in diagnostic.index and str(diagnostic.loc[key].diagnostic_class) != "nan"})

        if task == "diagnostic":
            df["diagnostic"] = df.scp_codes.apply(diagnostic_codes)
            df["diagnostic_len"] = df.diagnostic.apply(len)
        elif task == "subdiagnostic":
            df["subdiagnostic"] = df.scp_codes.apply(diagnostic_subclasses)
            df["subdiagnostic_len"] = df.subdiagnostic.apply(len)
        else:
            df["superdiagnostic"] = df.scp_codes.apply(diagnostic_superclasses)
            df["superdiagnostic_len"] = df.superdiagnostic.apply(len)
    elif task in ["form", "rhythm"]:
        mask = aggregation[task] == 1.0
        selected = aggregation[mask]
        df[task] = df.scp_codes.apply(lambda codes: list({key for key in codes if key in selected.index}))
        df[f"{task}_len"] = df[task].apply(len)
    elif task == "all":
        df["all_scp"] = df.scp_codes.apply(lambda codes: list(set(codes.keys())))
        df["all_scp_len"] = df.all_scp.apply(len)
    else:
        raise ValueError(f"Unknown PTB-XL task: {task}")
    return df


def _select_data(X: np.ndarray, Y_df: pd.DataFrame, task: str, min_samples: int = 0):
    """Filter sparse labels and convert label lists into a multi-hot matrix."""

    label_col = "all_scp" if task == "all" else task
    length_col = f"{label_col}_len"
    mlb = MultiLabelBinarizer()

    if min_samples > 0:
        counts = pd.Series(np.concatenate(Y_df[label_col].values)).value_counts()
        counts = counts[counts > min_samples]
        Y_df[label_col] = Y_df[label_col].apply(lambda labels: list(set(labels).intersection(set(counts.index.values))))
        Y_df[length_col] = Y_df[label_col].apply(len)

    mask = Y_df[length_col] > 0
    X_out = X[mask.values]
    Y_out = Y_df[mask]
    mlb.fit(Y_out[label_col].values)
    y = mlb.transform(Y_out[label_col].values)
    return X_out, Y_out, y, mlb


def _load_ptbxl_signals(root: Path, sampling_rate: int = 100) -> Tuple[np.ndarray, pd.DataFrame]:
    db_csv = root / "ptbxl_database.csv"
    if not db_csv.exists():
        raise FileNotFoundError(f"PTB-XL metadata was not found at {db_csv}.")

    Y = pd.read_csv(db_csv, index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda value: ast.literal_eval(value))

    if sampling_rate == 100:
        cache = root / "raw100.npy"
        file_col = "filename_lr"
    elif sampling_rate == 500:
        cache = root / "raw500.npy"
        file_col = "filename_hr"
    else:
        raise ValueError("sampling_rate must be 100 or 500")

    if cache.exists():
        try:
            X = np.load(cache, allow_pickle=True)
        except ValueError:
            with open(cache, "rb") as handle:
                X = pickle.load(handle)
    else:
        paths = Y[file_col].tolist()
        records = [wfdb.rdsamp(str(root / path)) for path in tqdm(paths, desc=f"Reading PTB-XL {sampling_rate}Hz")]
        X = np.array([signal for signal, _ in records])
        np.save(cache, X)
    return X, Y


def _standardize_global(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_train).flatten()[:, None].astype(float))

    def apply(X):
        out = []
        for item in X:
            shape = item.shape
            out.append(scaler.transform(item.flatten()[:, None]).reshape(shape))
        return np.array(out)

    return apply(X_train), apply(X_val), apply(X_test)


def build_ptbxl_dataloaders(
    data_dir: str,
    task: str = "superdiagnostic",
    sampling_rate: int = 100,
    min_samples: int = 0,
    train_fold: int = 8,
    val_fold: int = 9,
    test_fold: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Return train/validation/test loaders plus benchmark metadata."""

    root = Path(data_dir).absolute()
    X, Y = _load_ptbxl_signals(root, sampling_rate=sampling_rate)
    Y = _compute_label_aggregations(Y, root / "scp_statements.csv", task=task)
    X, Y, y, mlb = _select_data(X, Y, task=task, min_samples=min_samples)

    X = X.transpose(0, 2, 1)
    _, d_input, sequence_length = X.shape
    d_output = y.shape[1]

    X_test, y_test = X[Y.strat_fold == test_fold], y[Y.strat_fold == test_fold]
    X_val, y_val = X[Y.strat_fold == val_fold], y[Y.strat_fold == val_fold]
    X_train, y_train = X[Y.strat_fold <= train_fold], y[Y.strat_fold <= train_fold]
    X_train, X_val, X_test = _standardize_global(X_train, X_val, X_test)

    train_set = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_set = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    test_set = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    meta = {"d_input": d_input, "d_output": d_output, "L": sequence_length, "needs_rearrange": False, "classes": list(mlb.classes_)}
    return train_loader, val_loader, test_loader, meta
