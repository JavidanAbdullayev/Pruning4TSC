"""UCR Archive loading and preprocessing utilities."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


def get_ucr_path() -> Path:
    """Resolve the UCR archive root from the ``UCR_ARCHIVE_DIR`` env var."""
    path = os.environ.get("UCR_ARCHIVE_DIR")
    if not path:
        raise EnvironmentError(
            "Set the UCR_ARCHIVE_DIR environment variable to the directory "
            "containing the UCR 2018 archive (with one folder per dataset)."
        )
    root = Path(path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"UCR_ARCHIVE_DIR is not a directory: {root}")
    return root


def load_ucr_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a UCR dataset by name. Returns ``(x_train, y_train, x_test, y_test)``."""
    folder = get_ucr_path() / name
    train_path = folder / f"{name}_TRAIN.tsv"
    test_path = folder / f"{name}_TEST.tsv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing TRAIN/TEST tsv files for {name} in {folder}")

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    y_train, y_test = train[:, 0], test[:, 0]
    x_train = np.delete(train, 0, axis=1)
    x_test = np.delete(test, 0, axis=1)
    return x_train, y_train, x_test, y_test


def znormalize(x: np.ndarray) -> np.ndarray:
    """Per-series z-normalization with zero-std safeguard."""
    stds = np.std(x, axis=1, keepdims=True)
    stds = np.where(stds == 0.0, 1.0, stds)
    return (x - x.mean(axis=1, keepdims=True)) / stds


def encode_labels(y: np.ndarray) -> np.ndarray:
    return LabelEncoder().fit_transform(y)


def make_dataloader(
    data: np.ndarray,
    target: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader from raw numpy arrays. Z-normalizes and adds a channel dim."""
    data = znormalize(data)
    data = np.expand_dims(data, axis=1)  # (N, 1, T)
    target = encode_labels(target)

    data_t = torch.from_numpy(data)
    target_t = torch.from_numpy(target)

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(data_t, target_t),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
