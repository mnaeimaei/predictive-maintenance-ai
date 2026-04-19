"""
dataset.py
──────────
PyTorch Dataset and DataLoader factory for predictive maintenance data.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class SensorWindowDataset(Dataset):
    """
    Dataset of sliding-window sensor sequences.

    Args:
        X      : (N, window_size, n_features) float32 array
        y_cls  : (N,) binary failure labels
        y_rul  : (N,) remaining useful life (clipped)
    """

    def __init__(
        self,
        X: np.ndarray,
        y_cls: np.ndarray,
        y_rul: np.ndarray,
    ):
        self.X = torch.from_numpy(X)
        self.y_cls = torch.from_numpy(y_cls)
        self.y_rul = torch.from_numpy(y_rul)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y_cls[idx], self.y_rul[idx]


def build_dataloader(
    X: np.ndarray,
    y_cls: np.ndarray,
    y_rul: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = True,
    use_weighted_sampler: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader with optional WeightedRandomSampler to handle
    class imbalance (failure events are rare).
    """
    dataset = SensorWindowDataset(X, y_cls, y_rul)

    sampler = None
    if use_weighted_sampler and shuffle:
        # Compute per-sample weights inversely proportional to class frequency
        class_counts = np.bincount(y_cls.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_cls.astype(int)]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False  # mutually exclusive with sampler

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
