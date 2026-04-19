"""
preprocessing.py
────────────────
Feature engineering and normalization for sensor time-series data.

Includes:
  - StandardScaler / MinMaxScaler fitting and transform
  - Sliding-window extraction
  - Rolling statistics (mean, std, min, max)
  - FFT magnitude features (optional)
  - RUL piecewise-linear clipping
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


SENSOR_COLS = [
    "vibration_x", "vibration_y", "vibration_z",
    "temperature_bearing", "temperature_ambient",
    "pressure_in", "pressure_out",
    "rpm", "current_draw", "oil_viscosity",
    "acoustic_emission", "torque",
    "humidity", "voltage",
]


# ─── Scaler ──────────────────────────────────────────────────────────────────

def fit_scaler(df: pd.DataFrame, method: str = "standard") -> object:
    """Fit a scaler on sensor columns of the training set."""
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    scaler.fit(df[SENSOR_COLS])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: object) -> pd.DataFrame:
    """Apply a pre-fitted scaler in-place."""
    df = df.copy()
    df[SENSOR_COLS] = scaler.transform(df[SENSOR_COLS])
    return df


def save_scaler(scaler: object, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Rolling features ─────────────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Add per-unit rolling mean and std for each sensor column.
    Groups by unit_id so windows don't bleed across units.
    """
    df = df.copy()
    new_cols = []
    for w in windows:
        for col in SENSOR_COLS:
            roll = df.groupby("unit_id")[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            std = df.groupby("unit_id")[col].transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
            df[f"{col}_roll{w}_mean"] = roll
            df[f"{col}_roll{w}_std"] = std
            new_cols += [f"{col}_roll{w}_mean", f"{col}_roll{w}_std"]
    return df


# ─── Sliding window ───────────────────────────────────────────────────────────

def extract_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 30,
    stride: int = 1,
    rul_clip: int = 125,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract sliding-window samples from a multi-unit dataframe.

    Returns
    -------
    X      : (N, window_size, n_features)   sensor windows
    y_cls  : (N,)                           binary failure label
    y_rul  : (N,)                           clipped RUL (regression target)
    """
    X_list, y_cls_list, y_rul_list = [], [], []

    for uid, group in df.groupby("unit_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        features = group[feature_cols].values  # (T, F)
        rul = np.clip(group["RUL"].values, 0, rul_clip)
        label = group["failure_imminent"].values

        T = len(group)
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            X_list.append(features[start:end])
            y_cls_list.append(label[end - 1])     # label at the last step
            y_rul_list.append(rul[end - 1])

    if not X_list:
        raise ValueError("No windows extracted — check window_size vs dataset length.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y_cls = np.array(y_cls_list, dtype=np.float32)
    y_rul = np.array(y_rul_list, dtype=np.float32)
    return X, y_cls, y_rul


# ─── FFT features (optional) ─────────────────────────────────────────────────

def add_fft_features(X: np.ndarray, n_freqs: int = 5) -> np.ndarray:
    """
    Append dominant FFT magnitude coefficients to each window.

    Args:
        X       : (N, T, F) sensor windows
        n_freqs : number of frequency bins to keep

    Returns
    -------
    X_aug : (N, T, F + n_freqs * n_vibration_channels)
    """
    vib_indices = [0, 1, 2]  # vibration_x, y, z
    fft_features = []

    for i in vib_indices:
        channel = X[:, :, i]                         # (N, T)
        magnitudes = np.abs(np.fft.rfft(channel, axis=1))  # (N, T//2+1)
        top_freqs = magnitudes[:, 1: n_freqs + 1]    # skip DC
        fft_features.append(top_freqs)               # (N, n_freqs)

    fft_block = np.concatenate(fft_features, axis=1)  # (N, n_freqs * 3)
    fft_tiled = np.tile(fft_block[:, np.newaxis, :], (1, X.shape[1], 1))  # (N, T, n_freqs*3)
    return np.concatenate([X, fft_tiled.astype(np.float32)], axis=2)
