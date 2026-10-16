"""Preprocessing: band-pass filtering and dataset splits."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split


def bandpass_continuous(
    x: np.ndarray,
    sfreq: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:

    if x.ndim != 2:
        raise ValueError("Expected continuous EEG with shape (n_samples, n_channels).")
    nyq = sfreq / 2.0
    if not (0 < low_hz < high_hz < nyq):
        raise ValueError(f"Invalid band [{low_hz}, {high_hz}] Hz for sfreq={sfreq}.")
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return filtfilt(b, a, x.astype(np.float64, copy=False), axis=0)


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, ...]:
    """
    Split ``(X, y)`` into train / val, or train / val / test when ``test_ratio > 0``.

    Uses stratified splits when each class has at least two samples in the split.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of trials.")

    strat = y if _can_stratify(y) else None

    if test_ratio and test_ratio > 0:
        X_tv, X_test, y_tv, y_test = _tts(X, y, test_ratio, seed, strat)
        strat2 = y_tv if _can_stratify(y_tv) else None
        val_rel = val_ratio / (1.0 - test_ratio)
        X_train, X_val, y_train, y_val = _tts(X_tv, y_tv, val_rel, seed + 1, strat2)
        return X_train, X_val, X_test, y_train, y_val, y_test

    return _tts(X, y, val_ratio, seed, strat)


def _tts(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
    strat: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=strat,
        )
    except ValueError:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )


def _can_stratify(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return bool(np.all(counts >= 2))


def preprocess(raw: Any, train: bool = True) -> Any:
    """Placeholder hook for future fitted preprocessors (e.g. EA)."""
    raise NotImplementedError("Use load_bcic2a_epochs + bandpass for the current pipeline.")
