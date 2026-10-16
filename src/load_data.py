"""Load BCI Competition IV dataset 2a (Graz) from MATLAB .mat files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.io import loadmat

try:
    from .preprocess import bandpass_continuous
except ImportError:
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from preprocess import bandpass_continuous


def project_root() -> Path:
    """Return the repository root (parent of `src/`)."""
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Return the `data/` directory path."""
    return project_root() / "data"


def _resolve_mat_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = data_dir() / path
    return path


def list_mat_files(
    kind: Literal["training", "evaluation", "all"] = "training",
) -> list[Path]:
    """
    Discover Graz 2a ``.mat`` files in ``data/``.

    Parameters
    ----------
    kind
        ``training`` → ``A01T.mat`` … ``A09T.mat``; ``evaluation`` → ``A*E.mat``;
        ``all`` → every ``*.mat``.
    """
    d = data_dir()
    if kind == "training":
        files = sorted(d.glob("A[0-9][0-9]T.mat"))
    elif kind == "evaluation":
        files = sorted(d.glob("A[0-9][0-9]E.mat"))
    else:
        files = sorted(d.glob("*.mat"))
    return [p for p in files if p.is_file()]


def _load_one_mat_file(
    mat_path: Path,
    *,
    window_seconds: float,
    bandpass_hz: tuple[float, float],
    apply_bandpass: bool,
) -> tuple[list[np.ndarray], list[int], int, int | None, int | None]:
    """Return trial list, label list, sessions_used, fs_ref, n_chans_ref."""
    raw = loadmat(mat_path)
    if "data" not in raw:
        raise ValueError(f"No 'data' field in {mat_path.name}; not a standard Graz 2a file.")

    block = raw["data"]
    trials_x: list[np.ndarray] = []
    trials_y: list[int] = []
    fs_ref: int | None = None
    n_chans_ref: int | None = None
    sessions_used = 0

    for si in range(block.shape[1]):
        sess = block[0, si]
        X = np.asarray(sess["X"][0, 0], dtype=np.float64)
        y = np.asarray(sess["y"][0, 0])
        trial = np.asarray(sess["trial"][0, 0])
        fs = int(np.squeeze(sess["fs"][0, 0]))

        if y.size == 0 or trial.size == 0:
            continue

        if fs_ref is None:
            fs_ref = fs
        elif fs_ref != fs:
            raise ValueError(f"Inconsistent sampling rate in {mat_path.name}: {fs_ref} vs {fs}")

        if n_chans_ref is None:
            n_chans_ref = X.shape[1]
        elif X.shape[1] != n_chans_ref:
            raise ValueError(f"Inconsistent channel count in {mat_path.name}.")

        n_samples = int(round(window_seconds * fs))
        if apply_bandpass:
            X = bandpass_continuous(X, float(fs), bandpass_hz[0], bandpass_hz[1])

        n_trials = int(np.asarray(trial).shape[0])
        for i in range(n_trials):
            start = int(np.ravel(trial[i])[0])
            end = start + n_samples
            if end > X.shape[0]:
                continue
            segment = X[start:end, :].T.copy()
            lab = int(np.ravel(y[i])[0])
            if lab == 1:
                trials_x.append(segment)
                trials_y.append(0)
            elif lab == 2:
                trials_x.append(segment)
                trials_y.append(1)
            else:
                continue
        sessions_used += 1

    return trials_x, trials_y, sessions_used, fs_ref, n_chans_ref


def load_bcic2a_epochs(
    path: str | Path | list[str | Path] | None = None,
    *,
    window_seconds: float = 4.0,
    bandpass_hz: tuple[float, float] = (8.0, 30.0),
    apply_bandpass: bool = True,
    use_all_training: bool = False,
    use_all_mat: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load Graz 2a trials for **left vs right hand** motor imagery only.

    Graz labels 1 = left, 2 = right — feet/tongue are skipped. Output: **0 = left**, **1 = right**.

    Parameters
    ----------
    path
        Single file, list of files under ``data/``, or ``None`` (see ``use_all_*``).
    use_all_training
        If True, load every ``A01T.mat`` … ``A09T.mat`` found in ``data/`` (ignores ``path``).
    use_all_mat
        If True, load every ``*.mat`` in ``data/`` (ignores ``path``). Mutually exclusive
        with ``use_all_training`` in practice; ``use_all_mat`` wins if both are True.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_samples)
    y : ndarray, shape (n_trials,), int64 in ``{0, 1}``
    meta
        ``paths`` (list of files used), ``n_files``, ``n_classes``, ``task``, etc.
    """
    if use_all_mat:
        paths = list_mat_files("all")
    elif use_all_training:
        paths = list_mat_files("training")
    elif path is None:
        paths = list_mat_files("training")
    elif isinstance(path, (list, tuple)):
        paths = [_resolve_mat_path(p) for p in path]
    else:
        paths = [_resolve_mat_path(path)]

    if not paths:
        raise FileNotFoundError(
            f"No .mat files found in {data_dir()}. Add A01T.mat … A09T.mat (or use --mat-file)."
        )

    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    total_sessions = 0
    fs_ref: int | None = None
    n_chans_ref: int | None = None
    used_paths: list[str] = []

    for mat_path in paths:
        if not mat_path.is_file():
            raise FileNotFoundError(f"Missing file: {mat_path}")

        trials_x, trials_y, sessions_used, fs_i, ch_i = _load_one_mat_file(
            mat_path,
            window_seconds=window_seconds,
            bandpass_hz=bandpass_hz,
            apply_bandpass=apply_bandpass,
        )
        if not trials_x:
            continue

        if fs_ref is None:
            fs_ref = fs_i
        elif fs_i is not None and fs_ref != fs_i:
            raise ValueError(f"Sampling rate mismatch across files: {fs_ref} vs {fs_i}")

        if n_chans_ref is None:
            n_chans_ref = ch_i
        elif ch_i is not None and n_chans_ref != ch_i:
            raise ValueError("Channel count mismatch across files.")

        all_x.extend(trials_x)
        all_y.extend(trials_y)
        total_sessions += sessions_used
        used_paths.append(str(mat_path))

    if not all_x:
        raise ValueError(
            "No left/right trials extracted from the selected file(s); "
            "check paths, window_seconds, and that labels 1 and 2 exist."
        )

    X_arr = np.stack(all_x, axis=0)
    y_arr = np.asarray(all_y, dtype=np.int64)

    meta: dict[str, Any] = {
        "fs": fs_ref,
        "n_samples": X_arr.shape[2],
        "n_channels": X_arr.shape[1],
        "n_classes": 2,
        "task": "left_right",
        "paths": used_paths,
        "n_files": len(used_paths),
        "path": used_paths[0] if len(used_paths) == 1 else None,
        "n_sessions_used": total_sessions,
        "window_seconds": window_seconds,
        "bandpass_hz": bandpass_hz,
    }
    return X_arr, y_arr, meta


def load_dataset(path: str | Path | list[str | Path] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load Graz 2a ``*.mat`` data (default: all training subjects ``A*T``)."""
    return load_bcic2a_epochs(path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        X, y, meta = load_bcic2a_epochs(sys.argv[1])
    else:
        X, y, meta = load_bcic2a_epochs(None)
    print(meta)
    print("X", X.shape, "y", y.shape, "labels unique", np.unique(y))
