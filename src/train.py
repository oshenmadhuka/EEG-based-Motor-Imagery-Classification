"""Train EEGNetv4 on BCI IV 2a epoched data (baseline pipeline)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.load_data import load_bcic2a_epochs
from src.model import build_model, count_parameters
from src.preprocess import train_val_test_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_train(
    mat_files: list[str] | None = None,
    *,
    use_all_mat: bool = False,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.2,
    seed: int = 42,
    device: str | None = None,
    no_bandpass: bool = False,
) -> None:
    set_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {dev}")

    if use_all_mat:
        X, y, meta = load_bcic2a_epochs(
            use_all_mat=True,
            apply_bandpass=not no_bandpass,
        )
    elif mat_files:
        X, y, meta = load_bcic2a_epochs(
            mat_files,
            apply_bandpass=not no_bandpass,
        )
    else:
        X, y, meta = load_bcic2a_epochs(
            None,
            apply_bandpass=not no_bandpass,
        )

    n_chans = int(meta["n_channels"])
    n_times = int(meta["n_samples"])
    paths = meta.get("paths") or ([meta["path"]] if meta.get("path") else [])
    print(
        f"Loaded {meta.get('n_files', len(paths))} file(s): {paths[:5]}"
        + (" ..." if len(paths) > 5 else "")
    )
    print(
        f"  X {X.shape}, y {y.shape}, fs={meta['fs']}, "
        f"sessions={meta['n_sessions_used']}, left vs right (0/1)"
    )
    print(f"  class counts: 0=left {int((y == 0).sum())}, 1=right {int((y == 1).sum())}")

    X_train, X_val, y_train, y_val = train_val_test_split(X, y, val_ratio=val_ratio, seed=seed)

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    if X_train_t.ndim == 3:
        X_train_t = X_train_t.unsqueeze(-1)
        X_val_t = X_val_t.unsqueeze(-1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False,
    )

    n_cls = int(meta.get("n_classes", 2))
    model = build_model(
        n_chans=n_chans,
        n_classes=n_cls,
        input_window_samples=n_times,
    ).to(dev)
    print(f"EEGNetv4 parameters: {count_parameters(model):,}")

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    results_dir = _ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    best_val = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False):
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb)
                pred = out.argmax(dim=1)
                val_correct += (pred == yb).sum().item()
                val_total += yb.numel()
        val_acc = val_correct / max(val_total, 1)
        train_loss = float(np.mean(losses)) if losses else 0.0
        print(f"Epoch {ep}: train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {"model_state": model.state_dict(), "meta": meta},
                results_dir / "eegnetv4_bcic2a_best.pt",
            )

    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Checkpoint: {results_dir / 'eegnetv4_bcic2a_best.pt'}")


def train(
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> None:
    """Backward-compatible entry point used by ``if __name__`` defaults."""
    run_train(
        mat_file="A01T.mat",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline EEGNetv4 on BCI IV 2a (.mat).")
    p.add_argument(
        "--mat-file",
        nargs="*",
        default=None,
        metavar="FILE",
        help="One or more files under data/ (e.g. A01T.mat). If omitted, loads all A01T–A09T found.",
    )
    p.add_argument(
        "--all-mat",
        action="store_true",
        help="Load every *.mat in data/ (training + evaluation). Default is training-only A*T.mat.",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto).")
    p.add_argument("--no-bandpass", action="store_true", help="Skip 8–30 Hz filter (debug).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_train(
        mat_files=args.mat_file if args.mat_file else None,
        use_all_mat=args.all_mat,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        no_bandpass=args.no_bandpass,
    )
