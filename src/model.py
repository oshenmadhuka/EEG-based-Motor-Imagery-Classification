"""EEGNet v4 (Braindecode 0.7) for motor-imagery decoding."""

from __future__ import annotations

from typing import Any

from braindecode.models import EEGNetv4


def build_model(
    n_chans: int = 25,
    n_classes: int = 2,
    input_window_samples: int = 1000,
    **kwargs: Any,
) -> EEGNetv4:

    return EEGNetv4(
        in_chans=n_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
        final_conv_length="auto",
        **kwargs,
    )


def count_parameters(model: Any) -> int:
    """Return trainable parameter count if PyTorch-like; else -1."""
    if hasattr(model, "parameters"):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return -1
