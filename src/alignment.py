"""Alignment utilities (e.g. sequence alignment, embedding alignment, cross-modal alignment)."""

from __future__ import annotations

from typing import Any


def align(a: Any, b: Any, **kwargs: Any) -> Any:
    """
    Align two representations or sequences.

    Override with your method (e.g. Procrustes, CCA, DTW, contrastive learning).

    Parameters
    ----------
    a, b : Any
        Tensors, matrices, or sequences to align.
    **kwargs
        Method-specific options.

    Returns
    -------
    Any
        Aligned pair or alignment mapping.
    """
    raise NotImplementedError("Implement align for your alignment task.")


def alignment_loss(pred: Any, target: Any, **kwargs: Any) -> Any:
    """Optional: differentiable alignment loss for end-to-end training."""
    raise NotImplementedError("Implement alignment_loss if needed.")
