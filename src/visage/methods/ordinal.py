"""Conditional ordinal regression for VisAge's three ordered age groups.

The class order is under 18, 18--59, and 60+. The two model outputs are
unrestricted logits for P(Y > 0) and P(Y > 1 | Y > 0), respectively. Neither
these probabilities nor their maximum are calibrated confidence estimates.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

NUM_AGE_GROUPS = 3
NUM_ORDINAL_LOGITS = NUM_AGE_GROUPS - 1


def _validate_logits(logits: Tensor) -> None:
    if logits.ndim != 2 or logits.shape[1] != NUM_ORDINAL_LOGITS:
        raise ValueError("logits must have shape [batch, 2].")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating-point tensors.")


def corn_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    task_weights: Tensor | Sequence[float] | None = None,
) -> Tensor:
    """Return the observation-weighted CORN binary cross-entropy loss.

    Task zero uses every example, with target ``labels > 0``. Task one uses
    only examples for which ``labels > 0``, with target ``labels > 1``.
    Summed BCE is divided by the number of eligible conditional-task
    observations, rather than averaging the two task means. This matters
    when a minibatch contains different numbers of examples per class.

    Optional nonnegative ``task_weights`` give importance weights to the two
    conditional tasks; the denominator uses the same weights. They are not
    per-class weights or probability calibration. Empty batches, or batches
    with no eligible positive-weight task, return a differentiable zero.
    """
    _validate_logits(logits)
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch] matching logits.")
    if labels.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("labels must contain integer class indices.")
    if labels.device != logits.device:
        raise ValueError("labels and logits must be on the same device.")
    if torch.any((labels < 0) | (labels >= NUM_AGE_GROUPS)):
        raise ValueError("labels must be 0, 1, or 2.")

    if task_weights is None:
        weights = logits.new_ones(NUM_ORDINAL_LOGITS)
    else:
        weights = torch.as_tensor(
            task_weights, dtype=logits.dtype, device=logits.device
        )
        if weights.shape != (NUM_ORDINAL_LOGITS,):
            raise ValueError("task_weights must contain exactly two values.")
        if not torch.all(torch.isfinite(weights)) or torch.any(weights < 0):
            raise ValueError("task_weights must be finite and nonnegative.")
        if not torch.any(weights > 0):
            raise ValueError("At least one task weight must be positive.")

    numerator = logits.sum() * 0.0
    denominator = logits.new_zeros(())
    for task in range(NUM_ORDINAL_LOGITS):
        # Avoid comparing uint8 labels with -1: the scalar would wrap to 255.
        eligible = torch.ones_like(labels, dtype=torch.bool) if task == 0 else labels > 0
        if not torch.any(eligible):
            continue
        targets = (labels[eligible] > task).to(dtype=logits.dtype)
        task_loss = F.binary_cross_entropy_with_logits(
            logits[eligible, task], targets, reduction="sum"
        )
        numerator = numerator + weights[task] * task_loss
        denominator = denominator + weights[task] * eligible.sum()

    # A positive weight may belong only to an absent conditional task.
    safe_denominator = torch.where(
        denominator > 0, denominator, torch.ones_like(denominator)
    )
    return numerator / safe_denominator


def ordinal_probabilities(logits: Tensor) -> Tensor:
    """Convert two CORN logits into coherent probabilities for three classes."""
    _validate_logits(logits)
    conditional = torch.sigmoid(logits)
    above_first = conditional[:, 0]
    above_second = above_first * conditional[:, 1]
    return torch.stack(
        (1.0 - above_first, above_first - above_second, above_second), dim=1
    )


def predict_age_group(logits: Tensor) -> Tensor:
    """Return the most probable class index, with lower-class ties first."""
    return ordinal_probabilities(logits).argmax(dim=1)
