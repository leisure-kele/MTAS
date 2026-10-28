from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F


# Mirrors the codes written by exporter.build_label_arrays.
BOUNDARY_TYPE_CLASS_CHANGE = 1
BOUNDARY_TYPE_SAME_LABEL = 2


def soft_boundary_targets(
    boundary_hard: torch.Tensor,
    boundary_type: torch.Tensor,
    boundary_loss_mask: torch.Tensor,
    sigma_ticks: float,
) -> torch.Tensor:
    """Rebuild per-type Gaussian boundary targets from the cached hard markers.

    ``exporter._soft_boundary_target`` bakes one merged, smeared target into the
    sensor cache, so the smearing width can otherwise only be changed by
    re-exporting the whole corpus.  The cache also stores ``boundary_hard`` and
    ``boundary_type``, which is all that is needed to rebuild the target here.
    Two things follow: ``sigma_ticks`` becomes a training hyper-parameter, and
    the class-change and same-label families stay separable for a per-type head
    and per-type weighting.

    Bumps combine with a maximum rather than a sum, matching the exporter, so
    two human boundaries closer than the kernel never sum past 1.0.

    Returns ``[B,T,2]`` with channel 0 class-change and channel 1 same-label.
    Taking ``amax`` over the last axis reproduces the merged cached target.
    """

    if boundary_hard.ndim != 2:
        raise ValueError("boundary_hard must have shape [B,T]")
    if boundary_type.shape != boundary_hard.shape:
        raise ValueError("boundary_type must match boundary_hard shape")
    if boundary_loss_mask.shape != boundary_hard.shape:
        raise ValueError("boundary_loss_mask must match boundary_hard shape")
    if not math.isfinite(sigma_ticks) or sigma_ticks < 0:
        raise ValueError("sigma_ticks must be finite and non-negative")

    hard = boundary_hard.bool()
    is_class_change = boundary_type == BOUNDARY_TYPE_CLASS_CHANGE
    is_same_label = boundary_type == BOUNDARY_TYPE_SAME_LABEL
    if bool((hard & ~(is_class_change | is_same_label)).any()):
        # Without this the marker would be dropped from both channels and the
        # supervision would simply lose a human boundary with no visible error.
        raise ValueError(
            "Every hard boundary must carry boundary_type "
            f"{BOUNDARY_TYPE_CLASS_CHANGE} (class change) or "
            f"{BOUNDARY_TYPE_SAME_LABEL} (same label)"
        )
    typed = torch.stack(
        [hard & is_class_change, hard & is_same_label], dim=1
    ).to(torch.float32)

    if sigma_ticks > 0:
        batch_size, families, steps = typed.shape
        radius = max(1, int(math.ceil(4.0 * sigma_ticks)))
        flat = typed.reshape(batch_size * families, steps)
        # Zero padding keeps the kernel from inventing markers outside the
        # window, which is how the exporter truncates at the array bounds.
        padded = F.pad(flat, (radius, radius))
        smoothed = torch.zeros_like(flat)
        for index in range(2 * radius + 1):
            weight = math.exp(-0.5 * ((index - radius) / sigma_ticks) ** 2)
            smoothed = torch.maximum(
                smoothed, padded[:, index : index + steps] * weight
            )
        typed = smoothed.reshape(batch_size, families, steps)

    masked = typed * boundary_loss_mask.to(typed.dtype)[:, None, :]
    return masked.permute(0, 2, 1).contiguous()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    safe_target = target.masked_fill(~mask, -100)
    loss = F.cross_entropy(
        logits.transpose(1, 2),
        safe_target,
        weight=class_weights,
        ignore_index=-100,
        reduction="none",
    )
    return _masked_mean(loss, mask)


def _smoothness_loss(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    boundary_target: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    difference = log_prob[:, 1:] - log_prob[:, :-1].detach()
    squared = difference.pow(2).clamp(max=tau**2).mean(dim=-1)
    within_segment = boundary_target[:, 1:] < 0.1
    pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1] & within_segment
    return _masked_mean(squared, pair_mask)


def _boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    positive_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_weight = torch.as_tensor(positive_weight, dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight
    )
    bce = _masked_mean(bce, mask)

    probability = torch.sigmoid(logits) * mask.to(logits.dtype)
    target_masked = target * mask.to(target.dtype)
    intersection = (probability * target_masked).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (
        probability.sum() + target_masked.sum() + 1.0
    )
    return bce + dice, bce, dice


class ContactTASLoss(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.action_weight = float(config.get("action_weight", 1.0))
        self.smooth_weight = float(config.get("smooth_weight", 0.15))
        self.boundary_weight = float(config.get("boundary_weight", 1.0))
        self.subtask_weight = float(config.get("subtask_weight", 0.3))
        self.robust_weight = float(config.get("robust_weight", 0.1))
        self.boundary_positive_weight = float(
            config.get("boundary_positive_weight", 5.0)
        )
        self.smooth_tau = float(config.get("smooth_tau", 4.0))
        raw_sigma = config.get("boundary_sigma_ticks")
        self.boundary_sigma_ticks = None if raw_sigma is None else float(raw_sigma)
        if self.boundary_sigma_ticks is not None and (
            not math.isfinite(self.boundary_sigma_ticks)
            or self.boundary_sigma_ticks < 0
        ):
            raise ValueError("boundary_sigma_ticks must be finite and non-negative")
        weights = config.get("action_class_weights")
        if weights is None:
            self.register_buffer("action_class_weights", None)
        else:
            self.register_buffer(
                "action_class_weights", torch.tensor(weights, dtype=torch.float32)
            )

    def boundary_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the [B,T] target consumed by the boundary head.

        Leaving ``boundary_sigma_ticks`` unset keeps the cached exporter target
        untouched, so this stays a no-op for every existing run.  Setting it
        rebuilds the target from the cached hard markers at the requested width.
        """
        if self.boundary_sigma_ticks is None:
            return batch["boundary_target"]
        return soft_boundary_targets(
            batch["boundary_hard"],
            batch["boundary_type"],
            batch["boundary_loss_mask"],
            self.boundary_sigma_ticks,
        ).amax(dim=-1)

    def supervised(
        self, output: Dict[str, Any], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        loss_mask = batch["loss_mask"]
        boundary_mask = batch["boundary_loss_mask"]
        boundary_target = self.boundary_target(batch)
        action_terms = []
        smooth_terms = []
        boundary_terms = []
        boundary_bce_terms = []
        boundary_dice_terms = []

        for action_logits, boundary_logits in zip(
            output["atomic_stages"], output["boundary_stages"]
        ):
            action_terms.append(
                _masked_cross_entropy(
                    action_logits,
                    batch["atomic_target"],
                    loss_mask,
                    self.action_class_weights,
                )
            )
            # The smoothness term deliberately keeps the cached target: its job
            # is to skip action-logit pairs straddling a human boundary, and
            # that exclusion window should not shrink just because the boundary
            # head is asked for a sharper peak.
            smooth_terms.append(
                _smoothness_loss(
                    action_logits,
                    loss_mask,
                    batch["boundary_target"],
                    self.smooth_tau,
                )
            )
            boundary, bce, dice = _boundary_loss(
                boundary_logits,
                boundary_target,
                boundary_mask,
                self.boundary_positive_weight,
            )
            boundary_terms.append(boundary)
            boundary_bce_terms.append(bce)
            boundary_dice_terms.append(dice)

        action_loss = torch.stack(action_terms).mean()
        smooth_loss = torch.stack(smooth_terms).mean()
        boundary_loss = torch.stack(boundary_terms).mean()
        subtask_loss = action_loss.new_zeros(())
        if "subtask_logits" in output:
            subtask_mask = loss_mask & (batch["subtask_target"] >= 0)
            if bool(subtask_mask.any()):
                subtask_loss = _masked_cross_entropy(
                    output["subtask_logits"],
                    batch["subtask_target"],
                    subtask_mask,
                )

        total = (
            self.action_weight * action_loss
            + self.smooth_weight * smooth_loss
            + self.boundary_weight * boundary_loss
            + self.subtask_weight * subtask_loss
        )
        return {
            "loss": total,
            "action": action_loss,
            "smooth": smooth_loss,
            "boundary": boundary_loss,
            "boundary_bce": torch.stack(boundary_bce_terms).mean(),
            "boundary_dice": torch.stack(boundary_dice_terms).mean(),
            "subtask": subtask_loss,
        }

    def consistency(
        self,
        clean: Dict[str, Any],
        corrupted: Dict[str, Any],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        clean_probability = torch.softmax(clean["atomic_logits"].detach(), dim=-1)
        corrupt_log_probability = F.log_softmax(corrupted["atomic_logits"], dim=-1)
        kl = F.kl_div(
            corrupt_log_probability,
            clean_probability,
            reduction="none",
        ).sum(dim=-1)
        action_consistency = _masked_mean(kl, mask)

        clean_boundary = torch.sigmoid(clean["boundary_logits"].detach())
        corrupt_boundary = torch.sigmoid(corrupted["boundary_logits"])
        boundary_consistency = _masked_mean(
            (clean_boundary - corrupt_boundary).pow(2), mask
        )
        return action_consistency + boundary_consistency
