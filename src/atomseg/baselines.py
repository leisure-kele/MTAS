from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
from torch import nn

from .model import ContactTAS, TemporalEncoder


class FixedFusion(nn.Module):
    """Globally parameterized, hard-presence-masked fusion for the M0 control.

    The two logits are learned, but they are global scalars: before applying the
    present/missing mask, their values do not depend on a sample, tick, feature,
    or continuous quality magnitude.  This keeps M0 distinct from M1's
    observable-reliability network while still allowing a fair learned global
    semantic/event balance.
    """

    def __init__(self, dimension: int, weights: Tuple[float, float] = (0.5, 0.5)):
        super().__init__()
        fixed = torch.as_tensor(weights, dtype=torch.float32)
        if fixed.shape != (2,) or bool((fixed < 0).any()) or float(fixed.sum()) <= 0:
            raise ValueError("fixed_fusion_weights must contain two non-negative values")
        normalized = fixed / fixed.sum()
        # Retain the former initializer's legal zero weight while representing
        # it as a finite, trainable logit rather than a frozen hard exclusion.
        initial_logits = torch.where(
            normalized > 0,
            normalized.log(),
            torch.full_like(normalized, -20.0),
        )
        self.fusion_logits = nn.Parameter(initial_logits)
        self.norm = nn.LayerNorm(dimension)
        self.register_buffer("null_token", torch.zeros(dimension))

    def forward(
        self,
        semantic: torch.Tensor,
        event: torch.Tensor,
        semantic_quality: torch.Tensor,
        event_quality: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        quality = torch.stack([semantic_quality, event_quality], dim=-1)
        present = torch.isfinite(quality) & (quality > 0)
        logits = self.fusion_logits.to(dtype=semantic.dtype).expand_as(quality)
        masked_logits = logits.masked_fill(~present, -1.0e4)
        gate = torch.softmax(masked_logits, dim=-1) * present.to(semantic.dtype)
        denominator = gate.sum(dim=-1, keepdim=True)
        gate = torch.where(
            denominator > 0,
            gate / denominator.clamp_min(1.0e-6),
            torch.zeros_like(gate),
        )
        fused = gate[..., 0:1] * semantic + gate[..., 1:2] * event
        fused = torch.where(
            denominator <= 0,
            self.null_token.to(dtype=fused.dtype)[None, None, :],
            fused,
        )
        return self.norm(fused), gate


class FixedContactTAS(ContactTAS):
    """M0: M1's two streams/decoder with learned global-only fusion logits."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        dimension = int(config.get("d_model", 256))
        configured = config.get("fixed_fusion_weights", [0.5, 0.5])
        weights = (float(configured[0]), float(configured[1]))
        self.fusion = FixedFusion(dimension, weights)


class EarlyFusionContactTAS(ContactTAS):
    """B2: shared modality encoders followed by one early-fusion temporal stream."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        dimension = int(config.get("d_model", 256))
        dropout = float(config.get("dropout", 0.2))

        # B2 reuses ContactTAS input encoders and decoder, but intentionally removes
        # the dual-stream and reliability modules so they are not unused DDP params.
        del self.semantic_projection
        del self.semantic_norm
        del self.event_projection
        del self.event_temporal
        del self.fusion

        self.early_projection = nn.Linear(dimension * 3, dimension)
        self.early_norm = nn.LayerNorm(dimension)
        self.early_temporal = TemporalEncoder(
            dimension,
            dimension,
            int(config.get("stream_layers", 4)),
            dropout,
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        apply_modality_dropout: bool = False,
        return_aux: bool = False,
    ) -> Dict[str, Any]:
        encoded = self._encode_modalities(batch, apply_modality_dropout)
        sequence_mask = encoded["sequence_mask"]

        vision = encoded["vision_token"] * encoded["vision_quality"][..., None]
        skeleton = encoded["skeleton_token"] * encoded["skeleton_quality"][..., None]
        tactile = encoded["tactile_token"] * encoded["tactile_quality"][..., None]
        kinematics = (
            encoded["kinematic_token"] * encoded["kinematic_quality"][..., None]
        )
        event_evidence = self._limit_event_clock(
            torch.cat([tactile, kinematics], dim=-1), sequence_mask
        )
        early = self.early_norm(
            torch.nn.functional.gelu(
                self.early_projection(
                    torch.cat([vision, skeleton, event_evidence], dim=-1)
                )
            )
        )
        fused = self.early_temporal(early, sequence_mask)
        fused = fused * sequence_mask[..., None].to(fused.dtype)
        output = self.decoder(fused, sequence_mask)
        output["reliability_weights"] = torch.stack(
            [encoded["semantic_quality"], encoded["event_quality"]], dim=-1
        )
        output["view_weights"] = encoded["view_weights"]
        if return_aux:
            output["fused_features"] = fused
        return output


MODEL_VARIANTS = {
    "m1": ContactTAS,
    "m1_reliability": ContactTAS,
    "contacttas_reliability": ContactTAS,
    "m0": FixedContactTAS,
    "m0_fixed": FixedContactTAS,
    "contacttas_fixed": FixedContactTAS,
    "b2": EarlyFusionContactTAS,
    "b2_early_fusion": EarlyFusionContactTAS,
    "early_fusion": EarlyFusionContactTAS,
}


def normalize_variant(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in MODEL_VARIANTS:
        choices = ", ".join(sorted(MODEL_VARIANTS))
        raise ValueError(f"Unknown model variant {value!r}; choose one of: {choices}")
    if normalized in {"m1", "contacttas_reliability"}:
        return "m1_reliability"
    if normalized in {"m0", "contacttas_fixed"}:
        return "m0_fixed"
    if normalized in {"b2", "early_fusion"}:
        return "b2_early_fusion"
    return normalized


def build_model(config: Mapping[str, Any]) -> ContactTAS:
    model_config = dict(config)
    variant = normalize_variant(str(model_config.get("variant", "m1_reliability")))
    model_config["variant"] = variant
    return MODEL_VARIANTS[variant](model_config)
