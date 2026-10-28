from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax that returns all zeros when every item is invalid."""
    masked_scores = scores.masked_fill(~mask, -1.0e4)
    weights = torch.softmax(masked_scores, dim=dim)
    weights = weights * mask.to(weights.dtype)
    normalizer = weights.sum(dim=dim, keepdim=True)
    return torch.where(normalizer > 0, weights / normalizer.clamp_min(1.0e-6), weights)


class DilatedResidualLayer(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.dilated = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.project = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = F.gelu(self.norm(self.dilated(x)))
        residual = self.dropout(self.project(residual))
        return (x + residual) * mask


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, channels: int, layers: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.layers = nn.ModuleList(
            DilatedResidualLayer(channels, 2**layer, dropout) for layer in range(layers)
        )

    def forward(self, sequence: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        mask = sequence_mask[:, None, :].to(sequence.dtype)
        x = self.input_projection(sequence.transpose(1, 2)) * mask
        for layer in self.layers:
            x = layer(x, mask)
        return x.transpose(1, 2)


class SharedHandTactileEncoder(nn.Module):
    """Flat per-hand encoder over raw, signed-delta, and absolute-delta taxels."""

    REPRESENTATIONS = {"raw", "raw_delta_abs"}

    def __init__(
        self,
        tactile_dim: int,
        output_dim: int,
        dropout: float,
        representation: str = "raw_delta_abs",
    ):
        super().__init__()
        self.representation = str(representation).strip().lower()
        if self.representation not in self.REPRESENTATIONS:
            raise ValueError(
                "tactile_representation must be one of "
                f"{sorted(self.REPRESENTATIONS)}, got {representation!r}"
            )
        self.hand_encoder = nn.Sequential(
            # Keep a fixed 3*Dt input in both controls.  The raw control zeros
            # the two delta blocks rather than changing parameter count.
            nn.Linear(tactile_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )
        self.hand_embedding = nn.Parameter(torch.randn(2, output_dim) * 0.02)

    def forward(self, tactile: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        tactile = tactile * valid[..., None].to(tactile.dtype)
        difference = torch.zeros_like(tactile)
        pair_valid = valid[:, 1:] & valid[:, :-1]
        difference[:, 1:] = (tactile[:, 1:] - tactile[:, :-1]) * pair_valid[
            ..., None
        ].to(tactile.dtype)
        if self.representation == "raw":
            difference = torch.zeros_like(difference)
        encoded = self.hand_encoder(
            torch.cat([tactile, difference, difference.abs()], dim=-1)
        )
        encoded = encoded + self.hand_embedding[None, None, :, :]
        encoded = encoded * valid[..., None].to(encoded.dtype)
        return encoded.flatten(start_dim=2)


class ReliabilityFusion(nn.Module):
    """Independent semantic/event reliability gates, not a forced softmax competition."""

    def __init__(self, dimension: int, hidden: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dimension * 2 + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.residual = nn.Linear(dimension * 2, dimension, bias=False)
        self.norm = nn.LayerNorm(dimension)
        self.null_token = nn.Parameter(torch.zeros(dimension))

    def forward(
        self,
        semantic: torch.Tensor,
        event: torch.Tensor,
        semantic_quality: torch.Tensor,
        event_quality: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        quality = torch.stack([semantic_quality, event_quality], dim=-1).clamp(0.0, 1.0)
        semantic_masked = semantic * quality[..., 0:1]
        event_masked = event * quality[..., 1:2]
        raw_gate = torch.sigmoid(
            self.gate(torch.cat([semantic_masked, event_masked, quality], dim=-1))
        )
        gate = raw_gate * quality
        denominator = gate.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        mixture = (
            gate[..., 0:1] * semantic + gate[..., 1:2] * event
        ) / denominator
        mixture = mixture + self.residual(
            torch.cat(
                [gate[..., 0:1] * semantic, gate[..., 1:2] * event], dim=-1
            )
        )
        all_missing = quality.sum(dim=-1, keepdim=True) <= 0
        mixture = torch.where(all_missing, self.null_token[None, None, :], mixture)
        return self.norm(mixture), gate


class DualHeadStage(nn.Module):
    """One ASRF-inspired action/boundary prediction stage.

    This is a clean-room, compact implementation of the mature design pattern,
    not the official ASRF repository.
    """

    def __init__(
        self,
        input_dim: int,
        channels: int,
        layers: int,
        dropout: float,
        num_actions: int,
        use_boundary_head: bool = True,
    ):
        super().__init__()
        self.use_boundary_head = bool(use_boundary_head)
        self.encoder = TemporalEncoder(input_dim, channels, layers, dropout)
        self.action_head = nn.Linear(channels, num_actions)
        self.boundary_head = (
            nn.Linear(channels, 1) if self.use_boundary_head else None
        )

    def forward(
        self, sequence: torch.Tensor, sequence_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(sequence, sequence_mask)
        action = self.action_head(features)
        if self.boundary_head is None:
            # Preserve the public [B,T] boundary-logit contract while removing
            # every learnable boundary parameter from the controlled ablation.
            boundary = features.new_full(features.shape[:2], -20.0)
        else:
            boundary = self.boundary_head(features).squeeze(-1)
        return action, boundary, features


class BoundaryAwareDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int,
        layers: int,
        stages: int,
        dropout: float,
        num_actions: int,
        num_subtasks: int,
        use_boundary_head: bool = True,
    ):
        super().__init__()
        if stages < 1:
            raise ValueError("Decoder must contain at least one stage")
        self.use_boundary_head = bool(use_boundary_head)
        self.first = DualHeadStage(
            input_dim,
            channels,
            layers,
            dropout,
            num_actions,
            use_boundary_head=self.use_boundary_head,
        )
        self.refinement = nn.ModuleList(
            DualHeadStage(
                num_actions + 1,
                channels,
                layers,
                dropout,
                num_actions,
                use_boundary_head=self.use_boundary_head,
            )
            for _ in range(stages - 1)
        )
        self.subtask_head = (
            nn.Linear(channels, num_subtasks) if num_subtasks > 0 else None
        )

    def forward(self, sequence: torch.Tensor, sequence_mask: torch.Tensor) -> Dict[str, Any]:
        action, boundary, features = self.first(sequence, sequence_mask)
        action_stages: List[torch.Tensor] = [action]
        boundary_stages: List[torch.Tensor] = [boundary]

        for stage in self.refinement:
            boundary_feature = (
                torch.sigmoid(boundary)[..., None]
                if self.use_boundary_head
                else torch.zeros_like(boundary)[..., None]
            )
            refinement_input = torch.cat(
                [torch.softmax(action, dim=-1), boundary_feature],
                dim=-1,
            ).detach()
            action, boundary, features = stage(refinement_input, sequence_mask)
            action_stages.append(action)
            boundary_stages.append(boundary)

        output: Dict[str, Any] = {
            "atomic_logits": action_stages[-1],
            "boundary_logits": boundary_stages[-1],
            "atomic_stages": action_stages,
            "boundary_stages": boundary_stages,
            "decoder_features": features,
        }
        if self.subtask_head is not None:
            output["subtask_logits"] = self.subtask_head(features)
        return output


class ContactTAS(nn.Module):
    """Dual-evidence ContactTAS for cached multimodal features.

    Vision is projected once on its low-rate cache, then gathered by timestamp to
    the 60 Hz grid.  Touch and derived kinematics remain on the 60 Hz event clock.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        vision_dim = int(config["vision_dim"])
        tactile_dim = int(config["tactile_dim"])
        joints = int(config["skeleton_joints"])
        joint_dim = int(config["skeleton_joint_dim"])
        kinematic_dim = int(config["kinematic_dim"])
        dimension = int(config.get("d_model", 256))
        dropout = float(config.get("dropout", 0.2))
        self.output_rate_hz = int(config.get("output_rate_hz", 60))
        self.event_rate_hz = int(config.get("event_rate_hz", self.output_rate_hz))
        if self.output_rate_hz <= 0 or self.event_rate_hz <= 0:
            raise ValueError("output_rate_hz and event_rate_hz must be positive")
        ratio = self.output_rate_hz / self.event_rate_hz
        self.event_stride = int(round(ratio))
        if (
            self.event_rate_hz > self.output_rate_hz
            or self.event_stride < 1
            or abs(ratio - self.event_stride) > 1.0e-6
        ):
            raise ValueError(
                "event_rate_hz must divide output_rate_hz exactly and cannot exceed it: "
                f"event={self.event_rate_hz}, output={self.output_rate_hz}"
            )
        self.modality_dropout = float(config.get("modality_dropout", 0.1))
        self.vision_freshness_scale_ms = float(
            config.get("vision_freshness_scale_ms", 250.0)
        )
        if self.vision_freshness_scale_ms <= 0:
            raise ValueError("vision_freshness_scale_ms must be positive")
        self.enabled_modalities = set(
            config.get(
                "enabled_modalities",
                ["vision", "tactile", "skeleton", "kinematics"],
            )
        )
        unknown_modalities = self.enabled_modalities.difference(
            {"vision", "tactile", "skeleton", "kinematics"}
        )
        if unknown_modalities:
            raise ValueError(f"Unknown enabled_modalities: {sorted(unknown_modalities)}")

        self.vision_adapter = nn.Sequential(
            nn.Linear(vision_dim, dimension),
            nn.LayerNorm(dimension),
            nn.GELU(),
        )
        self.view_score = nn.Linear(dimension, 1)
        self.view_null = nn.Parameter(torch.zeros(dimension))

        skeleton_flat_dim = 2 * joints * joint_dim
        self.skeleton_encoder = nn.Sequential(
            nn.Linear(skeleton_flat_dim, dimension // 2),
            nn.LayerNorm(dimension // 2),
            nn.GELU(),
        )
        self.vision_temporal = TemporalEncoder(
            dimension,
            dimension,
            int(config.get("stream_layers", 4)),
            dropout,
        )
        self.semantic_projection = nn.Linear(dimension + dimension // 2, dimension)
        self.semantic_norm = nn.LayerNorm(dimension)

        hand_dimension = dimension // 2
        self.tactile_encoder = SharedHandTactileEncoder(
            tactile_dim,
            hand_dimension,
            dropout,
            representation=str(
                config.get("tactile_representation", "raw_delta_abs")
            ),
        )
        self.kinematic_encoder = nn.Sequential(
            nn.Linear(kinematic_dim, dimension // 2),
            nn.LayerNorm(dimension // 2),
            nn.GELU(),
        )
        self.event_projection = nn.Linear(
            hand_dimension * 2 + dimension // 2 + dimension // 2,
            dimension,
        )
        self.event_temporal = TemporalEncoder(
            dimension,
            dimension,
            int(config.get("stream_layers", 4)),
            dropout,
        )

        self.fusion = ReliabilityFusion(
            dimension, int(config.get("reliability_hidden", 96))
        )
        self.decoder = BoundaryAwareDecoder(
            input_dim=dimension,
            channels=int(config.get("decoder_channels", 128)),
            layers=int(config.get("decoder_layers", 8)),
            stages=int(config.get("decoder_stages", 3)),
            dropout=dropout,
            num_actions=int(config["num_actions"]),
            num_subtasks=int(config.get("num_subtasks", 0)),
            use_boundary_head=bool(config.get("use_boundary_head", True)),
        )

    def _drop_modalities(
        self, batch: Dict[str, torch.Tensor], enabled: bool
    ) -> Dict[str, torch.Tensor]:
        names = (
            "vision_valid_60",
            "tactile_valid",
            "skeleton_valid",
            "kinematics_valid",
        )
        masks = {name: batch[name] for name in names}
        modality_for_mask = {
            "vision_valid_60": "vision",
            "tactile_valid": "tactile",
            "skeleton_valid": "skeleton",
            "kinematics_valid": "kinematics",
        }
        masks = {
            name: value
            if modality_for_mask[name] in self.enabled_modalities
            else torch.zeros_like(value)
            for name, value in masks.items()
        }
        if not enabled or not self.training or self.modality_dropout <= 0:
            return masks

        batch_size = batch["sequence_mask"].shape[0]
        output = dict(masks)
        for name in names:
            keep_shape = (batch_size,) + (1,) * (batch[name].ndim - 1)
            keep = torch.rand(keep_shape, device=batch[name].device) >= self.modality_dropout
            output[name] = masks[name] & keep
        return output

    @staticmethod
    def _gather_vision(
        cached: torch.Tensor,
        cache_valid: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gathered_views: List[torch.Tensor] = []
        gathered_valid: List[torch.Tensor] = []
        for view in range(3):
            view_index = indices[..., view].clamp_min(0)
            gather_index = view_index[..., None].expand(-1, -1, cached.shape[-1])
            gathered_views.append(torch.gather(cached[:, :, view], 1, gather_index))
            gathered_valid.append(torch.gather(cache_valid[:, :, view], 1, view_index))
        return torch.stack(gathered_views, dim=2), torch.stack(gathered_valid, dim=2)

    @staticmethod
    def _pool_sequence(
        sequence: torch.Tensor,
        sequence_mask: torch.Tensor,
        stride: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask-aware block average used by the controlled event-rate ablation."""
        if stride == 1:
            return sequence, sequence_mask
        batch_size, steps, dimension = sequence.shape
        padded_steps = ((steps + stride - 1) // stride) * stride
        padding = padded_steps - steps
        if padding:
            sequence = F.pad(sequence, (0, 0, 0, padding))
            sequence_mask = F.pad(sequence_mask, (0, padding), value=False)
        block_mask = sequence_mask.reshape(batch_size, -1, stride)
        weights = block_mask.to(sequence.dtype)
        pooled = (sequence.reshape(batch_size, -1, stride, dimension) * weights[..., None]).sum(
            dim=2
        )
        pooled = pooled / weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        return pooled, block_mask.any(dim=2)

    @staticmethod
    def _repeat_to_steps(sequence: torch.Tensor, stride: int, steps: int) -> torch.Tensor:
        if stride == 1:
            return sequence[:, :steps]
        return sequence.repeat_interleave(stride, dim=1)[:, :steps]

    @staticmethod
    def _masked_temporal_difference(
        sequence: torch.Tensor, quality: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiate only across two consecutively observable packets.

        Returning the pair quality makes the exact validity signal available to
        reliability fusion; a recovered packet after a gap must not create a
        large synthetic motion event relative to the stale packet before it.
        """

        if quality.shape != sequence.shape[:2]:
            raise ValueError("temporal-difference quality must have shape [B,T]")
        difference = torch.zeros_like(sequence)
        pair_quality = torch.zeros_like(quality, dtype=sequence.dtype)
        if sequence.shape[1] > 1:
            pair_quality[:, 1:] = torch.minimum(
                quality[:, 1:].to(sequence.dtype),
                quality[:, :-1].to(sequence.dtype),
            )
            difference[:, 1:] = (
                sequence[:, 1:] - sequence[:, :-1]
            ) * pair_quality[:, 1:, None]
        return difference, pair_quality

    def _limit_event_clock(
        self,
        sequence: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Downsample then hold event evidence on the configured output grid."""
        pooled, _ = self._pool_sequence(sequence, sequence_mask, self.event_stride)
        return self._repeat_to_steps(pooled, self.event_stride, sequence.shape[1])

    def _limit_event_quality_clock(
        self,
        quality: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate and hold reliability on the exact event-feature blocks.

        A low-rate event ablation must not feed a 15 Hz event feature together
        with a quality/presence signal that still changes at 60 Hz.  Reusing the
        same mask-aware block average and hold operation makes both M1's
        continuous reliability and M0's derived hard-presence mask obey the
        configured event clock.
        """
        if quality.shape != sequence_mask.shape:
            raise ValueError("event quality and sequence mask must have shape [B,T]")
        held = self._limit_event_clock(quality[..., None], sequence_mask)
        return held.squeeze(-1)

    def _encode_event_stream(
        self,
        sequence: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled, pooled_mask = self._pool_sequence(
            sequence, sequence_mask, self.event_stride
        )
        encoded = self.event_temporal(pooled, pooled_mask)
        return self._repeat_to_steps(encoded, self.event_stride, sequence.shape[1])

    def _encode_modalities(
        self,
        batch: Dict[str, torch.Tensor],
        apply_modality_dropout: bool,
    ) -> Dict[str, torch.Tensor]:
        """Encode shared modality tokens for ContactTAS and controlled baselines."""
        sequence_mask = batch["sequence_mask"]
        valid = self._drop_modalities(batch, apply_modality_dropout)

        cached_vision = self.vision_adapter(batch["vision_features"])
        cached_vision = torch.stack(
            [
                self.vision_temporal(
                    cached_vision[:, :, view],
                    batch["vision_cache_valid"][:, :, view],
                )
                for view in range(3)
            ],
            dim=2,
        )
        gathered_vision, gathered_cache_valid = self._gather_vision(
            cached_vision,
            batch["vision_cache_valid"],
            batch["vision_index_60"],
        )
        view_valid = valid["vision_valid_60"] & gathered_cache_valid
        freshness_penalty = (
            batch["vision_freshness_ms"].clamp_min(0)
            / self.vision_freshness_scale_ms
        )
        view_scores = self.view_score(gathered_vision).squeeze(-1) - freshness_penalty
        view_weights = _masked_softmax(view_scores, view_valid, dim=2)
        vision_token = (view_weights[..., None] * gathered_vision).sum(dim=2)
        all_views_missing = ~view_valid.any(dim=2)
        vision_token = torch.where(
            all_views_missing[..., None], self.view_null[None, None, :], vision_token
        )

        skeleton_mask = valid["skeleton_valid"]
        skeleton_joint_valid = batch.get("skeleton_joint_valid")
        if skeleton_joint_valid is not None:
            expected = batch["skeleton"].shape[:-1]
            if skeleton_joint_valid.shape != expected:
                raise ValueError(
                    "skeleton_joint_valid must have shape "
                    f"{tuple(expected)}, got {tuple(skeleton_joint_valid.shape)}"
                )
            skeleton_joint_valid = skeleton_joint_valid.bool() & skeleton_mask[..., None]
            skeleton = batch["skeleton"] * skeleton_joint_valid[..., None].to(
                batch["skeleton"].dtype
            )
        else:
            skeleton = batch["skeleton"] * skeleton_mask[..., None, None].to(
                batch["skeleton"].dtype
            )
        skeleton_token = self.skeleton_encoder(skeleton.flatten(start_dim=2))
        tactile_token = self.tactile_encoder(batch["tactile"], valid["tactile_valid"])
        kinematics = batch["kinematics"] * valid["kinematics_valid"][..., None].to(
            batch["kinematics"].dtype
        )
        kinematic_token = self.kinematic_encoder(kinematics)

        # Reliability is computed only from observable cache fields.  Debug NPZs
        # that predate the formal quality contract retain a conservative mask-only
        # fallback, while formal caches supply all three continuous quality inputs.
        freshness_quality = torch.exp(-freshness_penalty).to(vision_token.dtype)
        vision_quality = (view_weights * freshness_quality).sum(dim=2)

        if skeleton_joint_valid is None:
            skeleton_quality = skeleton_mask.any(dim=2).to(skeleton_token.dtype)
        else:
            skeleton_quality = skeleton_joint_valid.to(skeleton_token.dtype).mean(
                dim=(2, 3)
            )
        skeleton_token = skeleton_token * (skeleton_quality > 0)[..., None].to(
            skeleton_token.dtype
        )

        tactile_observed = batch.get("tactile_quality")
        if tactile_observed is None:
            tactile_quality = valid["tactile_valid"].any(dim=2).to(
                tactile_token.dtype
            )
        else:
            expected = (*valid["tactile_valid"].shape, 5)
            if tactile_observed.shape != expected:
                raise ValueError(
                    f"tactile_quality must have shape {expected}, "
                    f"got {tuple(tactile_observed.shape)}"
                )
            tactile_observed = torch.nan_to_num(
                tactile_observed.to(tactile_token.dtype), nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)
            packet_valid = tactile_observed[..., 0]
            # zero_fraction is diagnostic only: a zero tactile reading is legal
            # and must never be reinterpreted as missing/anomalous.  Reliability
            # penalizes observable stuck/jump/saturation summaries (2:5).
            anomaly_fraction = tactile_observed[..., 2:].mean(dim=-1)
            per_hand_quality = packet_valid * (1.0 - anomaly_fraction)
            per_hand_quality = per_hand_quality * valid["tactile_valid"].to(
                per_hand_quality.dtype
            )
            tactile_quality = per_hand_quality.mean(dim=2)

        pose_observed = batch.get("pose_quality")
        if pose_observed is None:
            kinematic_quality = valid["kinematics_valid"].to(kinematic_token.dtype)
        else:
            expected = (*valid["kinematics_valid"].shape, 6)
            if pose_observed.shape != expected:
                raise ValueError(
                    f"pose_quality must have shape {expected}, "
                    f"got {tuple(pose_observed.shape)}"
                )
            pose_observed = torch.nan_to_num(
                pose_observed.to(kinematic_token.dtype),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            pose_validity = pose_observed[..., :3].mean(dim=-1)
            timing_quality = pose_observed[..., 3]
            artifact_quality = 1.0 - pose_observed[..., 4:].mean(dim=-1)
            kinematic_quality = pose_validity * timing_quality * artifact_quality
            kinematic_quality = kinematic_quality * valid["kinematics_valid"].to(
                kinematic_quality.dtype
            )
        kinematic_token = kinematic_token * valid["kinematics_valid"][..., None].to(
            kinematic_token.dtype
        )
        _, skeleton_motion_quality = self._masked_temporal_difference(
            skeleton_token, skeleton_quality
        )

        semantic_components = []
        if "vision" in self.enabled_modalities:
            semantic_components.append(vision_quality)
        if "skeleton" in self.enabled_modalities:
            semantic_components.append(skeleton_quality)
        semantic_quality = (
            torch.stack(semantic_components, dim=0).mean(dim=0)
            if semantic_components
            else torch.zeros_like(vision_quality)
        )
        event_components = []
        if "tactile" in self.enabled_modalities:
            event_components.append(tactile_quality)
        if "kinematics" in self.enabled_modalities:
            event_components.append(kinematic_quality)
        if "skeleton" in self.enabled_modalities:
            event_components.append(skeleton_motion_quality)
        event_quality = (
            torch.stack(event_components, dim=0).mean(dim=0)
            if event_components
            else torch.zeros_like(tactile_quality)
        )
        return {
            "sequence_mask": sequence_mask,
            "vision_token": vision_token,
            "skeleton_token": skeleton_token,
            "tactile_token": tactile_token,
            "kinematic_token": kinematic_token,
            "vision_quality": vision_quality,
            "skeleton_quality": skeleton_quality,
            "skeleton_motion_quality": skeleton_motion_quality,
            "tactile_quality": tactile_quality,
            "kinematic_quality": kinematic_quality,
            "semantic_quality": semantic_quality,
            "event_quality": event_quality,
            "view_weights": view_weights,
        }

    def _encode_streams(
        self, encoded: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        semantic = self.semantic_norm(
            F.gelu(
                self.semantic_projection(
                    torch.cat(
                        [encoded["vision_token"], encoded["skeleton_token"]], dim=-1
                    )
                )
            )
        )
        skeleton_difference, _ = self._masked_temporal_difference(
            encoded["skeleton_token"], encoded["skeleton_quality"]
        )
        event_input = self.event_projection(
            torch.cat(
                [
                    encoded["tactile_token"],
                    encoded["kinematic_token"],
                    skeleton_difference,
                ],
                dim=-1,
            )
        )
        event = self._encode_event_stream(event_input, encoded["sequence_mask"])
        return semantic, event

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        apply_modality_dropout: bool = False,
        return_aux: bool = False,
    ) -> Dict[str, Any]:
        encoded = self._encode_modalities(batch, apply_modality_dropout)
        sequence_mask = encoded["sequence_mask"]
        semantic, event = self._encode_streams(encoded)
        event_quality_clocked = self._limit_event_quality_clock(
            encoded["event_quality"], sequence_mask
        )
        fused, reliability = self.fusion(
            semantic,
            event,
            encoded["semantic_quality"],
            event_quality_clocked,
        )
        fused = fused * sequence_mask[..., None].to(fused.dtype)
        output = self.decoder(fused, sequence_mask)
        output["reliability_weights"] = reliability
        output["view_weights"] = encoded["view_weights"]
        if return_aux:
            output["semantic_features"] = semantic
            output["event_features"] = event
            output["fused_features"] = fused
            output["event_quality_clocked"] = event_quality_clocked
        return output
