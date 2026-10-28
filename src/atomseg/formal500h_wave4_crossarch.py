"""Formal500h Wave-4 cross-architecture validation matrix.

This module freezes a 3 x 3 x 3 experiment:

* temporal heads: MS-TCN++-style, ASRF-style, and DiffAct-style;
* representations: vision-only (V), vision+tactile early fusion (VT), and an
  explicitly adapted M2R2-style vision+tactile representation (M2R2);
* the three paired Formal500h seeds used by Wave-1/Wave-3B.

All implementations are clean-room adaptations to the frozen cached-feature
and 30 Hz evaluator contract.  They are not official reproductions.  The
M2R2-style representation contains independent modality encoders, a modality
Transformer, action-order contrastive supervision, and Gaussian soft-boundary
MSE.  The formal test manifest remains sealed throughout this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .baselines import EarlyFusionContactTAS
from .formal500h_evaluation import evaluate_dataset
from .formal500h_infra_runtime import (
    VariantAwareFormal500hDataset,
    collate_formal500h_fast,
)
from .formal500h_protocol import NUM_CLASSES, sha256_file
from .formal500h_runtime import formal500h_loss_terms as base_formal500h_loss_terms
from .formal500h_wave1_data import WAVE1_DATA_SCHEMA, WINDOW_SCHEMA
from .model import (
    BoundaryAwareDecoder,
    DilatedResidualLayer,
    SharedHandTactileEncoder,
    TemporalEncoder,
    _masked_softmax,
)
from . import formal500h_wave1_train as wave1
from . import formal500h_wave3b_vt as wave3b


TRAINING_SCHEMA = "formal500h_wave4_crossarch_training_v1"
MODEL_GATE_SCHEMA = "formal500h_wave4_crossarch_model_gate_v1"
AGGREGATE_SCHEMA = "formal500h_wave4_crossarch_aggregate_v1"
COMPLETE_SCHEMA = "formal500h_wave4_crossarch_complete_v1"
RUNTIME_SCHEMA = "formal500h_wave4_crossarch_runtime_v1"

HEADS = ("MSTCNPP", "ASRF", "DIFFACT")
REPRESENTATIONS = ("V", "VT", "M2R2")
SEEDS = (20260824, 20260825, 20260826)
MODALITIES = {
    "V": ("vision",),
    "VT": ("vision", "tactile"),
    "M2R2": ("vision", "tactile"),
}
BOUNDARY_HEADS = frozenset({"ASRF"})
BOUNDARY_TOLERANCES_MS = (50, 100, 250, 500, 1000)

FROZEN_PRECISION = "bf16"
FROZEN_BATCH_SIZE = 16
FROZEN_NUM_WORKERS = 20
FROZEN_EPOCHS = 3
FROZEN_LEARNING_RATE = 3.0e-4
FROZEN_WEIGHT_DECAY = 1.0e-4
FROZEN_HAND_WEIGHT = 0.1
FROZEN_CALIBRATION_WORKERS = 4
FROZEN_CHECKPOINT_INTERVAL = 2000
ACTION_ORDER_CONTRASTIVE_WEIGHT = 0.1
SOFT_BOUNDARY_MSE_WEIGHT = 0.1
ACTION_ORDER_TEMPERATURE = 0.07


@dataclass(frozen=True)
class CellSpec:
    head: str
    representation: str

    @property
    def cell(self) -> str:
        return f"{self.head}_{self.representation}"

    @property
    def modalities(self) -> tuple[str, ...]:
        return MODALITIES[self.representation]

    @property
    def use_boundary_head(self) -> bool:
        return self.head in BOUNDARY_HEADS

    @property
    def is_m2r2_style(self) -> bool:
        return self.representation == "M2R2"

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "head": self.head,
            "representation": self.representation,
            "reporting_label": (
                "D30B"
                if self.representation == "VT"
                else (
                    "M2R2-style adapted"
                    if self.representation == "M2R2"
                    else "V"
                )
            ),
            "enabled_modalities": list(self.modalities),
            "use_boundary_head": self.use_boundary_head,
            "boundary_mechanism": (
                "learned_boundary_head_plus_action_change"
                if self.use_boundary_head
                else "decoded_action_change_only"
            ),
            "official_reproduction": False,
        }


CELL_SPECS = tuple(
    CellSpec(head, representation)
    for head in HEADS
    for representation in REPRESENTATIONS
)
CELL_BY_ID = {spec.cell: spec for spec in CELL_SPECS}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _architecture_contract(spec: CellSpec) -> dict[str, Any]:
    head_contract: dict[str, Any]
    if spec.head == "MSTCNPP":
        head_contract = {
            "family": "ms_tcn_pp_clean_room_adaptation",
            "prediction_generation": "dual_dilated_residual",
            "refinement_input": "detached_action_probability",
            "boundary_source": "decoded_action_change",
        }
    elif spec.head == "ASRF":
        head_contract = {
            "family": "asrf_clean_room_adaptation",
            "branches": ["action", "class_agnostic_boundary"],
            "refinement_input": "detached_action_probability_and_boundary",
            "boundary_source": "boundary_head_plus_action_change",
        }
    elif spec.head == "DIFFACT":
        head_contract = {
            "family": "diffact_clean_room_adaptation",
            "training": "random_timestep_gaussian_label_denoising",
            "inference": "deterministic_iterative_denoising",
            "boundary_source": "decoded_action_change",
        }
    else:  # pragma: no cover - guarded by CellSpec construction
        raise ValueError(spec.head)

    if spec.representation == "M2R2":
        representation_contract = {
            "family": "m2r2_style_adapted",
            "official_m2r2_reproduction": False,
            "independent_modality_encoders": ["vision", "tactile"],
            "fusion": "per_tick_modality_transformer_with_summary_token",
            "auxiliary_supervision": [
                "action_order_symmetric_info_nce",
                "gaussian_soft_boundary_mse",
            ],
        }
    else:
        representation_contract = {
            "family": "parameter_matched_early_fusion",
            "reporting_label": (
                "D30B" if spec.representation == "VT" else "V"
            ),
            "enabled_modalities": list(spec.modalities),
            "tactile_recipe": (
                {
                    "pre_registered_id": "D30B",
                    "representation": "raw_delta_abs",
                    "event_rate_hz": 30,
                }
                if spec.representation == "VT"
                else None
            ),
            "auxiliary_supervision": [],
        }
    return {
        "schema_version": "formal500h_wave4_architecture_contract_v1",
        "cell": spec.cell,
        "head": head_contract,
        "representation": representation_contract,
        "output_rate_hz": 30,
        "num_actions": NUM_CLASSES,
        "official_reproduction": False,
    }


def architecture_fingerprint(spec: CellSpec) -> str:
    return _canonical_sha256(_architecture_contract(spec))


def _model_config(
    spec: CellSpec,
    *,
    d_model: int = 128,
    dropout: float = 0.15,
) -> dict[str, Any]:
    contract = _architecture_contract(spec)
    return {
        "variant": "wave4_crossarch",
        "formal_variant": spec.cell,
        "head": spec.head,
        "representation": spec.representation,
        "architecture_contract": contract,
        "architecture_fingerprint_sha256": _canonical_sha256(contract),
        "official_reproduction": False,
        "enabled_modalities": list(spec.modalities),
        "vision_dim": 1536,
        "tactile_dim": 360,
        "tactile_representation": "raw_delta_abs",
        "skeleton_joints": 25,
        "skeleton_joint_dim": 7,
        "kinematic_dim": 48,
        "d_model": int(d_model),
        "dropout": float(dropout),
        "modality_dropout": 0.0,
        "stream_layers": 3,
        "decoder_channels": max(64, int(d_model)),
        "decoder_layers": 5,
        "decoder_stages": 2,
        "diffusion_train_steps": 100,
        "diffusion_inference_steps": 4,
        "modality_transformer_layers": 2,
        "modality_transformer_heads": 4,
        "num_actions": NUM_CLASSES,
        "num_subtasks": 0,
        "use_boundary_head": spec.use_boundary_head,
        "output_rate_hz": 30,
        "event_rate_hz": 30,
        "action_order_contrastive_weight": (
            ACTION_ORDER_CONTRASTIVE_WEIGHT if spec.is_m2r2_style else 0.0
        ),
        "soft_boundary_mse_weight": (
            SOFT_BOUNDARY_MSE_WEIGHT if spec.is_m2r2_style else 0.0
        ),
        "action_order_temperature": ACTION_ORDER_TEMPERATURE,
    }


class DualDilatedResidualLayer(nn.Module):
    """MS-TCN++ prediction-generation dual-dilation block."""

    def __init__(
        self,
        channels: int,
        dilation_forward: int,
        dilation_reverse: int,
        dropout: float,
    ):
        super().__init__()
        self.forward_dilated = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation_forward,
            dilation=dilation_forward,
        )
        self.reverse_dilated = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation_reverse,
            dilation=dilation_reverse,
        )
        self.fuse = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        first = self.forward_dilated(value)
        second = self.reverse_dilated(value)
        residual = F.gelu(self.norm(self.fuse(torch.cat([first, second], dim=1))))
        return (value + self.dropout(residual)) * mask


class MSTCNPPPredictionGeneration(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int,
        layers: int,
        dropout: float,
        num_actions: int,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.layers = nn.ModuleList(
            DualDilatedResidualLayer(
                channels,
                2**index,
                2 ** (layers - 1 - index),
                dropout,
            )
            for index in range(layers)
        )
        self.action_head = nn.Conv1d(channels, num_actions, kernel_size=1)

    def forward(
        self, sequence: torch.Tensor, sequence_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = sequence_mask[:, None, :].to(sequence.dtype)
        value = self.input_projection(sequence.transpose(1, 2)) * mask
        for layer in self.layers:
            value = layer(value, mask)
        logits = self.action_head(value).transpose(1, 2)
        return logits, value.transpose(1, 2)


class MSTCNPPRefinement(nn.Module):
    def __init__(
        self,
        num_actions: int,
        channels: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = TemporalEncoder(num_actions, channels, layers, dropout)
        self.action_head = nn.Linear(channels, num_actions)

    def forward(
        self, probability: torch.Tensor, sequence_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(probability, sequence_mask)
        return self.action_head(features), features


class MSTCNPPDecoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        channels = int(config["decoder_channels"])
        layers = int(config["decoder_layers"])
        stages = int(config["decoder_stages"])
        num_actions = int(config["num_actions"])
        dropout = float(config["dropout"])
        if stages < 1:
            raise ValueError("MS-TCN++ requires at least one stage")
        self.first = MSTCNPPPredictionGeneration(
            int(config["d_model"]), channels, layers, dropout, num_actions
        )
        self.refinement = nn.ModuleList(
            MSTCNPPRefinement(num_actions, channels, layers, dropout)
            for _ in range(stages - 1)
        )

    def forward(
        self,
        sequence: torch.Tensor,
        sequence_mask: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del target
        logits, features = self.first(sequence, sequence_mask)
        stages = [logits]
        for refinement in self.refinement:
            logits, features = refinement(
                torch.softmax(logits, dim=-1).detach(), sequence_mask
            )
            stages.append(logits)
        boundaries = [
            logits.new_full(logits.shape[:2], -20.0) for logits in stages
        ]
        return {
            "atomic_logits": stages[-1],
            "boundary_logits": boundaries[-1],
            "atomic_stages": stages,
            "boundary_stages": boundaries,
            "decoder_features": features,
        }


class DiffActDenoiser(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        channels = int(config["decoder_channels"])
        num_actions = int(config["num_actions"])
        self.condition = TemporalEncoder(
            int(config["d_model"]), channels, 3, float(config["dropout"])
        )
        self.label_projection = nn.Linear(num_actions, channels)
        self.time_embedding = nn.Embedding(
            int(config["diffusion_train_steps"]), channels
        )
        self.denoiser = TemporalEncoder(
            channels * 3,
            channels,
            int(config["decoder_layers"]),
            float(config["dropout"]),
        )
        self.action_head = nn.Linear(channels, num_actions)

    def forward(
        self,
        sequence: torch.Tensor,
        noisy_labels: torch.Tensor,
        timestep: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.condition(sequence, sequence_mask)
        noisy = self.label_projection(noisy_labels)
        timing = self.time_embedding(timestep)[:, None, :].expand_as(condition)
        features = self.denoiser(
            torch.cat([condition, noisy, timing], dim=-1), sequence_mask
        )
        return self.action_head(features), features


class DiffActDecoder(nn.Module):
    """Compact DiffAct-style Gaussian label denoising adaptation.

    Validation is deterministic and never consumes ``atomic_target``.  Targets
    are required only in training mode to construct the noisy label state.
    """

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.num_actions = int(config["num_actions"])
        self.train_steps = int(config["diffusion_train_steps"])
        self.inference_steps = int(config["diffusion_inference_steps"])
        if self.train_steps < 2 or self.inference_steps < 1:
            raise ValueError("invalid DiffAct step count")
        self.denoiser = DiffActDenoiser(config)

    def _alpha_bar(self, timestep: torch.Tensor) -> torch.Tensor:
        phase = (timestep.to(torch.float32) + 0.5) / self.train_steps
        return torch.cos(phase * math.pi / 2).square().clamp(1.0e-4, 0.9999)

    def forward(
        self,
        sequence: torch.Tensor,
        sequence_mask: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        batch_size, steps = sequence_mask.shape
        if self.training:
            if target is None or target.shape != sequence_mask.shape:
                raise ValueError("DiffAct training requires aligned atomic_target")
            timestep = torch.randint(
                0, self.train_steps, (batch_size,), device=sequence.device
            )
            safe_target = target.masked_fill(~sequence_mask, 0).clamp(
                0, self.num_actions - 1
            )
            clean = F.one_hot(safe_target, self.num_actions).to(sequence.dtype)
            alpha = self._alpha_bar(timestep).to(sequence.dtype)[:, None, None]
            noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * torch.randn_like(clean)
            noisy = noisy * sequence_mask[..., None].to(noisy.dtype)
            logits, features = self.denoiser(
                sequence, noisy, timestep, sequence_mask
            )
            stages = [logits]
        else:
            noisy = sequence.new_zeros(batch_size, steps, self.num_actions)
            schedule = torch.linspace(
                self.train_steps - 1,
                0,
                self.inference_steps,
                device=sequence.device,
            ).round().to(torch.long)
            stages: list[torch.Tensor] = []
            features = sequence
            for index, scalar_timestep in enumerate(schedule):
                timestep = scalar_timestep.expand(batch_size)
                logits, features = self.denoiser(
                    sequence, noisy, timestep, sequence_mask
                )
                stages.append(logits)
                predicted = torch.softmax(logits, dim=-1)
                blend = 1.0 / float(index + 1)
                noisy = (
                    blend * predicted + (1.0 - blend) * noisy
                ) * sequence_mask[..., None].to(predicted.dtype)
        boundaries = [
            logits.new_full(logits.shape[:2], -20.0) for logits in stages
        ]
        return {
            "atomic_logits": stages[-1],
            "boundary_logits": boundaries[-1],
            "atomic_stages": stages,
            "boundary_stages": boundaries,
            "decoder_features": features,
        }


def _build_decoder(config: Mapping[str, Any]) -> nn.Module:
    head = str(config["head"]).upper()
    if head == "MSTCNPP":
        return MSTCNPPDecoder(config)
    if head == "ASRF":
        return BoundaryAwareDecoder(
            input_dim=int(config["d_model"]),
            channels=int(config["decoder_channels"]),
            layers=int(config["decoder_layers"]),
            stages=int(config["decoder_stages"]),
            dropout=float(config["dropout"]),
            num_actions=int(config["num_actions"]),
            num_subtasks=int(config.get("num_subtasks", 0)),
            use_boundary_head=True,
        )
    if head == "DIFFACT":
        return DiffActDecoder(config)
    raise ValueError(f"unknown Wave-4 temporal head: {head}")


class EarlyFusionCrossArch(nn.Module):
    """Wave-3B early representation with a replaceable temporal head."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.encoder = EarlyFusionContactTAS(dict(config))
        self.encoder.decoder = _build_decoder(config)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        encoded = self.encoder._encode_modalities(batch, False)
        sequence_mask = encoded["sequence_mask"]
        vision = encoded["vision_token"] * encoded["vision_quality"][..., None]
        skeleton = (
            encoded["skeleton_token"] * encoded["skeleton_quality"][..., None]
        )
        tactile = encoded["tactile_token"] * encoded["tactile_quality"][..., None]
        kinematics = (
            encoded["kinematic_token"]
            * encoded["kinematic_quality"][..., None]
        )
        event_evidence = self.encoder._limit_event_clock(
            torch.cat([tactile, kinematics], dim=-1), sequence_mask
        )
        fused = self.encoder.early_norm(
            F.gelu(
                self.encoder.early_projection(
                    torch.cat([vision, skeleton, event_evidence], dim=-1)
                )
            )
        )
        fused = self.encoder.early_temporal(fused, sequence_mask)
        fused = fused * sequence_mask[..., None].to(fused.dtype)
        target = batch.get("atomic_target") if self.training else None
        if isinstance(self.encoder.decoder, DiffActDecoder):
            output = self.encoder.decoder(fused, sequence_mask, target=target)
        else:
            output = self.encoder.decoder(fused, sequence_mask)
        output["reliability_weights"] = torch.stack(
            [encoded["semantic_quality"], encoded["event_quality"]], dim=-1
        )
        output["view_weights"] = encoded["view_weights"]
        return output


class M2R2StyleRepresentation(nn.Module):
    """Independent V/T encoders plus a per-tick modality Transformer.

    This is an explicitly adapted cached-feature implementation.  It does not
    use ActionCLIP, text prompts, or the original robot modalities and therefore
    must never be described as the official M2R2 model.
    """

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        dimension = int(config["d_model"])
        dropout = float(config["dropout"])
        self.dimension = dimension
        self.num_actions = int(config["num_actions"])
        self.enabled_modalities = frozenset(config["enabled_modalities"])
        self.vision_adapter = nn.Sequential(
            nn.Linear(int(config["vision_dim"]), dimension),
            nn.LayerNorm(dimension),
            nn.GELU(),
        )
        self.vision_temporal = TemporalEncoder(
            dimension, dimension, int(config["stream_layers"]), dropout
        )
        self.view_score = nn.Linear(dimension, 1)
        self.view_null = nn.Parameter(torch.zeros(dimension))
        self.tactile_encoder = SharedHandTactileEncoder(
            int(config["tactile_dim"]),
            dimension // 2,
            dropout,
            representation=str(config["tactile_representation"]),
        )
        self.summary_token = nn.Parameter(torch.zeros(dimension))
        self.modality_embedding = nn.Parameter(torch.randn(2, dimension) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=int(config["modality_transformer_heads"]),
            dim_feedforward=dimension * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.modality_transformer = nn.TransformerEncoder(
            layer, num_layers=int(config["modality_transformer_layers"])
        )
        self.temporal = TemporalEncoder(
            dimension, dimension, int(config["stream_layers"]), dropout
        )
        self.soft_boundary_head = nn.Linear(dimension, 1)
        self.sequence_projection = nn.Linear(dimension, dimension)
        self.order_label_embedding = nn.Embedding(self.num_actions, dimension)
        self.order_encoder = nn.GRU(dimension, dimension, batch_first=True)
        self.order_projection = nn.Linear(dimension, dimension)

    @staticmethod
    def _gather_vision(
        cached: torch.Tensor,
        cache_valid: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gathered: list[torch.Tensor] = []
        validity: list[torch.Tensor] = []
        for view in range(3):
            view_index = indices[..., view].clamp_min(0)
            gather_index = view_index[..., None].expand(
                -1, -1, cached.shape[-1]
            )
            gathered.append(torch.gather(cached[:, :, view], 1, gather_index))
            validity.append(
                torch.gather(cache_valid[:, :, view], 1, view_index)
            )
        return torch.stack(gathered, dim=2), torch.stack(validity, dim=2)

    def _vision(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        cached = self.vision_adapter(batch["vision_features"])
        cached = torch.stack(
            [
                self.vision_temporal(
                    cached[:, :, view], batch["vision_cache_valid"][:, :, view]
                )
                for view in range(3)
            ],
            dim=2,
        )
        gathered, gathered_valid = self._gather_vision(
            cached, batch["vision_cache_valid"], batch["vision_index_60"]
        )
        valid = batch["vision_valid_60"] & gathered_valid
        freshness = batch["vision_freshness_ms"].clamp_min(0) / 250.0
        scores = self.view_score(gathered).squeeze(-1) - freshness
        weights = _masked_softmax(scores, valid, dim=2)
        token = (weights[..., None] * gathered).sum(dim=2)
        present = valid.any(dim=2)
        token = torch.where(
            present[..., None], token, self.view_null[None, None, :]
        )
        return token, present, weights

    def _order_target(
        self,
        target: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequences: list[torch.Tensor] = []
        lengths: list[int] = []
        for labels, mask in zip(target, sequence_mask):
            valid = labels[mask]
            valid = valid[(valid >= 0) & (valid < self.num_actions)]
            if valid.numel() == 0:
                collapsed = labels.new_zeros(1)
            else:
                keep = torch.ones_like(valid, dtype=torch.bool)
                keep[1:] = valid[1:] != valid[:-1]
                collapsed = valid[keep]
            sequences.append(collapsed)
            lengths.append(int(collapsed.numel()))
        maximum = max(lengths)
        padded = target.new_zeros(len(sequences), maximum)
        for index, sequence in enumerate(sequences):
            padded[index, : sequence.numel()] = sequence
        encoded, _ = self.order_encoder(self.order_label_embedding(padded))
        gather = torch.tensor(
            [value - 1 for value in lengths], device=target.device, dtype=torch.long
        )
        final = encoded[torch.arange(len(sequences), device=target.device), gather]
        return F.normalize(self.order_projection(final), dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sequence_mask = batch["sequence_mask"]
        vision, vision_present, view_weights = self._vision(batch)
        tactile_valid = batch["tactile_valid"]
        if "tactile" not in self.enabled_modalities:
            tactile_valid = torch.zeros_like(tactile_valid)
        tactile = self.tactile_encoder(batch["tactile"], tactile_valid)
        tactile_present = tactile_valid.any(dim=2)
        batch_size, steps, _ = vision.shape
        summary = self.summary_token[None, None, :].expand(
            batch_size * steps, 1, -1
        )
        modality = torch.stack(
            [
                vision + self.modality_embedding[0],
                tactile + self.modality_embedding[1],
            ],
            dim=2,
        ).reshape(batch_size * steps, 2, self.dimension)
        tokens = torch.cat([summary, modality], dim=1)
        missing = torch.stack(
            [~vision_present, ~tactile_present], dim=-1
        ).reshape(batch_size * steps, 2)
        padding = torch.cat(
            [torch.zeros_like(missing[:, :1]), missing], dim=1
        )
        transformed = self.modality_transformer(
            tokens, src_key_padding_mask=padding
        )
        fused = transformed[:, 0].reshape(batch_size, steps, self.dimension)
        fused = self.temporal(fused, sequence_mask)
        fused = fused * sequence_mask[..., None].to(fused.dtype)
        denominator = sequence_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (
            fused * sequence_mask[..., None].to(fused.dtype)
        ).sum(dim=1) / denominator.to(fused.dtype)
        output = {
            "features": fused,
            "soft_boundary_logits": self.soft_boundary_head(fused).squeeze(-1),
            "representation_embedding": F.normalize(
                self.sequence_projection(pooled), dim=-1
            ),
            "view_weights": view_weights,
            "reliability_weights": torch.stack(
                [vision_present, tactile_present], dim=-1
            ).to(fused.dtype),
        }
        if self.training:
            target = batch.get("atomic_target")
            if target is None:
                raise ValueError("M2R2-style training requires atomic_target")
            output["order_embedding"] = self._order_target(target, sequence_mask)
        return output


class M2R2CrossArch(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.representation = M2R2StyleRepresentation(config)
        self.decoder = _build_decoder(config)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        represented = self.representation(batch)
        target = batch.get("atomic_target") if self.training else None
        if isinstance(self.decoder, DiffActDecoder):
            output = self.decoder(
                represented["features"], batch["sequence_mask"], target=target
            )
        else:
            output = self.decoder(represented["features"], batch["sequence_mask"])
        for name in (
            "soft_boundary_logits",
            "representation_embedding",
            "order_embedding",
            "view_weights",
            "reliability_weights",
        ):
            if name in represented:
                output[name] = represented[name]
        return output


class CrossArchFormal500hModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        representation = str(config["representation"]).upper()
        if representation == "M2R2":
            self.backbone = M2R2CrossArch(config)
        elif representation in {"V", "VT"}:
            self.backbone = EarlyFusionCrossArch(config)
        else:
            raise ValueError(f"unknown Wave-4 representation: {representation}")
        self.hand_head = nn.Linear(int(config["decoder_channels"]), 3)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        output = self.backbone(batch)
        output["hand_logits"] = self.hand_head(output["decoder_features"])
        return output


def crossarch_loss_terms(
    criterion: nn.Module,
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    hand_weight: float,
) -> dict[str, torch.Tensor]:
    terms = base_formal500h_loss_terms(
        criterion, output, batch, hand_weight
    )
    zero = terms["loss"].new_zeros(())
    order_loss = zero
    soft_boundary_mse = zero
    if "representation_embedding" in output or "order_embedding" in output:
        if "representation_embedding" not in output or "order_embedding" not in output:
            raise ValueError("incomplete M2R2-style action-order contrastive outputs")
        representation = output["representation_embedding"]
        order = output["order_embedding"]
        if representation.shape != order.shape or representation.ndim != 2:
            raise ValueError("action-order embeddings must share shape [B,D]")
        logits = representation @ order.transpose(0, 1)
        logits = logits / ACTION_ORDER_TEMPERATURE
        labels = torch.arange(logits.shape[0], device=logits.device)
        order_loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.transpose(0, 1), labels)
        )
        if "soft_boundary_logits" not in output:
            raise ValueError("M2R2-style soft-boundary prediction is missing")
        target = criterion.boundary_target(batch)
        mask = batch["boundary_loss_mask"]
        squared = (torch.sigmoid(output["soft_boundary_logits"]) - target).square()
        soft_boundary_mse = (
            squared * mask.to(squared.dtype)
        ).sum() / mask.sum().clamp_min(1)
        terms["loss"] = (
            terms["loss"]
            + ACTION_ORDER_CONTRASTIVE_WEIGHT * order_loss
            + SOFT_BOUNDARY_MSE_WEIGHT * soft_boundary_mse
        )
    terms["action_order_contrastive"] = order_loss
    terms["soft_boundary_mse"] = soft_boundary_mse
    return terms


_ACTIVE_SPEC: CellSpec | None = None


def _evaluate_predictions(
    predictions,
    truth,
    *,
    boundary_threshold: float,
    boundary_min_distance_frames: int,
    min_segment_frames: int,
) -> dict[str, Any]:
    if _ACTIVE_SPEC is None:
        raise RuntimeError("Wave-4 evaluator called outside an active cell")
    result = evaluate_dataset(
        predictions,
        truth,
        num_actions=NUM_CLASSES,
        segment_iou_thresholds=(0.10, 0.25, 0.50),
        boundary_tolerances_ms=BOUNDARY_TOLERANCES_MS,
        primary_boundary_tolerance_ms=250,
        output_rate_hz=30.0,
        boundary_threshold=boundary_threshold,
        boundary_min_distance_frames=boundary_min_distance_frames,
        min_segment_frames=min_segment_frames,
        use_boundary_head=_ACTIVE_SPEC.use_boundary_head,
        include_action_changes=True,
        strict=True,
        collect_artifacts=False,
    )
    metrics = result.metrics
    metrics["wave4_common_metric_contract"] = {
        "boundary_tolerances_ms": list(BOUNDARY_TOLERANCES_MS),
        "m2r2_style_adapted_dr_field": "boundary/all/1000ms/f1",
        "m2r2_style_adapted_dr_official_reproduction": False,
        "test_manifest_read": False,
    }
    return metrics


def _release_lineage(spec: CellSpec) -> dict[str, Any]:
    module = Path(__file__).resolve()
    module_dir = module.parent
    sources = {
        "wave4_crossarch": module,
        "wave1_trainer": module_dir / "formal500h_wave1_train.py",
        "infra_runtime": module_dir / "formal500h_infra_runtime.py",
        "evaluator": module_dir / "formal500h_evaluation.py",
        "model": module_dir / "model.py",
        "baselines": module_dir / "baselines.py",
        "losses": module_dir / "losses.py",
    }
    source_records = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sources.items()
    }
    return {
        "schema_version": RUNTIME_SCHEMA,
        "release_manifest": str(module),
        "release_manifest_sha256": sha256_file(module),
        "sources": source_records,
        "cell": spec.as_dict(),
        "architecture_contract": _architecture_contract(spec),
        "architecture_fingerprint_sha256": architecture_fingerprint(spec),
        "dataset": "VariantAwareFormal500hDataset",
        "collate": "collate_formal500h_fast",
        "hdf5_cache_size": 8,
        "official_reproduction": False,
        "fidelity_statement": (
            "clean-room cached-feature adaptations; M2R2 omits original "
            "ActionCLIP/text/robot-modality stack and is not official"
        ),
    }


def _state_equal(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
) -> bool:
    return list(first) == list(second) and all(
        torch.equal(first[name], second[name]) for name in first
    )


def _max_output_difference(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
) -> float:
    return max(
        float((first[name].float() - second[name].float()).abs().max())
        for name in ("atomic_logits", "boundary_logits", "hand_logits")
    )


def _fixture_batch(*, with_targets: bool = False) -> dict[str, Any]:
    batch: dict[str, Any] = wave3b._fixture_batch()
    if with_targets:
        batch_size, steps = batch["sequence_mask"].shape
        target = torch.arange(steps).remainder(NUM_CLASSES)[None].expand(
            batch_size, -1
        ).clone()
        target[1] = torch.arange(steps).add(3).remainder(NUM_CLASSES)
        boundary = torch.zeros(batch_size, steps)
        boundary[:, 2] = 0.5
        boundary[:, 3] = 1.0
        batch.update(
            {
                "atomic_target": target,
                "hand_target": torch.zeros(batch_size, steps, dtype=torch.long),
                "subtask_target": torch.full(
                    (batch_size, steps), -100, dtype=torch.long
                ),
                "loss_mask": batch["sequence_mask"].clone(),
                "boundary_loss_mask": batch["sequence_mask"].clone(),
                "boundary_target": boundary,
            }
        )
    return batch


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def build_model_gate(*, output: Path) -> dict[str, Any]:
    module = Path(__file__).resolve()
    if output.is_file():
        existing = _read_json(output)
        checks = {
            "schema": existing.get("schema_version") == MODEL_GATE_SCHEMA,
            "pass": existing.get("status") == "pass",
            "trainer": existing.get("trainer_sha256") == sha256_file(module),
            "test_sealed": existing.get("test_manifest_read") is False,
            "nine_cells": set(existing.get("cells") or {}) == set(CELL_BY_ID),
        }
        if all(checks.values()):
            print(json.dumps({"event": "wave4_crossarch_model_gate_reused", **checks}))
            return existing
        raise ValueError(f"stale Wave-4 cross-architecture gate: {checks}")

    models: dict[str, CrossArchFormal500hModel] = {}
    counts: dict[str, int] = {}
    states: dict[str, Mapping[str, torch.Tensor]] = {}
    fingerprints: dict[str, str] = {}
    batch = _fixture_batch()
    outputs: dict[str, Mapping[str, torch.Tensor]] = {}
    for spec in CELL_SPECS:
        wave1._seed_everything(20260831)
        model = CrossArchFormal500hModel(_model_config(spec)).eval()
        models[spec.cell] = model
        counts[spec.cell] = sum(value.numel() for value in model.parameters())
        states[spec.cell] = model.state_dict()
        fingerprints[spec.cell] = architecture_fingerprint(spec)
        with torch.no_grad():
            outputs[spec.cell] = model(_clone_batch(batch))

    sensitivity: dict[str, float] = {}
    for spec in CELL_SPECS:
        changed = _clone_batch(batch)
        changed["tactile"] = changed["tactile"] + 7.0
        with torch.no_grad():
            altered = models[spec.cell](changed)
        sensitivity[spec.cell] = _max_output_difference(
            outputs[spec.cell], altered
        )

    diffact_eval_isolation = True
    for representation in REPRESENTATIONS:
        spec = CELL_BY_ID[f"DIFFACT_{representation}"]
        first = _fixture_batch(with_targets=True)
        second = _clone_batch(first)
        second["atomic_target"] = second["atomic_target"].flip(1)
        model = models[spec.cell].eval()
        with torch.no_grad():
            before = model(first)
            after = model(second)
        diffact_eval_isolation = diffact_eval_isolation and (
            _max_output_difference(before, after) == 0.0
        )

    m2r2_auxiliary = True
    for head in HEADS:
        model = models[f"{head}_M2R2"].train()
        torch.manual_seed(1147)
        output_value = model(_fixture_batch(with_targets=True))
        m2r2_auxiliary = m2r2_auxiliary and all(
            name in output_value
            for name in (
                "representation_embedding",
                "order_embedding",
                "soft_boundary_logits",
            )
        )
        model.eval()

    checks = {
        "three_by_three_cells": len(CELL_SPECS) == 9,
        "three_paired_seeds": len(SEEDS) == 3,
        "architecture_fingerprints_unique": len(set(fingerprints.values())) == 9,
        "all_outputs_finite": all(
            all(
                torch.isfinite(value[name]).all().item()
                for name in ("atomic_logits", "boundary_logits", "hand_logits")
            )
            for value in outputs.values()
        ),
        "output_shapes_match": all(
            value["atomic_logits"].shape == (2, 6, NUM_CLASSES)
            and value["boundary_logits"].shape == (2, 6)
            and value["hand_logits"].shape == (2, 6, 3)
            for value in outputs.values()
        ),
        "v_vt_parameter_matched_within_head": all(
            counts[f"{head}_V"] == counts[f"{head}_VT"] for head in HEADS
        ),
        "v_vt_initial_state_matched_within_head": all(
            _state_equal(states[f"{head}_V"], states[f"{head}_VT"])
            for head in HEADS
        ),
        "vision_cells_ignore_tactile": all(
            sensitivity[f"{head}_V"] == 0.0 for head in HEADS
        ),
        "vt_cells_use_tactile": all(
            sensitivity[f"{head}_VT"] > 0.0 for head in HEADS
        ),
        "m2r2_cells_use_tactile": all(
            sensitivity[f"{head}_M2R2"] > 0.0 for head in HEADS
        ),
        "asrf_has_learned_boundary": all(
            not torch.equal(
                outputs[f"ASRF_{representation}"]["boundary_logits"],
                torch.full_like(
                    outputs[f"ASRF_{representation}"]["boundary_logits"], -20.0
                ),
            )
            for representation in REPRESENTATIONS
        ),
        "nonboundary_heads_emit_fixed_boundary": all(
            torch.equal(
                outputs[f"{head}_{representation}"]["boundary_logits"],
                torch.full_like(
                    outputs[f"{head}_{representation}"]["boundary_logits"], -20.0
                ),
            )
            for head in ("MSTCNPP", "DIFFACT")
            for representation in REPRESENTATIONS
        ),
        "diffact_eval_target_isolation": diffact_eval_isolation,
        "m2r2_auxiliary_outputs_present_in_training": m2r2_auxiliary,
        "m2r2_contract_is_explicitly_adapted": all(
            _architecture_contract(CELL_BY_ID[f"{head}_M2R2"])[
                "representation"
            ]["official_m2r2_reproduction"]
            is False
            for head in HEADS
        ),
    }
    result = {
        "schema_version": MODEL_GATE_SCHEMA,
        "status": "pass" if all(checks.values()) else "fail",
        "purpose": "formal_wave4_cross_architecture_correctness_gate",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "test_selection_deferred": True,
        "trainer": str(module),
        "trainer_sha256": sha256_file(module),
        "heads": list(HEADS),
        "representations": list(REPRESENTATIONS),
        "seeds": list(SEEDS),
        "jobs": len(CELL_SPECS) * len(SEEDS),
        "cells": {spec.cell: spec.as_dict() for spec in CELL_SPECS},
        "architecture_fingerprints": fingerprints,
        "parameter_counts": counts,
        "initial_model_sha256": {
            name: wave1._model_state_sha256(model)
            for name, model in models.items()
        },
        "tactile_sensitivity_max_abs": sensitivity,
        "checks": checks,
    }
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result["status"] != "pass":
        raise RuntimeError(f"Wave-4 cross-architecture model gate failed: {checks}")
    return result


def train_cell(
    *,
    data_dir: Path,
    output_dir: Path,
    head: str,
    representation: str,
    seed: int,
    device: str = "cuda:0",
    precision: str = FROZEN_PRECISION,
    batch_size: int = FROZEN_BATCH_SIZE,
    num_workers: int = FROZEN_NUM_WORKERS,
    epochs: int = FROZEN_EPOCHS,
    learning_rate: float = FROZEN_LEARNING_RATE,
    calibration_workers: int = FROZEN_CALIBRATION_WORKERS,
) -> dict[str, Any]:
    head = head.upper()
    representation = representation.upper()
    cell = f"{head}_{representation}"
    if cell not in CELL_BY_ID or seed not in SEEDS:
        raise ValueError("Wave-4 requires one frozen head/representation/seed cell")
    spec = CELL_BY_ID[cell]
    if (
        precision != FROZEN_PRECISION
        or batch_size != FROZEN_BATCH_SIZE
        or num_workers != FROZEN_NUM_WORKERS
        or epochs != FROZEN_EPOCHS
        or not math.isclose(
            learning_rate, FROZEN_LEARNING_RATE, rel_tol=1.0e-12
        )
        or calibration_workers != FROZEN_CALIBRATION_WORKERS
    ):
        raise ValueError(
            "released Wave-4 cross-architecture recipe is bf16 batch=16 "
            "workers=20 epochs=3 lr=3e-4 calibration_workers=4"
        )

    originals = {
        "modalities": wave1.VARIANT_MODALITIES,
        "dataset": wave1._dataset,
        "collate": wave1.collate_formal500h,
        "release": wave1._implementation_release,
        "model_config": wave1._model_config,
        "loss_config": wave1._loss_config,
        "loss_terms": wave1.formal500h_loss_terms,
        "model": wave1.Formal500hModel,
        "evaluate": wave1._evaluate_predictions,
        "training_schema": wave1.TRAINING_SCHEMA,
    }

    def optimized_dataset(manifest: Path, statistics_path: Path):
        rows = wave1._read_jsonl(manifest)
        if any(row.get("schema_version") != WINDOW_SCHEMA for row in rows):
            raise ValueError("Wave-1 window manifest schema changed")
        return VariantAwareFormal500hDataset(
            manifest,
            statistics_path,
            expected_statistics_schema=WAVE1_DATA_SCHEMA,
            enabled_modalities=spec.modalities,
            hdf5_cache_size=8,
        )

    lineage = _release_lineage(spec)

    def released_implementation() -> dict[str, Any]:
        upstream = originals["release"]()
        return {
            **upstream,
            "formal_wave1_upstream": {
                "release_manifest": upstream["release_manifest"],
                "release_manifest_sha256": upstream["release_manifest_sha256"],
            },
            "release_manifest": lineage["release_manifest"],
            "release_manifest_sha256": lineage["release_manifest_sha256"],
            "formal_wave4_crossarch": lineage,
        }

    def model_config(variant: str, *, d_model: int, dropout: float):
        if variant != spec.cell:
            raise ValueError("Wave-4 cross-architecture variant dispatch changed")
        return _model_config(spec, d_model=d_model, dropout=dropout)

    def loss_config(statistics_path: Path):
        value = originals["loss_config"](statistics_path)
        value["boundary_weight"] = 0.5 if spec.use_boundary_head else 0.0
        value["action_order_contrastive_weight"] = (
            ACTION_ORDER_CONTRASTIVE_WEIGHT if spec.is_m2r2_style else 0.0
        )
        value["soft_boundary_mse_weight"] = (
            SOFT_BOUNDARY_MSE_WEIGHT if spec.is_m2r2_style else 0.0
        )
        value["action_order_temperature"] = ACTION_ORDER_TEMPERATURE
        return value

    global _ACTIVE_SPEC
    prior_active = _ACTIVE_SPEC
    _ACTIVE_SPEC = spec
    wave1.VARIANT_MODALITIES = {
        **originals["modalities"],
        spec.cell: spec.modalities,
    }
    wave1._dataset = optimized_dataset
    wave1.collate_formal500h = collate_formal500h_fast
    wave1._implementation_release = released_implementation
    wave1._model_config = model_config
    wave1._loss_config = loss_config
    wave1.formal500h_loss_terms = crossarch_loss_terms
    wave1.Formal500hModel = CrossArchFormal500hModel
    wave1._evaluate_predictions = _evaluate_predictions
    wave1.TRAINING_SCHEMA = TRAINING_SCHEMA
    try:
        return wave1.train_formal_variant(
            data_dir=data_dir,
            output_dir=output_dir,
            variant=spec.cell,
            seed=seed,
            device_name=device,
            precision=precision,
            batch_size=batch_size,
            num_workers=num_workers,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=FROZEN_WEIGHT_DECAY,
            d_model=128,
            dropout=0.15,
            hand_weight=FROZEN_HAND_WEIGHT,
            checkpoint_interval=FROZEN_CHECKPOINT_INTERVAL,
            calibration_workers=calibration_workers,
            resume=True,
        )
    finally:
        _ACTIVE_SPEC = prior_active
        wave1.VARIANT_MODALITIES = originals["modalities"]
        wave1._dataset = originals["dataset"]
        wave1.collate_formal500h = originals["collate"]
        wave1._implementation_release = originals["release"]
        wave1._model_config = originals["model_config"]
        wave1._loss_config = originals["loss_config"]
        wave1.formal500h_loss_terms = originals["loss_terms"]
        wave1.Formal500hModel = originals["model"]
        wave1._evaluate_predictions = originals["evaluate"]
        wave1.TRAINING_SCHEMA = originals["training_schema"]


def validate_run(
    output_dir: Path,
    *,
    head: str,
    representation: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cell = f"{head.upper()}_{representation.upper()}"
    if cell not in CELL_BY_ID or seed not in SEEDS:
        raise ValueError("invalid Wave-4 validation identity")
    spec = CELL_BY_ID[cell]
    summary_path = output_dir / "summary.json"
    complete_path = output_dir / "COMPLETE.json"
    summary = _read_json(summary_path)
    complete = _read_json(complete_path)
    resolved_path = Path(str(summary.get("resolved_config", "")))
    best_path = Path(str(summary.get("best_checkpoint", "")))
    calibration_path = Path(str(summary.get("validation_calibration", "")))
    if not all(path.is_file() for path in (resolved_path, best_path, calibration_path)):
        raise ValueError(f"missing Wave-4 run artifact: {output_dir}")
    resolved = _read_json(resolved_path)
    data_path = Path(str(resolved.get("data_summary", "")))
    data_summary = _read_json(data_path) if data_path.is_file() else {}
    model = resolved.get("model") or {}
    implementation = resolved.get("implementation_release") or {}
    crossarch = implementation.get("formal_wave4_crossarch") or {}
    selected = (summary.get("calibrated_validation") or {}).get("selected") or {}
    checks = {
        "schema": summary.get("schema_version") == TRAINING_SCHEMA
        and complete.get("schema_version") == TRAINING_SCHEMA,
        "complete": summary.get("status") == "complete"
        and complete.get("status") == "complete",
        "identity": summary.get("variant") == cell
        and int(summary.get("seed", -1)) == seed,
        "modalities": summary.get("enabled_modalities") == list(spec.modalities),
        "summary_hash": complete.get("summary_sha256") == sha256_file(summary_path),
        "best_hash": summary.get("best_checkpoint_sha256") == sha256_file(best_path)
        and complete.get("best_checkpoint_sha256") == sha256_file(best_path),
        "resolved_hash": summary.get("resolved_config_sha256")
        == sha256_file(resolved_path),
        "calibration_hash": summary.get("validation_calibration_sha256")
        == sha256_file(calibration_path),
        "architecture": model.get("head") == spec.head
        and model.get("representation") == spec.representation
        and model.get("architecture_fingerprint_sha256")
        == architecture_fingerprint(spec)
        and crossarch.get("architecture_fingerprint_sha256")
        == architecture_fingerprint(spec),
        "recipe": resolved.get("precision") == FROZEN_PRECISION
        and int(resolved.get("batch_size_per_gpu", -1)) == FROZEN_BATCH_SIZE
        and int(resolved.get("num_workers", -1)) == FROZEN_NUM_WORKERS
        and int(resolved.get("epochs", -1)) == FROZEN_EPOCHS
        and math.isclose(
            float(resolved.get("learning_rate", float("nan"))),
            FROZEN_LEARNING_RATE,
            rel_tol=1.0e-12,
        ),
        "calibration": len(
            (summary.get("calibrated_validation") or {}).get("candidates") or []
        )
        == 96
        and set(selected)
        >= {
            "boundary_threshold",
            "boundary_min_distance_frames",
            "min_segment_frames",
        },
        "data": data_summary.get("status") == "complete",
        "test_sealed": summary.get("test_manifest_read") is False
        and complete.get("test_manifest_read") is False
        and resolved.get("test_manifest_read") is False
        and data_summary.get("test_manifest_read") is False,
        "not_official": model.get("official_reproduction") is False
        and crossarch.get("official_reproduction") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid Wave-4 cross-architecture run {output_dir}: {checks}")
    return summary, resolved


def _mean_std(values: Sequence[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": list(values),
    }


METRICS: Mapping[str, Callable[[Mapping[str, Any]], float]] = {
    "primary_composite": lambda value: float(
        value["calibrated_validation"]["metrics"]["primary"][
            "checkpoint_composite"
        ]
    ),
    "collapsed_f1@10": lambda value: float(
        value["calibrated_validation"]["metrics"]["collapsed"]["f1@10"]
    ),
    "collapsed_f1@25": lambda value: float(
        value["calibrated_validation"]["metrics"]["collapsed"]["f1@25"]
    ),
    "collapsed_f1@50": lambda value: float(
        value["calibrated_validation"]["metrics"]["collapsed"]["f1@50"]
    ),
    "collapsed_edit": lambda value: float(
        value["calibrated_validation"]["metrics"]["collapsed"]["edit"]
    ),
    "class_change_bf1@250ms": lambda value: float(
        value["calibrated_validation"]["metrics"]["boundary"]["class_change"][
            "250ms"
        ]["f1"]
    ),
    "class_change_bf1@500ms": lambda value: float(
        value["calibrated_validation"]["metrics"]["boundary"]["class_change"][
            "500ms"
        ]["f1"]
    ),
    "class_change_bf1@1000ms": lambda value: float(
        value["calibrated_validation"]["metrics"]["boundary"]["class_change"][
            "1000ms"
        ]["f1"]
    ),
    "m2r2_style_adapted_dr_all_boundaries@1000ms": lambda value: float(
        value["calibrated_validation"]["metrics"]["boundary"]["all"][
            "1000ms"
        ]["f1"]
    ),
    "frame_accuracy": lambda value: float(
        value["calibrated_validation"]["metrics"]["frame_accuracy"]
    ),
    "tick_macro_f1": lambda value: float(
        value["calibrated_validation"]["metrics"]["tick_macro_f1"]
    ),
    "hand_macro_f1": lambda value: float(
        value["calibrated_validation"]["hand"]["macro_f1"]
    ),
}


def aggregate_crossarch(
    *, runs_root: Path, model_gate: Path, output: Path
) -> dict[str, Any]:
    gate = _read_json(model_gate)
    gate_checks = {
        "schema": gate.get("schema_version") == MODEL_GATE_SCHEMA,
        "pass": gate.get("status") == "pass",
        "test_sealed": gate.get("test_manifest_read") is False,
        "jobs": gate.get("jobs") == 27,
        "trainer": gate.get("trainer_sha256") == sha256_file(Path(__file__)),
    }
    if not all(gate_checks.values()):
        raise ValueError(f"invalid Wave-4 model gate: {gate_checks}")

    summaries: dict[str, dict[int, dict[str, Any]]] = {
        spec.cell: {} for spec in CELL_SPECS
    }
    resolved: dict[str, dict[int, dict[str, Any]]] = {
        spec.cell: {} for spec in CELL_SPECS
    }
    for spec in CELL_SPECS:
        for seed in SEEDS:
            run = runs_root / spec.head / spec.representation / f"seed_{seed}"
            summaries[spec.cell][seed], resolved[spec.cell][seed] = validate_run(
                run,
                head=spec.head,
                representation=spec.representation,
                seed=seed,
            )

    lineage = {
        (
            summary["data_summary_sha256"],
            summary["source_protocol_sha256"],
            summary["source_audit_sha256"],
            summary["release_manifest_sha256"],
        )
        for values in summaries.values()
        for summary in values.values()
    }
    if len(lineage) != 1:
        raise ValueError("Wave-4 runs do not share one frozen lineage")
    updates = {
        int(summary["updates"])
        for values in summaries.values()
        for summary in values.values()
    }
    if len(updates) != 1:
        raise ValueError("Wave-4 runs do not share one update budget")

    fingerprints = {
        spec.cell: resolved[spec.cell][SEEDS[0]]["model"][
            "architecture_fingerprint_sha256"
        ]
        for spec in CELL_SPECS
    }
    if fingerprints != gate.get("architecture_fingerprints"):
        raise ValueError("training architecture fingerprints differ from model gate")

    parameter_counts = {
        spec.cell: int(summaries[spec.cell][SEEDS[0]]["parameter_count"])
        for spec in CELL_SPECS
    }
    v_vt_parameter_matched = all(
        parameter_counts[f"{head}_V"] == parameter_counts[f"{head}_VT"]
        for head in HEADS
    )
    v_vt_initialization_matched = all(
        summaries[f"{head}_V"][seed]["initial_model_sha256"]
        == summaries[f"{head}_VT"][seed]["initial_model_sha256"]
        for head in HEADS
        for seed in SEEDS
    )
    if not v_vt_parameter_matched or not v_vt_initialization_matched:
        raise ValueError("V/VT cells lost parameter or initialization pairing")

    aggregate: dict[str, Any] = {}
    for spec in CELL_SPECS:
        aggregate[spec.cell] = {
            name: _mean_std(
                [getter(summaries[spec.cell][seed]) for seed in SEEDS]
            )
            for name, getter in METRICS.items()
        }

    paired: dict[str, Any] = {}
    for head in HEADS:
        for first, second in (("V", "VT"), ("M2R2", "VT")):
            contrast = f"{head}_{second}_minus_{first}"
            paired[contrast] = {}
            for name, getter in METRICS.items():
                differences = [
                    getter(summaries[f"{head}_{second}"][seed])
                    - getter(summaries[f"{head}_{first}"][seed])
                    for seed in SEEDS
                ]
                paired[contrast][name] = {
                    **_mean_std(differences),
                    "wins": sum(value > 0 for value in differences),
                }

    touch_consistency: dict[str, Any] = {}
    for name in METRICS:
        points = [
            value
            for head in HEADS
            for value in paired[f"{head}_VT_minus_V"][name]["values"]
        ]
        head_means = {
            head: paired[f"{head}_VT_minus_V"][name]["mean"] for head in HEADS
        }
        touch_consistency[name] = {
            **_mean_std(points),
            "wins": sum(value > 0 for value in points),
            "comparisons": len(points),
            "head_means": head_means,
            "positive_heads": sum(value > 0 for value in head_means.values()),
        }

    data_hash, protocol_hash, audit_hash, release_hash = next(iter(lineage))
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "complete",
        "purpose": "formal_wave4_cross_architecture_validation_only",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "test_selection_deferred": True,
        "official_reproduction": False,
        "heads": list(HEADS),
        "representations": list(REPRESENTATIONS),
        "representation_reporting_labels": {
            "V": "V",
            "VT": "D30B",
            "M2R2": "M2R2-style adapted",
        },
        "seeds": list(SEEDS),
        "jobs": len(CELL_SPECS) * len(SEEDS),
        "cells": {spec.cell: spec.as_dict() for spec in CELL_SPECS},
        "controls": {
            "same_train_validation_data": True,
            "same_30hz_full_episode_evaluator": True,
            "same_optimizer_update_budget": True,
            "updates_per_job": next(iter(updates)),
            "precision": FROZEN_PRECISION,
            "batch_size_per_gpu": FROZEN_BATCH_SIZE,
            "epochs": FROZEN_EPOCHS,
            "learning_rate": FROZEN_LEARNING_RATE,
            "weight_decay": FROZEN_WEIGHT_DECAY,
            "v_vt_parameter_matched_within_head": v_vt_parameter_matched,
            "v_vt_initialization_matched_within_head_and_seed": (
                v_vt_initialization_matched
            ),
            "m2r2_method_specific_auxiliary_supervision": [
                "action_order_symmetric_info_nce",
                "gaussian_soft_boundary_mse",
            ],
            "architecture_fingerprints": fingerprints,
            "parameter_counts": parameter_counts,
            "data_summary_sha256": data_hash,
            "source_protocol_sha256": protocol_hash,
            "source_audit_sha256": audit_hash,
            "release_manifest_sha256": release_hash,
            "model_gate": str(model_gate.resolve()),
            "model_gate_sha256": sha256_file(model_gate),
            "test_sealed": True,
        },
        "aggregate": aggregate,
        "paired": paired,
        "touch_cross_head_consistency": touch_consistency,
        "metric_contract": {
            "boundary_tolerances_ms": list(BOUNDARY_TOLERANCES_MS),
            "m2r2_style_adapted_dr": (
                "all annotated action boundaries, +/-1000ms, within-dataset "
                "adaptation; not an official M2R2 reproduction"
            ),
        },
        "decision_scope": (
            "validation-only evidence for cross-head tactile consistency and "
            "adapted representation comparison; formal test remains deferred"
        ),
    }
    _atomic_json(output, result)
    completion = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "complete",
        "aggregate": str(output.resolve()),
        "aggregate_sha256": sha256_file(output),
        "model_gate": str(model_gate.resolve()),
        "model_gate_sha256": sha256_file(model_gate),
        "jobs": 27,
        "test_manifest_read": False,
        "test_selection_deferred": True,
    }
    _atomic_json(output.parent / "WAVE4_CROSSARCH_COMPLETE.json", completion)
    print(
        json.dumps(
            {
                "event": "wave4_crossarch_aggregate_complete",
                "status": "complete",
                "jobs": 27,
                "test_manifest_read": False,
                "output": str(output.resolve()),
            }
        ),
        flush=True,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("model-gate")
    gate.add_argument("--output", type=Path, required=True)

    train = commands.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--head", choices=HEADS, required=True)
    train.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    train.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--precision", choices=(FROZEN_PRECISION,), default=FROZEN_PRECISION)
    train.add_argument("--batch-size", type=int, default=FROZEN_BATCH_SIZE)
    train.add_argument("--num-workers", type=int, default=FROZEN_NUM_WORKERS)
    train.add_argument("--epochs", type=int, default=FROZEN_EPOCHS)
    train.add_argument("--learning-rate", type=float, default=FROZEN_LEARNING_RATE)
    train.add_argument(
        "--calibration-workers", type=int, default=FROZEN_CALIBRATION_WORKERS
    )

    validate = commands.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--head", choices=HEADS, required=True)
    validate.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    validate.add_argument("--seed", type=int, choices=SEEDS, required=True)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--runs-root", type=Path, required=True)
    aggregate.add_argument("--model-gate", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "model-gate":
        build_model_gate(output=arguments.output)
    elif arguments.command == "train":
        train_cell(
            data_dir=arguments.data_dir,
            output_dir=arguments.output_dir,
            head=arguments.head,
            representation=arguments.representation,
            seed=arguments.seed,
            device=arguments.device,
            precision=arguments.precision,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            calibration_workers=arguments.calibration_workers,
        )
    elif arguments.command == "validate":
        summary, _ = validate_run(
            arguments.output_dir,
            head=arguments.head,
            representation=arguments.representation,
            seed=arguments.seed,
        )
        print(
            json.dumps(
                {
                    "event": "wave4_crossarch_run_valid",
                    "cell": f"{arguments.head}_{arguments.representation}",
                    "seed": arguments.seed,
                    "summary": str((arguments.output_dir / "summary.json").resolve()),
                    "primary": summary["calibrated_validation"]["metrics"]["primary"],
                    "test_manifest_read": False,
                }
            )
        )
    elif arguments.command == "aggregate":
        aggregate_crossarch(
            runs_root=arguments.runs_root,
            model_gate=arguments.model_gate,
            output=arguments.output,
        )
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
