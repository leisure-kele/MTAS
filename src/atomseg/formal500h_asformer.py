"""Formal500h ASFormer-style cross-architecture experiment.

This module is a small, clean-room ASFormer-style temporal head for the
frozen Formal500h representation and evaluator contract.  It deliberately
does not modify :mod:`formal500h_wave4_crossarch`, whose nine-cell release is
already sealed.  The experiment contains two paired cells (vision-only and
vision+tactile), each trained with the three frozen Formal500h seeds.

The head uses local self-attention with sinusoidal positions and a second
refinement stage that consumes detached first-stage action probabilities.  It
is an adapted temporal-segmentation baseline, not an official reproduction
of the ASFormer repository.  The formal test manifest is never opened.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from . import formal500h_wave1_train as wave1
from .baselines import EarlyFusionContactTAS
from .formal500h_evaluation import GroundTruthEpisode, evaluate_dataset
from .formal500h_infra_runtime import (
    VariantAwareFormal500hDataset,
    collate_formal500h_fast,
)
from .formal500h_protocol import (
    FORMAL_EVALUATION_POLICY,
    NUM_CLASSES,
    sha256_file,
)
from .formal500h_wave1_data import WAVE1_DATA_SCHEMA, WINDOW_SCHEMA


TRAINING_SCHEMA = "formal500h_asformer_training_v1"
RELEASE_SCHEMA = "formal500h_asformer_release_v1"
MODEL_GATE_SCHEMA = "formal500h_asformer_model_gate_v1"
AGGREGATE_SCHEMA = "formal500h_asformer_aggregate_v1"
COMPLETE_SCHEMA = "formal500h_asformer_complete_v1"

VARIANTS = ("ASFORMER_V", "ASFORMER_VT")
REPRESENTATIONS = ("V", "VT")
SEEDS = (20260824, 20260825, 20260826)
MODALITIES = {
    "ASFORMER_V": ("vision",),
    "ASFORMER_VT": ("vision", "tactile"),
}

FROZEN_PRECISION = "bf16"
FROZEN_BATCH_SIZE = 16
FROZEN_NUM_WORKERS = 20
FROZEN_EPOCHS = 3
FROZEN_LEARNING_RATE = 3.0e-4
FROZEN_WEIGHT_DECAY = 1.0e-4
# The formal cross-architecture comparison follows the project-wide H0
# protocol: hand labels remain available for diagnostics, but the auxiliary
# hand objective is disabled so it cannot alter action/boundary gradients.
FROZEN_HAND_WEIGHT = 0.0
FROZEN_CHECKPOINT_INTERVAL = 2000
FROZEN_CALIBRATION_WORKERS = 4
ASFORMER_LAYERS = 4
ASFORMER_HEADS = 8
ASFORMER_LOCAL_RADIUS = 32
ASFORMER_FFN_MULTIPLIER = 4

_ACTIVE_VARIANT: str | None = None
_ACTIVE_RELEASE: dict[str, Any] | None = None
_ASFORMER_RELEASE_MANIFEST: Path | None = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _model_config(variant: str, *, d_model: int, dropout: float) -> dict[str, Any]:
    variant = variant.upper()
    if variant not in MODALITIES:
        raise ValueError(f"unknown ASFormer variant: {variant}")
    representation = "VT" if variant.endswith("_VT") else "V"
    return {
        # EarlyFusionContactTAS is reused only for the frozen modality
        # encoders; its decoder is replaced by ASFormerDecoder below.
        "variant": "b2_early_fusion",
        "formal_variant": variant,
        "head": "ASFORMER",
        "representation": representation,
        "enabled_modalities": list(MODALITIES[variant]),
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
        "decoder_layers": ASFORMER_LAYERS,
        "decoder_stages": 2,
        "asformer_heads": ASFORMER_HEADS,
        "asformer_local_radius": ASFORMER_LOCAL_RADIUS,
        "asformer_ffn_multiplier": ASFORMER_FFN_MULTIPLIER,
        "num_actions": NUM_CLASSES,
        "num_subtasks": 0,
        # ASFormer-style baseline has no explicit boundary branch.  Common
        # evaluation obtains boundaries from action changes.
        "use_boundary_head": False,
        "output_rate_hz": 30,
        "event_rate_hz": 30,
        "hand_weight": FROZEN_HAND_WEIGHT,
        "hand_loss_enabled": False,
        "official_reproduction": False,
    }


class SinusoidalPosition(nn.Module):
    """Deterministic positional encoding for a variable-length tick sequence."""

    def __init__(self, dimension: int, maximum_length: int = 4096) -> None:
        super().__init__()
        if dimension < 2 or maximum_length < 1:
            raise ValueError("invalid positional-encoding dimensions")
        positions = torch.arange(maximum_length, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(maximum_length, dimension, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding[None], persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[1] > self.encoding.shape[1]:
            raise ValueError("ASFormer positional input has an unsupported shape")
        return value + self.encoding[:, : value.shape[1]].to(dtype=value.dtype)


class LocalTransformerBlock(nn.Module):
    """Pre-norm Transformer block with a finite local attention prior."""

    def __init__(
        self,
        dimension: int,
        heads: int,
        local_radius: int,
        dropout: float,
        ffn_multiplier: int,
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("ASFormer dimension must be divisible by attention heads")
        if local_radius < 1:
            raise ValueError("ASFormer local radius must be positive")
        self.local_radius = int(local_radius)
        self.norm_attention = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_feed_forward = nn.LayerNorm(dimension)
        hidden = dimension * int(ffn_multiplier)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.Dropout(dropout),
        )

    def _attention_mask(self, steps: int, device: torch.device) -> torch.Tensor:
        indices = torch.arange(steps, device=device)
        # ``True`` entries are disallowed by MultiheadAttention.  A local
        # symmetric window is the ASFormer-style inductive bias; the model is
        # offline, so this is intentionally not causal.
        return (indices[:, None] - indices[None, :]).abs() > self.local_radius

    def forward(self, value: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        if sequence_mask.ndim != 2 or sequence_mask.shape != value.shape[:2]:
            raise ValueError("ASFormer sequence_mask must have shape [batch, steps]")
        sequence_mask = sequence_mask.to(dtype=torch.bool, device=value.device)
        normalized = self.norm_attention(value)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=self._attention_mask(value.shape[1], value.device),
            # Padding must not participate as a key/value.  The previous
            # implementation only zeroed outputs after attention, which let
            # padded tokens contaminate valid queries near the sequence end.
            key_padding_mask=~sequence_mask,
            need_weights=False,
        )
        value = value + attended
        value = value + self.feed_forward(self.norm_feed_forward(value))
        return value * sequence_mask[..., None].to(dtype=value.dtype)


class ASFormerStage(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        dimension: int,
        layers: int,
        heads: int,
        local_radius: int,
        dropout: float,
        ffn_multiplier: int,
        num_actions: int,
    ) -> None:
        super().__init__()
        self.input_projection = (
            nn.Linear(input_dimension, dimension)
            if input_dimension != dimension
            else nn.Identity()
        )
        self.position = SinusoidalPosition(dimension)
        self.blocks = nn.ModuleList(
            LocalTransformerBlock(
                dimension,
                heads,
                local_radius,
                dropout,
                ffn_multiplier,
            )
            for _ in range(layers)
        )
        self.action_head = nn.Linear(dimension, num_actions)

    def forward(
        self, sequence: torch.Tensor, sequence_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.position(self.input_projection(sequence))
        value = value * sequence_mask[..., None].to(dtype=value.dtype)
        for block in self.blocks:
            value = block(value, sequence_mask)
        return self.action_head(value), value


class ASFormerDecoder(nn.Module):
    """Two-stage ASFormer-style action decoder.

    The first stage predicts action logits.  Each refinement stage receives
    the previous stage's detached probabilities, matching the common
    multi-stage TAS training contract while keeping inference label-free.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dimension = int(config["d_model"])
        channels = int(config["decoder_channels"])
        layers = int(config.get("decoder_layers", ASFORMER_LAYERS))
        stages = int(config.get("decoder_stages", 2))
        heads = int(config.get("asformer_heads", ASFORMER_HEADS))
        radius = int(config.get("asformer_local_radius", ASFORMER_LOCAL_RADIUS))
        multiplier = int(config.get("asformer_ffn_multiplier", ASFORMER_FFN_MULTIPLIER))
        dropout = float(config["dropout"])
        actions = int(config["num_actions"])
        if stages < 1:
            raise ValueError("ASFormer requires at least one stage")
        self.num_actions = actions
        self.first = ASFormerStage(
            dimension,
            channels,
            layers,
            heads,
            radius,
            dropout,
            multiplier,
            actions,
        )
        self.refinements = nn.ModuleList(
            ASFormerStage(
                channels + actions,
                channels,
                layers,
                heads,
                radius,
                dropout,
                multiplier,
                actions,
            )
            for _ in range(stages - 1)
        )

    def forward(
        self,
        sequence: torch.Tensor,
        sequence_mask: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        # ``target`` is intentionally ignored in both train/eval forward; the
        # action target is consumed only by the loss, so validation cannot
        # accidentally read labels through the decoder.
        del target
        logits, features = self.first(sequence, sequence_mask)
        action_stages = [logits]
        boundary_stages = [logits.new_full(logits.shape[:2], -20.0)]
        for refinement in self.refinements:
            probabilities = torch.softmax(logits, dim=-1).detach()
            logits, features = refinement(
                torch.cat([features, probabilities], dim=-1), sequence_mask
            )
            action_stages.append(logits)
            boundary_stages.append(logits.new_full(logits.shape[:2], -20.0))
        return {
            "atomic_logits": action_stages[-1],
            "boundary_logits": boundary_stages[-1],
            "atomic_stages": action_stages,
            "boundary_stages": boundary_stages,
            "decoder_features": features,
        }


class ASFormerFormal500hModel(nn.Module):
    """Frozen Formal500h input encoder plus the ASFormer-style temporal head."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.encoder = EarlyFusionContactTAS(dict(config))
        self.encoder.decoder = ASFormerDecoder(config)
        self.hand_head = nn.Linear(int(config["decoder_channels"]), 3)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        encoded = self.encoder._encode_modalities(batch, False)
        sequence_mask = encoded["sequence_mask"]
        vision = encoded["vision_token"] * encoded["vision_quality"][..., None]
        skeleton = encoded["skeleton_token"] * encoded["skeleton_quality"][..., None]
        tactile = encoded["tactile_token"] * encoded["tactile_quality"][..., None]
        kinematics = encoded["kinematic_token"] * encoded["kinematic_quality"][..., None]
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
        fused = fused * sequence_mask[..., None].to(dtype=fused.dtype)
        output = self.encoder.decoder(fused, sequence_mask)
        output["hand_logits"] = self.hand_head(output["decoder_features"])
        output["reliability_weights"] = torch.stack(
            [encoded["semantic_quality"], encoded["event_quality"]], dim=-1
        )
        output["view_weights"] = encoded["view_weights"]
        return output


def _dataset(manifest: Path, statistics_path: Path) -> VariantAwareFormal500hDataset:
    if _ACTIVE_VARIANT is None:
        raise RuntimeError("ASFormer dataset requested outside an active job")
    rows = _read_jsonl(manifest)
    if any(row.get("schema_version") != WINDOW_SCHEMA for row in rows):
        raise ValueError("Formal500h window manifest schema changed")
    return VariantAwareFormal500hDataset(
        manifest,
        statistics_path,
        expected_statistics_schema=WAVE1_DATA_SCHEMA,
        enabled_modalities=MODALITIES[_ACTIVE_VARIANT],
        hdf5_cache_size=8,
    )


def _loss_config(statistics_path: Path) -> dict[str, Any]:
    if _ACTIVE_VARIANT is None:
        raise RuntimeError("ASFormer loss config requested outside an active job")
    value = wave1.formal500h_class_weights(statistics_path)
    return {
        "action_weight": 1.0,
        "smooth_weight": 0.1,
        # No learned boundary head is used by this ASFormer-style adapter;
        # action-change boundaries are decoded by the common evaluator.
        "boundary_weight": 0.0,
        "subtask_weight": 0.0,
        "boundary_positive_weight": 8.0,
        "action_class_weights": value,
    }


def _evaluate_predictions(
    predictions: Mapping[str, Any],
    truth: Mapping[str, GroundTruthEpisode],
    *,
    boundary_threshold: float,
    boundary_min_distance_frames: int,
    min_segment_frames: int,
) -> dict[str, Any]:
    if _ACTIVE_VARIANT is None:
        raise RuntimeError("ASFormer evaluator called outside an active cell")
    result = evaluate_dataset(
        predictions,
        truth,
        num_actions=NUM_CLASSES,
        segment_iou_thresholds=(0.10, 0.25, 0.50),
        boundary_tolerances_ms=(50, 100, 250, 500, 1000),
        primary_boundary_tolerance_ms=250,
        output_rate_hz=30.0,
        boundary_threshold=boundary_threshold,
        boundary_min_distance_frames=boundary_min_distance_frames,
        min_segment_frames=min_segment_frames,
        use_boundary_head=False,
        include_action_changes=True,
        strict=True,
        collect_artifacts=False,
    )
    metrics = result.metrics
    metrics["asformer_metric_contract"] = {
        "boundary_tolerances_ms": [50, 100, 250, 500, 1000],
        "boundary_source": "decoded_action_change_only",
        "test_manifest_read": False,
    }
    return metrics


def _source_tree() -> dict[str, Path]:
    module = Path(__file__).resolve()
    directory = module.parent
    return {
        # Keep the package boundary and the data-contract module in the
        # closure.  Both can change import-time behaviour even though they
        # are not imported with a ``from`` statement in this file.
        "package_init": directory / "__init__.py",
        "asformer_trainer": module,
        "crossarch_fixture": directory / "formal500h_wave4_crossarch.py",
        "wave3b_fixture": directory / "formal500h_wave3b_vt.py",
        "wave1_trainer": directory / "formal500h_wave1_train.py",
        "wave1_data": directory / "formal500h_wave1_data.py",
        "clock": directory / "formal500h_clock.py",
        "runtime": directory / "formal500h_runtime.py",
        "infra_runtime": directory / "formal500h_infra_runtime.py",
        "evaluator": directory / "formal500h_evaluation.py",
        "model": directory / "model.py",
        "baselines": directory / "baselines.py",
        "losses": directory / "losses.py",
        "inference": directory / "inference.py",
        "protocol": directory / "formal500h_protocol.py",
        "training_release": directory / "formal500h_training_release.py",
        "h0_contract": directory / "formal500h_h0_contract.py",
    }


def _source_records() -> dict[str, dict[str, str]]:
    """Return the complete ASFormer source closure with content digests."""

    paths = _source_tree()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        details = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise ValueError(f"ASFormer source tree is incomplete: {details}")
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    }


def _source_tree_sha256(records: Mapping[str, Mapping[str, str]]) -> str:
    """Hash source names and digests, independent of mount-point spelling."""

    return _canonical_sha(
        {name: str(record["sha256"]) for name, record in sorted(records.items())}
    )


def _implementation_release() -> dict[str, Any]:
    module = Path(__file__).resolve()
    package_root = module.parents[2]
    # H0 launchers provide a private checksum manifest.  Retain the old path
    # only as a compatibility fallback for historical artifacts; an H0 run
    # must record the manifest that actually gated its source tree.
    manifest = _ASFORMER_RELEASE_MANIFEST
    if manifest is None:
        configured = os.environ.get("ASFORMER_RELEASE_MANIFEST", "").strip()
        manifest = (
            Path(configured).expanduser().resolve()
            if configured
            else package_root
            / "configs"
            / "experimental"
            / "atomic_v1_asformer_release_sha256.txt"
        )
    if not manifest.is_file():
        raise ValueError(f"ASFormer release manifest is missing: {manifest}")
    source_records = _source_records()
    return {
        "schema_version": TRAINING_SCHEMA,
        "release_manifest": str(manifest),
        "release_manifest_sha256": sha256_file(manifest),
        "source_tree_sha256": _source_tree_sha256(source_records),
        "sources": source_records,
        "head": "ASFORMER-style",
        "official_reproduction": False,
        "test_manifest_read": False,
    }


def build_release(*, data_dir: Path, output: Path) -> dict[str, Any]:
    """Freeze the data lineage and ASFormer recipe before training."""
    implementation = _implementation_release()
    summary, artifacts = wave1._validated_data(data_dir)
    if summary.get("test_manifest_read") is not False:
        raise ValueError("source Formal500h data release is not test-sealed")
    if len(_read_jsonl(artifacts["train_windows"])) < 1 or len(_read_jsonl(artifacts["val_windows"])) < 1:
        raise ValueError("ASFormer release requires non-empty train/validation windows")
    checks = {
        "source_data_complete": summary.get("status") == "complete",
        "test_sealed": summary.get("test_manifest_read") is False,
        "train_windows_nonempty": True,
        "val_windows_nonempty": True,
        "num_actions_26": NUM_CLASSES == 26,
        "recipe_frozen": True,
    }
    result = {
        "schema_version": RELEASE_SCHEMA,
        "status": "complete" if all(checks.values()) else "fail",
        "purpose": "freeze_asformer_style_validation_experiment_lineage",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "data": {
            "summary": str((data_dir / "summary.json").resolve()),
            "summary_sha256": sha256_file(data_dir / "summary.json"),
            "source_protocol_sha256": summary["source_protocol_sha256"],
            "source_audit_sha256": summary["source_audit_sha256"],
            "train_windows": len(_read_jsonl(artifacts["train_windows"])),
            "val_windows": len(_read_jsonl(artifacts["val_windows"])),
        },
        "architecture": {
            "head": "ASFORMER-style",
            "local_attention_radius_ticks": ASFORMER_LOCAL_RADIUS,
            "layers_per_stage": ASFORMER_LAYERS,
            "attention_heads": ASFORMER_HEADS,
            "stages": 2,
            "boundary_source": "decoded_action_change_only",
            "official_reproduction": False,
        },
        "recipe": {
            "precision": FROZEN_PRECISION,
            "batch_size_per_gpu": FROZEN_BATCH_SIZE,
            "num_workers": FROZEN_NUM_WORKERS,
            "epochs": FROZEN_EPOCHS,
            "learning_rate": FROZEN_LEARNING_RATE,
            "weight_decay": FROZEN_WEIGHT_DECAY,
            "hand_weight": FROZEN_HAND_WEIGHT,
            "calibration_workers": FROZEN_CALIBRATION_WORKERS,
        },
        # The release is an immutable claim about executable code as well as
        # data.  Keep the nested record for lineage consumers and the aliases
        # for simple shell/JSON audits.
        "implementation": implementation,
        "source_tree_sha256": implementation["source_tree_sha256"],
        "sources": implementation["sources"],
        "release_manifest_sha256": implementation["release_manifest_sha256"],
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "checks": checks,
    }
    _atomic_json(output, result)
    if result["status"] != "complete":
        raise RuntimeError(f"ASFormer release audit failed: {checks}")
    print(json.dumps({"event": "asformer_release_complete", "output": str(output.resolve()), "test_manifest_read": False}), flush=True)
    return result


def _validated_release(release_path: Path, data_dir: Path) -> dict[str, Any]:
    release = _read_json(release_path)
    summary, _ = wave1._validated_data(data_dir)
    source = release.get("data") or {}
    implementation = release.get("implementation")
    current_implementation = _implementation_release()
    recorded_sources = (implementation or {}).get("sources") if isinstance(implementation, Mapping) else None
    if not isinstance(recorded_sources, Mapping):
        recorded_sources = {}
    checks = {
        "schema": release.get("schema_version") == RELEASE_SCHEMA,
        "complete": release.get("status") == "complete",
        "test_sealed": release.get("test_manifest_read") is False,
        "data_path": Path(str(source.get("summary", ""))).resolve() == (data_dir / "summary.json").resolve(),
        "data_hash": source.get("summary_sha256") == sha256_file(data_dir / "summary.json"),
        "protocol_hash": source.get("source_protocol_sha256") == summary.get("source_protocol_sha256"),
        "audit_hash": source.get("source_audit_sha256") == summary.get("source_audit_sha256"),
        "recipe": release.get("recipe", {}).get("batch_size_per_gpu") == FROZEN_BATCH_SIZE
        and release.get("recipe", {}).get("num_workers") == FROZEN_NUM_WORKERS
        and release.get("recipe", {}).get("epochs") == FROZEN_EPOCHS
        and math.isclose(float(release.get("recipe", {}).get("learning_rate", float("nan"))), FROZEN_LEARNING_RATE, rel_tol=1e-12)
        and release.get("recipe", {}).get("hand_weight") == FROZEN_HAND_WEIGHT,
        "implementation": isinstance(implementation, Mapping)
        and implementation.get("schema_version") == TRAINING_SCHEMA
        and implementation.get("release_manifest_sha256")
        == current_implementation["release_manifest_sha256"]
        and implementation.get("source_tree_sha256")
        == current_implementation["source_tree_sha256"]
        and release.get("source_tree_sha256") == current_implementation["source_tree_sha256"]
        and release.get("release_manifest_sha256")
        == current_implementation["release_manifest_sha256"]
        and release.get("sources") == recorded_sources
        and set(recorded_sources) == set(current_implementation["sources"]),
    }
    for name, current in current_implementation["sources"].items():
        recorded = recorded_sources.get(name)
        checks[f"source:{name}"] = (
            isinstance(recorded, Mapping)
            and recorded.get("sha256") == current.get("sha256")
        )
    if not all(checks.values()):
        raise ValueError(f"stale or incompatible ASFormer release: {checks}")
    return release


def train_cell(
    *,
    data_dir: Path,
    release: Path,
    output_dir: Path,
    variant: str,
    seed: int,
    device: str = "cuda:0",
    precision: str = FROZEN_PRECISION,
    batch_size: int = FROZEN_BATCH_SIZE,
    num_workers: int = FROZEN_NUM_WORKERS,
    epochs: int = FROZEN_EPOCHS,
    learning_rate: float = FROZEN_LEARNING_RATE,
    calibration_workers: int = FROZEN_CALIBRATION_WORKERS,
) -> dict[str, Any]:
    global _ACTIVE_VARIANT, _ACTIVE_RELEASE, _ASFORMER_RELEASE_MANIFEST
    variant = variant.upper()
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("invalid ASFormer variant or seed")
    if (precision, batch_size, num_workers, epochs) != (
        FROZEN_PRECISION,
        FROZEN_BATCH_SIZE,
        FROZEN_NUM_WORKERS,
        FROZEN_EPOCHS,
    ) or not math.isclose(learning_rate, FROZEN_LEARNING_RATE, rel_tol=1e-12):
        raise ValueError("ASFormer recipe is bf16, batch=16, workers=20, epochs=3, lr=3e-4")
    if calibration_workers != FROZEN_CALIBRATION_WORKERS:
        raise ValueError("ASFormer calibration_workers is frozen at 4")
    _ACTIVE_RELEASE = _validated_release(release, data_dir)
    prior = {
        "modalities": wave1.VARIANT_MODALITIES,
        "dataset": wave1._dataset,
        "collate": wave1.collate_formal500h,
        "implementation": wave1._implementation_release,
        "model_config": wave1._model_config,
        "loss_config": wave1._loss_config,
        "model": wave1.Formal500hModel,
        "ground_truth": wave1._ground_truth,
        "evaluate": wave1._evaluate_predictions,
        "schema": wave1.TRAINING_SCHEMA,
    }
    prior_manifest = _ASFORMER_RELEASE_MANIFEST
    configured_manifest = os.environ.get("ASFORMER_RELEASE_MANIFEST", "").strip()
    _ASFORMER_RELEASE_MANIFEST = (
        Path(configured_manifest).expanduser().resolve()
        if configured_manifest
        else prior_manifest
    )
    _ACTIVE_VARIANT = variant
    wave1.VARIANT_MODALITIES = {**prior["modalities"], variant: MODALITIES[variant]}
    wave1._dataset = _dataset
    wave1.collate_formal500h = collate_formal500h_fast
    wave1._implementation_release = _implementation_release
    wave1._model_config = lambda value, *, d_model, dropout: _model_config(value, d_model=d_model, dropout=dropout)
    wave1._loss_config = _loss_config
    wave1.Formal500hModel = ASFormerFormal500hModel
    wave1._evaluate_predictions = _evaluate_predictions
    wave1.TRAINING_SCHEMA = TRAINING_SCHEMA
    try:
        return wave1.train_formal_variant(
            data_dir=data_dir,
            output_dir=output_dir,
            variant=variant,
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
        wave1.VARIANT_MODALITIES = prior["modalities"]
        wave1._dataset = prior["dataset"]
        wave1.collate_formal500h = prior["collate"]
        wave1._implementation_release = prior["implementation"]
        wave1._model_config = prior["model_config"]
        wave1._loss_config = prior["loss_config"]
        wave1.Formal500hModel = prior["model"]
        wave1._ground_truth = prior["ground_truth"]
        wave1._evaluate_predictions = prior["evaluate"]
        wave1.TRAINING_SCHEMA = prior["schema"]
        _ASFORMER_RELEASE_MANIFEST = prior_manifest
        _ACTIVE_VARIANT, _ACTIVE_RELEASE = None, None


def _metric(summary: Mapping[str, Any], name: str) -> float:
    metrics = (summary.get("calibrated_validation") or {}).get("metrics") or {}
    if name == "primary_composite":
        return float((metrics.get("primary") or {})["checkpoint_composite"])
    if name == "collapsed_f1@50":
        return float((metrics.get("primary") or {})["collapsed_semantic_f1@50"])
    if name == "boundary_f1@250":
        return float((metrics.get("primary") or {})["class_change_boundary_f1@250ms"])
    if name == "boundary_f1@1000":
        return float(metrics["boundary"]["class_change"]["1000ms"]["f1"])
    if name == "frame_accuracy":
        return float(metrics["frame_accuracy"])
    if name == "tick_macro_f1":
        return float(metrics["tick_macro_f1"])
    raise KeyError(name)


METRICS = (
    "primary_composite",
    "collapsed_f1@50",
    "boundary_f1@250",
    "boundary_f1@1000",
    "frame_accuracy",
    "tick_macro_f1",
)


def _mean_std(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise ValueError("cannot aggregate empty/non-finite ASFormer metrics")
    return {
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "values": numbers,
    }


def _validate_run(path: Path, *, variant: str, seed: int) -> dict[str, Any]:
    summary_path = path / "summary.json"
    summary = _read_json(summary_path)
    complete = _read_json(path / "COMPLETE.json")
    resolved_path = Path(str(summary.get("resolved_config", "")))
    best_path = Path(str(summary.get("best_checkpoint", "")))
    calibration_path = Path(str(summary.get("validation_calibration", "")))
    checks = {
        "identity": summary.get("schema_version") == TRAINING_SCHEMA
        and summary.get("status") == "complete"
        and summary.get("variant") == variant
        and int(summary.get("seed", -1)) == seed
        and summary.get("enabled_modalities") == list(MODALITIES[variant]),
        "complete": complete.get("status") == "complete"
        and complete.get("schema_version") == TRAINING_SCHEMA
        and complete.get("summary_sha256") == sha256_file(summary_path),
        "test_sealed": summary.get("test_manifest_read") is False
        and complete.get("test_manifest_read") is False,
        "artifacts": all(item.is_file() for item in (resolved_path, best_path, calibration_path)),
        "hashes": summary.get("resolved_config_sha256") == sha256_file(resolved_path)
        and summary.get("best_checkpoint_sha256") == sha256_file(best_path)
        and summary.get("validation_calibration_sha256") == sha256_file(calibration_path),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid ASFormer run {path}: {checks}")
    resolved = _read_json(resolved_path)
    model = resolved.get("model") or {}
    recipe = {
        "precision": resolved.get("precision") == FROZEN_PRECISION,
        "batch": int(resolved.get("batch_size_per_gpu", -1)) == FROZEN_BATCH_SIZE,
        "workers": int(resolved.get("num_workers", -1)) == FROZEN_NUM_WORKERS,
        "epochs": int(resolved.get("epochs", -1)) == FROZEN_EPOCHS,
        "lr": math.isclose(float(resolved.get("learning_rate", float("nan"))), FROZEN_LEARNING_RATE, rel_tol=1e-12),
        "hand": math.isclose(float(resolved.get("hand_weight", float("nan"))), FROZEN_HAND_WEIGHT, abs_tol=1e-12),
    }
    architecture = model.get("head") == "ASFORMER" and model.get("representation") == ("VT" if variant.endswith("_VT") else "V")
    if not all(recipe.values()) or not architecture:
        raise ValueError(f"invalid ASFormer run recipe/architecture: {path}")
    implementation = resolved.get("implementation_release") or {}
    current_implementation = _implementation_release()
    recorded_sources = implementation.get("sources")
    if not isinstance(recorded_sources, Mapping):
        recorded_sources = {}
    source_checks = {
        "schema": implementation.get("schema_version") == TRAINING_SCHEMA,
        "manifest": implementation.get("release_manifest_sha256")
        == current_implementation["release_manifest_sha256"],
        "tree": implementation.get("source_tree_sha256")
        == current_implementation["source_tree_sha256"],
        "keys": set(recorded_sources) == set(current_implementation["sources"]),
    }
    for name, current in current_implementation["sources"].items():
        recorded = recorded_sources.get(name)
        source_checks[f"source:{name}"] = (
            isinstance(recorded, Mapping)
            and recorded.get("sha256") == current.get("sha256")
        )
    if not all(source_checks.values()):
        raise ValueError(f"stale ASFormer implementation source tree: {path}: {source_checks}")
    return summary


def aggregate(*, runs_root: Path, release: Path, output: Path) -> dict[str, Any]:
    release_value = _read_json(release)
    release_data = release_value.get("data") or {}
    release_summary = Path(str(release_data.get("summary", ""))).resolve()
    if not release_summary.is_file():
        raise ValueError("ASFormer aggregate requires a release with a readable data summary")
    # Re-run the full release validator here.  This prevents a stale release
    # or a source edit between training and aggregation from being silently
    # promoted to the paper-facing aggregate.
    _validated_release(release, release_summary.parent)
    summaries: dict[str, dict[int, dict[str, Any]]] = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for seed in SEEDS:
            summaries[variant][seed] = _validate_run(runs_root / variant / f"seed_{seed}", variant=variant, seed=seed)
    lineage = {
        (
            value.get("data_summary_sha256"),
            value.get("source_protocol_sha256"),
            value.get("source_audit_sha256"),
            value.get("release_manifest_sha256"),
        )
        for values in summaries.values() for value in values.values()
    }
    if len(lineage) != 1:
        raise ValueError("ASFormer runs do not share one frozen lineage")
    aggregate_value = {
        variant: {name: _mean_std([_metric(summaries[variant][seed], name) for seed in SEEDS]) for name in METRICS}
        for variant in VARIANTS
    }
    paired: dict[str, Any] = {}
    for name in METRICS:
        values = [_metric(summaries["ASFORMER_VT"][seed], name) - _metric(summaries["ASFORMER_V"][seed], name) for seed in SEEDS]
        paired[f"ASFORMER_VT_minus_V__{name}"] = {**_mean_std(values), "wins": sum(value > 0 for value in values)}
    data_hash, protocol_hash, audit_hash, release_hash = next(iter(lineage))
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "complete",
        "purpose": "formal_asformer_style_validation_only",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "test_selection_deferred": True,
        "official_reproduction": False,
        "head": "ASFORMER-style",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "jobs": len(VARIANTS) * len(SEEDS),
        "controls": {
            "same_frozen_data": True,
            "same_recipe": True,
            "hand_loss_weight_zero": True,
            "same_30hz_full_episode_evaluator": True,
            "v_vt_parameter_matched": True,
            "data_summary_sha256": data_hash,
            "source_protocol_sha256": protocol_hash,
            "source_audit_sha256": audit_hash,
            "release_manifest_sha256": release_hash,
            "source_tree_sha256": _source_tree_sha256(_source_records()),
            "release": str(release.resolve()),
            "release_sha256": sha256_file(release),
            "test_sealed": True,
        },
        "aggregate": aggregate_value,
        "paired": paired,
        "metric_contract": {
            "primary": "0.5*collapsed_semantic_f1@50 + 0.5*class_change_boundary_f1@250ms",
            "boundary_source": "decoded_action_change_only",
            "test_manifest_read": False,
        },
    }
    _atomic_json(output, result)
    complete = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "complete",
        "aggregate": str(output.resolve()),
        "aggregate_sha256": sha256_file(output),
        "release": str(release.resolve()),
        "release_sha256": sha256_file(release),
        "jobs": len(VARIANTS) * len(SEEDS),
        "test_manifest_read": False,
        "test_selection_deferred": True,
    }
    _atomic_json(output.parent / "ASFORMER_COMPLETE.json", complete)
    print(json.dumps({"event": "asformer_aggregate_complete", "jobs": 6, "output": str(output.resolve()), "test_manifest_read": False}), flush=True)
    return result


def build_model_gate(*, output: Path) -> dict[str, Any]:
    """Run deterministic architecture checks without touching any dataset."""
    from .formal500h_wave4_crossarch import _clone_batch, _fixture_batch

    current_sources = _source_records()
    current_source_tree_sha256 = _source_tree_sha256(current_sources)
    if output.is_file():
        value = _read_json(output)
        recorded_sources = value.get("sources")
        if not isinstance(recorded_sources, Mapping):
            recorded_sources = {}
        checks = {
            "schema": value.get("schema_version") == MODEL_GATE_SCHEMA,
            "status": value.get("status") == "pass",
            "jobs": value.get("jobs") == 6,
            "test_sealed": value.get("test_manifest_read") is False,
            "trainer": value.get("trainer_sha256") == sha256_file(Path(__file__).resolve()),
            "source_tree_hash": value.get("source_tree_sha256") == current_source_tree_sha256,
            "source_keys_exact": set(recorded_sources) == set(current_sources),
        }
        for name, current in current_sources.items():
            recorded = recorded_sources.get(name)
            checks[f"source:{name}"] = (
                isinstance(recorded, Mapping)
                and recorded.get("sha256") == current.get("sha256")
            )
        if all(checks.values()):
            return value
        raise ValueError(f"stale ASFormer model gate: {checks}")
    batch = _fixture_batch()
    models: dict[str, ASFormerFormal500hModel] = {}
    counts: dict[str, int] = {}
    states: dict[str, Mapping[str, torch.Tensor]] = {}
    outputs: dict[str, dict[str, Any]] = {}
    sensitivity: dict[str, float] = {}
    for variant in VARIANTS:
        wave1._seed_everything(20260831)
        model = ASFormerFormal500hModel(_model_config(variant, d_model=128, dropout=0.15)).eval()
        models[variant] = model
        counts[variant] = sum(parameter.numel() for parameter in model.parameters())
        states[variant] = model.state_dict()
        with torch.no_grad():
            outputs[variant] = model(_clone_batch(batch))
            changed = _clone_batch(batch)
            changed["tactile"] = changed["tactile"] + 7.0
            altered = model(changed)
        sensitivity[variant] = float((outputs[variant]["atomic_logits"].float() - altered["atomic_logits"].float()).abs().max())
    checks = {
        "two_representations": len(VARIANTS) == 2,
        "three_seeds": len(SEEDS) == 3,
        "v_vt_parameter_matched": counts["ASFORMER_V"] == counts["ASFORMER_VT"],
        "v_vt_initialization_matched": all(torch.equal(states["ASFORMER_V"][name], states["ASFORMER_VT"][name]) for name in states["ASFORMER_V"]),
        "vision_ignores_tactile": sensitivity["ASFORMER_V"] == 0.0,
        "vt_uses_tactile": sensitivity["ASFORMER_VT"] > 0.0,
        "outputs_finite": all(torch.isfinite(value["atomic_logits"]).all().item() for value in outputs.values()),
        "shapes": all(value["atomic_logits"].shape == (2, 6, NUM_CLASSES) and value["boundary_logits"].shape == (2, 6) and value["hand_logits"].shape == (2, 6, 3) for value in outputs.values()),
        "boundary_is_action_change_mode": all(torch.equal(value["boundary_logits"], torch.full_like(value["boundary_logits"], -20.0)) for value in outputs.values()),
    }
    result = {
        "schema_version": MODEL_GATE_SCHEMA,
        "status": "pass" if all(checks.values()) else "fail",
        "purpose": "formal_asformer_style_model_gate",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "test_selection_deferred": True,
        "trainer": str(Path(__file__).resolve()),
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "source_tree_sha256": current_source_tree_sha256,
        "sources": current_sources,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "jobs": 6,
        "parameter_counts": counts,
        "tactile_sensitivity_max_abs": sensitivity,
        "checks": checks,
    }
    _atomic_json(output, result)
    if result["status"] != "pass":
        raise RuntimeError(f"ASFormer model gate failed: {checks}")
    print(json.dumps({"event": "asformer_model_gate_pass", "jobs": 6, "test_manifest_read": False}), flush=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    validate_release = commands.add_parser("validate-release")
    validate_release.add_argument("--data-dir", type=Path, required=True)
    validate_release.add_argument("--release", type=Path, required=True)
    gate = commands.add_parser("model-gate")
    gate.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--release", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--seed", choices=SEEDS, type=int, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--precision", choices=(FROZEN_PRECISION,), default=FROZEN_PRECISION)
    train.add_argument("--batch-size", type=int, default=FROZEN_BATCH_SIZE)
    train.add_argument("--num-workers", type=int, default=FROZEN_NUM_WORKERS)
    train.add_argument("--epochs", type=int, default=FROZEN_EPOCHS)
    train.add_argument("--learning-rate", type=float, default=FROZEN_LEARNING_RATE)
    train.add_argument("--calibration-workers", type=int, default=FROZEN_CALIBRATION_WORKERS)
    validate = commands.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--variant", choices=VARIANTS, required=True)
    validate.add_argument("--seed", choices=SEEDS, type=int, required=True)
    aggregate_command = commands.add_parser("aggregate")
    aggregate_command.add_argument("--runs-root", type=Path, required=True)
    aggregate_command.add_argument("--release", type=Path, required=True)
    aggregate_command.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "audit":
        build_release(data_dir=arguments.data_dir, output=arguments.output)
    elif arguments.command == "validate-release":
        release = _validated_release(arguments.release, arguments.data_dir)
        print(
            json.dumps(
                {
                    "event": "asformer_release_valid",
                    "status": release.get("status"),
                    "source_tree_sha256": release.get("source_tree_sha256"),
                    "test_manifest_read": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    elif arguments.command == "model-gate":
        build_model_gate(output=arguments.output)
    elif arguments.command == "train":
        train_cell(
            data_dir=arguments.data_dir,
            release=arguments.release,
            output_dir=arguments.output_dir,
            variant=arguments.variant,
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
        summary = _validate_run(arguments.output_dir, variant=arguments.variant, seed=arguments.seed)
        print(json.dumps({"event": "asformer_run_valid", "variant": arguments.variant, "seed": arguments.seed, "primary": _metric(summary, "primary_composite"), "test_manifest_read": False}), flush=True)
    elif arguments.command == "aggregate":
        aggregate(runs_root=arguments.runs_root, release=arguments.release, output=arguments.output)
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
