"""Formal500h phase-space training release.

This module adds the two experiments that cannot be obtained by post-hoc
merging of a 26-class prediction:

* ``DIRECT8``: train and evaluate an eight-phase action head;
* ``HIER26P8``: retain the 26-class head and add an eight-phase auxiliary head.
* ``F26_H0_*``: matched 26-class controls with the hand-loss term disabled;
  ``F26_H0_B2`` is the full vision+tactile+skeleton+kinematics control.

The implementation deliberately reuses the frozen Formal500h window reader,
stitcher, decoder and validation-only calibration from ``wave1_train``.  The
only label-space change is made through the frozen
``ontology_robot_phase_v1.yaml`` mapping.  No test manifest is read here.
The hand head is retained for parameter-shape comparability, but its loss is
zero for these new experiments (the final segmentation task does not use
hand prediction).
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from . import formal500h_wave1_train as wave1
from .formal500h_evaluation import GroundTruthEpisode
from .formal500h_infra_runtime import (
    VariantAwareFormal500hDataset,
    collate_formal500h_fast,
)
from .formal500h_protocol import NUM_CLASSES, sha256_file
from .formal500h_clock import clock_metadata, downsample_targets_60_to_30
from .formal500h_runtime import _soft_boundary
from .formal500h_wave1_data import WAVE1_DATA_SCHEMA, WINDOW_SCHEMA
from .formal500h_wave5_robustness import coarse_ground_truth, load_group_mapping


PHASE8_SCHEMA = "formal500h_phase8_training_release_v1"
TRAINING_SCHEMA = "formal500h_phase8_training_v1"
AGGREGATE_SCHEMA = "formal500h_phase8_aggregate_v1"
PHASE8_COUNT = 8
SEEDS = (20260824, 20260825, 20260826)
MODALITIES = {
    "F26_H0_V": ("vision",),
    "F26_H0_VT": ("vision", "tactile"),
    # Full-modality no-hand-loss control.  This is the like-for-like
    # counterpart of Wave-1 B2, but with the auxiliary hand objective
    # disabled.  Keep the V/VT controls above: together they separate the
    # effects of adding body/event streams from the effect of hand loss.
    "F26_H0_B2": ("vision", "tactile", "skeleton", "kinematics"),
    "DIRECT8_V": ("vision",),
    "DIRECT8_VT": ("vision", "tactile"),
    "HIER26P8_V": ("vision",),
    "HIER26P8_VT": ("vision", "tactile"),
}
DIRECT_VARIANTS = frozenset({"DIRECT8_V", "DIRECT8_VT"})
HIER_VARIANTS = frozenset({"HIER26P8_V", "HIER26P8_VT"})
H0_VARIANTS = frozenset({"F26_H0_V", "F26_H0_VT", "F26_H0_B2"})
ALL_VARIANTS = tuple(MODALITIES)

FROZEN_PRECISION = "bf16"
FROZEN_BATCH_SIZE = 16
FROZEN_NUM_WORKERS = 20
FROZEN_EPOCHS = 3
FROZEN_LEARNING_RATE = 3.0e-4
FROZEN_WEIGHT_DECAY = 1.0e-4
FROZEN_HAND_WEIGHT = 0.0
FROZEN_CHECKPOINT_INTERVAL = 2000
FROZEN_CALIBRATION_WORKERS = 4

_ACTIVE_VARIANT: str | None = None
_ACTIVE_MAPPING: dict[str, Any] | None = None
# Keep an immutable reference before ``train_phase_job`` temporarily
# monkey-patches ``wave1._ground_truth``.  Calling the patched symbol from the
# phase wrapper would recurse forever in every validation job.
_BASE_GROUND_TRUTH = wave1._ground_truth


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def phase_mapping(ontology: Path) -> dict[str, Any]:
    """Load and validate the frozen 26 -> 8 mapping."""
    record = load_group_mapping(ontology)
    expected = [
        0, 1, 7, 2, 2, 3, 6, 4, 4, 4, 6,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    ]
    if record["group_count"] != PHASE8_COUNT or record["class_to_group"] != expected:
        raise ValueError("ontology does not match the preregistered Phase-8 mapping")
    if len(record["group_ids"]) != PHASE8_COUNT:
        raise ValueError("Phase-8 ontology group count changed")
    record = dict(record)
    record["mapping_schema"] = PHASE8_SCHEMA
    record["mapping_sha256"] = _canonical_sha(
        {
            "ontology_sha256": record["ontology_sha256"],
            "group_ids": record["group_ids"],
            "class_to_group": record["class_to_group"],
        }
    )
    return record


def _map_tensor(target: torch.Tensor, mapping: Sequence[int]) -> torch.Tensor:
    table = torch.as_tensor(mapping, dtype=torch.long, device=target.device)
    valid = (target >= 0) & (target < len(mapping))
    output = torch.full_like(target, -100)
    output[valid] = table[target[valid]]
    return output


def _phase_boundaries(
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build phase-instance IDs and phase-change boundary targets for one window."""
    if target.ndim != 1:
        raise ValueError("phase target must be one-dimensional")
    valid = target >= 0
    steps = int(target.numel())
    instance = torch.full_like(target, -100)
    hard = torch.zeros(steps, dtype=torch.bool)
    boundary_type = torch.zeros(steps, dtype=torch.long)
    current = -1
    previous = -100
    for index in range(steps):
        value = int(target[index])
        if not bool(valid[index]):
            previous = -100
            continue
        if previous < 0 or value != previous:
            current += 1
            if previous >= 0:
                hard[index] = True
                boundary_type[index] = 1
        instance[index] = current
        previous = value
    soft = torch.from_numpy(_soft_boundary(hard.numpy())).to(torch.float32)
    return instance, hard, boundary_type, soft


class PhaseLabelDataset(Dataset[dict[str, Any]]):
    """A label-space view over the immutable Formal500h HDF5 window reader."""

    def __init__(
        self,
        base: Dataset[dict[str, Any]],
        variant: str,
        mapping: Mapping[str, Any],
    ) -> None:
        self.base = base
        self.variant = variant
        self.mapping = tuple(int(value) for value in mapping["class_to_group"])
        if len(self.mapping) != NUM_CLASSES:
            raise ValueError("Phase-8 mapping must contain 26 entries")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base[index])
        phase = _map_tensor(sample["atomic_target"], self.mapping)
        if self.variant in DIRECT_VARIANTS:
            instance, hard, boundary_type, soft = _phase_boundaries(phase)
            sample["atomic_target"] = phase
            sample["atomic_instance_id"] = instance
            sample["boundary_hard"] = hard
            sample["boundary_type"] = boundary_type
            sample["boundary_target"] = soft
            sample["boundary_eval_mask"] = sample["loss_mask"].clone()
            sample["boundary_loss_mask"] = sample["loss_mask"].clone()
        elif self.variant in HIER_VARIANTS:
            sample["subtask_target"] = phase
        elif self.variant in H0_VARIANTS:
            # Explicitly retain the fine target and no-hand-loss control.
            pass
        else:
            raise ValueError(f"unknown phase training variant: {self.variant}")
        return sample


def _num_actions(variant: str) -> int:
    return PHASE8_COUNT if variant in DIRECT_VARIANTS else NUM_CLASSES


def _model_config(variant: str, *, d_model: int, dropout: float) -> dict[str, Any]:
    if variant not in MODALITIES:
        raise ValueError(f"unknown phase training variant: {variant}")
    return {
        "variant": "b2_early_fusion",
        "formal_variant": variant,
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
        "decoder_layers": 5,
        "decoder_stages": 2,
        "num_actions": _num_actions(variant),
        "num_subtasks": PHASE8_COUNT if variant in HIER_VARIANTS else 0,
        "use_boundary_head": True,
        "output_rate_hz": 30,
        "event_rate_hz": 30,
        "phase_mapping_sha256": (_ACTIVE_MAPPING or {}).get("mapping_sha256"),
        "hand_loss_enabled": False,
    }


def _phase_class_weights(statistics_path: Path, mapping: Sequence[int]) -> list[float]:
    with np.load(statistics_path, allow_pickle=False) as values:
        counts = values["action_count"].astype(np.float64)
    if counts.shape != (NUM_CLASSES,) or np.any(counts <= 0):
        raise ValueError("cannot derive Phase-8 weights from incomplete 26-class statistics")
    phase_counts = np.zeros(PHASE8_COUNT, dtype=np.float64)
    for source, target in enumerate(mapping):
        phase_counts[int(target)] += counts[source]
    if np.any(phase_counts <= 0):
        raise ValueError("all Phase-8 classes must have positive training support")
    weights = 1.0 / np.sqrt(phase_counts)
    weights /= weights.mean()
    return weights.astype(np.float32).tolist()


def _loss_config(statistics_path: Path) -> dict[str, Any]:
    if _ACTIVE_VARIANT is None or _ACTIVE_MAPPING is None:
        raise RuntimeError("phase loss config requested outside active job")
    if _ACTIVE_VARIANT in DIRECT_VARIANTS:
        action_weights = _phase_class_weights(
            statistics_path, _ACTIVE_MAPPING["class_to_group"]
        )
    else:
        action_weights = wave1.formal500h_class_weights(statistics_path)
    return {
        "action_weight": 1.0,
        "smooth_weight": 0.1,
        "boundary_weight": 0.5,
        "subtask_weight": 0.1 if _ACTIVE_VARIANT in HIER_VARIANTS else 0.0,
        "boundary_positive_weight": 8.0,
        "action_class_weights": action_weights,
    }


def _dataset(manifest: Path, statistics: Path):
    if _ACTIVE_VARIANT is None:
        raise RuntimeError("phase dataset requested outside active job")
    rows = _read_jsonl(manifest)
    if any(row.get("schema_version") != WINDOW_SCHEMA for row in rows):
        raise ValueError("window manifest schema changed")
    base = VariantAwareFormal500hDataset(
        manifest,
        statistics,
        expected_statistics_schema=WAVE1_DATA_SCHEMA,
        enabled_modalities=MODALITIES[_ACTIVE_VARIANT],
        hdf5_cache_size=8,
    )
    return PhaseLabelDataset(base, _ACTIVE_VARIANT, _ACTIVE_MAPPING or {})


def _ground_truth(rows: Sequence[Mapping[str, Any]]) -> dict[str, GroundTruthEpisode]:
    base = _BASE_GROUND_TRUTH(rows)
    if _ACTIVE_VARIANT in DIRECT_VARIANTS:
        coarse = coarse_ground_truth(base, (_ACTIVE_MAPPING or {})["class_to_group"])
        # ``coarse_ground_truth`` is shared with the generic upper-level audit
        # and labels its view from the supplied mapping.  Keep the phase-8
        # lineage explicit in this training release.
        return {
            episode_id: GroundTruthEpisode(
                episode_id=episode.episode_id,
                global_frame_60=episode.global_frame_60,
                atomic_target=episode.atomic_target,
                atomic_instance_id=episode.atomic_instance_id,
                boundary_hard=episode.boundary_hard,
                boundary_type=episode.boundary_type,
                label_mask=episode.label_mask,
                boundary_eval_mask=episode.boundary_eval_mask,
                time_ns=episode.time_ns,
                metadata={**episode.metadata, "view": "ontology_group_8"},
            )
            for episode_id, episode in coarse.items()
        }
    return base


def _evaluate_predictions(
    predictions: Mapping[str, Any],
    truth: Mapping[str, GroundTruthEpisode],
    *,
    boundary_threshold: float,
    boundary_min_distance_frames: int,
    min_segment_frames: int,
) -> dict[str, Any]:
    if _ACTIVE_VARIANT is None:
        raise RuntimeError("phase evaluator called outside active job")
    # wave1's evaluator uses its module-level NUM_CLASSES constant, so call it
    # directly with the correct number of classes for the active label space.
    from .formal500h_evaluation import evaluate_dataset

    result = evaluate_dataset(
        predictions,
        truth,
        num_actions=_num_actions(_ACTIVE_VARIANT),
        segment_iou_thresholds=(0.10, 0.25, 0.50),
        # Keep the formal evaluator identical to Wave 1.  The separate
        # ±1-second/M2R2-compatible audit remains read-only and must not alter
        # the training score or decoder selection.
        # Keep the strict 250 ms primary while emitting the complete
        # pre-specified compatibility curve, including the M2R2-style
        # one-second boundary tolerance.  This changes reporting only; model
        # selection and the Primary Composite remain anchored at 250 ms.
        boundary_tolerances_ms=(50, 100, 250, 500, 1000),
        primary_boundary_tolerance_ms=250,
        output_rate_hz=30.0,
        boundary_threshold=boundary_threshold,
        boundary_min_distance_frames=boundary_min_distance_frames,
        min_segment_frames=min_segment_frames,
        use_boundary_head=True,
        include_action_changes=True,
        strict=True,
        collect_artifacts=False,
    )
    metrics = result.metrics
    metrics["phase_training_contract"] = {
        "variant": _ACTIVE_VARIANT,
        "num_actions": _num_actions(_ACTIVE_VARIANT),
        "mapping_sha256": (_ACTIVE_MAPPING or {}).get("mapping_sha256"),
        "test_manifest_read": False,
    }
    return metrics


def _implementation_release() -> dict[str, Any]:
    module = Path(__file__).resolve()
    module_dir = module.parent
    package_root = module_dir.parents[1]
    release_manifest = (
        package_root
        / "configs"
        / "experimental"
        / "atomic_v1_phase8_hierarchy_release_sha256.txt"
    )
    if not release_manifest.is_file():
        raise ValueError(f"Phase-8 release manifest is missing: {release_manifest}")
    source_paths = {
        "phase_trainer": module,
        "clock": module_dir / "formal500h_clock.py",
        "wave1_trainer": module_dir / "formal500h_wave1_train.py",
        "runtime": module_dir / "formal500h_runtime.py",
        "infra_runtime": module_dir / "formal500h_infra_runtime.py",
        "model": module_dir / "model.py",
        "losses": module_dir / "losses.py",
        "evaluator": module_dir / "formal500h_evaluation.py",
        "inference": module_dir / "inference.py",
    }
    return {
        "schema_version": TRAINING_SCHEMA,
        "release_manifest": str(release_manifest),
        "release_manifest_sha256": sha256_file(release_manifest),
        "source_tree_sha256": _canonical_sha(
            {name: sha256_file(path) for name, path in source_paths.items()}
        ),
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "mapping": _ACTIVE_MAPPING,
        "clock_metadata": dict(clock_metadata()),
    }


def build_phase8_release(
    *, data_dir: Path, ontology: Path, output: Path
) -> dict[str, Any]:
    """Audit train/val support and freeze the mapping/data lineage."""
    mapping = phase_mapping(ontology)
    summary, artifacts = wave1._validated_data(data_dir)
    if summary.get("test_manifest_read") is not False:
        raise ValueError("source data release is not test-sealed")
    split_counts: dict[str, Counter[int]] = {"train": Counter(), "val": Counter()}
    split_ticks: dict[str, int] = {"train": 0, "val": 0}
    split_segments: dict[str, int] = {"train": 0, "val": 0}
    for split, artifact_name in (("train", "train_episodes"), ("val", "val_episodes")):
        rows = _read_jsonl(artifacts[artifact_name])
        for index, row in enumerate(rows, 1):
            path = Path(str(row["cache_path"]))
            with h5py.File(path, "r") as handle:
                action = downsample_targets_60_to_30(
                    action_class_60=np.asarray(
                        handle["targets/action_class"], dtype=np.int64
                    )
                )["action_class"]
            if np.any((action < 0) | (action >= NUM_CLASSES)):
                raise ValueError(f"invalid fine label in {path}")
            phase = np.asarray(mapping["class_to_group"], dtype=np.int64)[action]
            split_counts[split].update(int(value) for value in np.unique(phase))
            split_ticks[split] += int(len(phase))
            split_segments[split] += int(1 + np.count_nonzero(phase[1:] != phase[:-1])) if len(phase) else 0
            if index % 1000 == 0:
                print(json.dumps({"event": "phase8_release_progress", "split": split, "episodes": index}), flush=True)
    checks = {
        "source_data_complete": summary.get("status") == "complete",
        "test_sealed": summary.get("test_manifest_read") is False,
        "mapping_26_to_8": len(mapping["class_to_group"]) == NUM_CLASSES,
        "all_phase_train_support": all(split_counts["train"][i] > 0 for i in range(PHASE8_COUNT)),
        "all_phase_val_support": all(split_counts["val"][i] > 0 for i in range(PHASE8_COUNT)),
    }
    result = {
        "schema_version": PHASE8_SCHEMA,
        "status": "complete" if all(checks.values()) else "fail",
        "purpose": "freeze_phase8_training_label_view_before_validation",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "ontology": mapping,
        "source_data": {
            "summary": str((data_dir / "summary.json").resolve()),
            "summary_sha256": sha256_file(data_dir / "summary.json"),
            "source_protocol_sha256": summary["source_protocol_sha256"],
            "source_audit_sha256": summary["source_audit_sha256"],
            "train_episodes": len(_read_jsonl(artifacts["train_episodes"])),
            "val_episodes": len(_read_jsonl(artifacts["val_episodes"])),
        },
        "splits": {
            split: {
                "ticks_30": split_ticks[split],
                "phase_support": [split_counts[split][i] for i in range(PHASE8_COUNT)],
                "phase_segments": split_segments[split],
            }
            for split in ("train", "val")
        },
        "checks": checks,
    }
    _atomic_json(output, result)
    if result["status"] != "complete":
        raise RuntimeError(f"Phase-8 release audit failed: {checks}")
    return result


def _validated_phase_release(
    phase_release: Path, data_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the frozen phase release before opening any training HDF5.

    A status flag alone is not sufficient for a long formal run: an old
    release could otherwise be paired with a different split or a modified
    ontology.  This check intentionally touches only the already-frozen
    train/validation summary and mapping; it never opens the sealed test
    manifest.
    """

    release = _read_json(phase_release)
    if (
        release.get("schema_version") != PHASE8_SCHEMA
        or release.get("status") != "complete"
        or release.get("test_manifest_read") is not False
    ):
        raise ValueError("invalid or unsealed Phase-8 release")
    mapping = release.get("ontology")
    if not isinstance(mapping, dict):
        raise ValueError("Phase-8 release lacks ontology mapping")
    # Keep the validator in lockstep with ``phase_mapping``: the first
    # eleven fine classes have explicit phase groups and the remaining fifteen
    # fine classes belong to phase index 5 (26 entries in total).
    expected_mapping = [
        0, 1, 7, 2, 2, 3, 6, 4, 4, 4, 6,
    ] + [5] * 15
    if (
        mapping.get("group_count") != PHASE8_COUNT
        or mapping.get("class_to_group") != expected_mapping
        or len(mapping.get("group_ids") or []) != PHASE8_COUNT
    ):
        raise ValueError("Phase-8 release mapping is not the preregistered 26-to-8 mapping")
    canonical = _canonical_sha(
        {
            "ontology_sha256": mapping.get("ontology_sha256"),
            "group_ids": mapping.get("group_ids"),
            "class_to_group": mapping.get("class_to_group"),
        }
    )
    if mapping.get("mapping_sha256") != canonical:
        raise ValueError("Phase-8 release mapping digest mismatch")
    ontology_path = Path(str(mapping.get("ontology_path", "")))
    if not ontology_path.is_file() or sha256_file(ontology_path) != mapping.get("ontology_sha256"):
        raise ValueError("Phase-8 ontology changed or is unavailable")

    summary_path = (data_dir / "summary.json").resolve()
    summary, _ = wave1._validated_data(data_dir)
    source = release.get("source_data") or {}
    if (
        Path(str(source.get("summary", ""))).resolve() != summary_path
        or source.get("summary_sha256") != sha256_file(summary_path)
        or source.get("source_protocol_sha256") != summary.get("source_protocol_sha256")
        or source.get("source_audit_sha256") != summary.get("source_audit_sha256")
        or summary.get("test_manifest_read") is not False
    ):
        raise ValueError("Phase-8 release does not match the selected frozen data release")
    return release, mapping


def train_phase_job(
    *,
    data_dir: Path,
    phase_release: Path,
    output_dir: Path,
    variant: str,
    seed: int,
    device: str = "cuda:0",
    precision: str = FROZEN_PRECISION,
    batch_size: int = FROZEN_BATCH_SIZE,
    num_workers: int = FROZEN_NUM_WORKERS,
    epochs: int = FROZEN_EPOCHS,
    learning_rate: float = FROZEN_LEARNING_RATE,
    hand_weight: float = FROZEN_HAND_WEIGHT,
    calibration_workers: int = FROZEN_CALIBRATION_WORKERS,
) -> dict[str, Any]:
    global _ACTIVE_VARIANT, _ACTIVE_MAPPING
    variant = variant.upper()
    if variant not in MODALITIES or seed not in SEEDS:
        raise ValueError("invalid Phase-8 variant or seed")
    if (precision, batch_size, num_workers, epochs) != (
        FROZEN_PRECISION, FROZEN_BATCH_SIZE, FROZEN_NUM_WORKERS, FROZEN_EPOCHS
    ) or not math.isclose(learning_rate, FROZEN_LEARNING_RATE, rel_tol=1e-12):
        raise ValueError("frozen recipe is bf16, batch=16, workers=20, epochs=3, lr=3e-4")
    if not math.isclose(hand_weight, 0.0, abs_tol=1e-12):
        raise ValueError("Phase-8 release fixes hand loss weight to zero")
    release, mapping = _validated_phase_release(phase_release, data_dir)
    prior = {
        "variant_modalities": wave1.VARIANT_MODALITIES,
        "dataset": wave1._dataset,
        "collate": wave1.collate_formal500h,
        "implementation": wave1._implementation_release,
        "model_config": wave1._model_config,
        "loss_config": wave1._loss_config,
        "model": wave1.Formal500hModel,
        "loss_terms": wave1.formal500h_loss_terms,
        "ground_truth": wave1._ground_truth,
        "evaluate": wave1._evaluate_predictions,
        "schema": wave1.TRAINING_SCHEMA,
    }
    _ACTIVE_VARIANT, _ACTIVE_MAPPING = variant, mapping
    wave1.VARIANT_MODALITIES = {**prior["variant_modalities"], variant: MODALITIES[variant]}
    wave1._dataset = _dataset
    wave1.collate_formal500h = collate_formal500h_fast
    wave1._implementation_release = _implementation_release
    wave1._model_config = _model_config
    wave1._loss_config = _loss_config
    wave1.Formal500hModel = wave1.Formal500hModel
    wave1._ground_truth = _ground_truth
    wave1._evaluate_predictions = _evaluate_predictions
    wave1.TRAINING_SCHEMA = TRAINING_SCHEMA
    try:
        result = wave1.train_formal_variant(
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
            hand_weight=0.0,
            checkpoint_interval=FROZEN_CHECKPOINT_INTERVAL,
            calibration_workers=calibration_workers,
            resume=True,
        )
        return result
    finally:
        wave1.VARIANT_MODALITIES = prior["variant_modalities"]
        wave1._dataset = prior["dataset"]
        wave1.collate_formal500h = prior["collate"]
        wave1._implementation_release = prior["implementation"]
        wave1._model_config = prior["model_config"]
        wave1._loss_config = prior["loss_config"]
        wave1.Formal500hModel = prior["model"]
        wave1.formal500h_loss_terms = prior["loss_terms"]
        wave1._ground_truth = prior["ground_truth"]
        wave1._evaluate_predictions = prior["evaluate"]
        wave1.TRAINING_SCHEMA = prior["schema"]
        _ACTIVE_VARIANT, _ACTIVE_MAPPING = None, None


def _metric(summary: Mapping[str, Any], name: str) -> float:
    metrics = (summary.get("calibrated_validation") or {}).get("metrics") or {}
    if name == "primary_composite":
        return float((metrics.get("primary") or {})["checkpoint_composite"])
    if name == "collapsed_f1@50":
        return float((metrics.get("primary") or {})["collapsed_semantic_f1@50"])
    if name == "boundary_f1@250":
        return float((metrics.get("primary") or {})["class_change_boundary_f1@250ms"])
    if name == "frame_accuracy":
        return float(metrics["frame_accuracy"])
    if name == "tick_macro_f1":
        return float(metrics["tick_macro_f1"])
    raise KeyError(name)


def _mean_std(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise ValueError("cannot aggregate empty or non-finite metric values")
    return {
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "values": numbers,
    }


def _record_path(raw: Any, *, owner: Path, field: str) -> Path:
    """Resolve a path embedded in a run record.

    Training writes absolute paths, while accepting a relative path here makes
    the provenance checker useful for copied, self-contained run bundles.  A
    relative path is always interpreted relative to the record that contains
    it; it is never interpreted relative to the caller's current directory.
    """

    value = str(raw or "").strip()
    if not value:
        raise ValueError(f"missing {field} path in {owner}")
    path = Path(value)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def _checked_digest(path: Path, expected: Any, *, field: str) -> str:
    """Return a file digest, failing with a provenance-specific message."""

    if not path.is_file():
        raise ValueError(f"{field} is missing: {path}")
    actual = sha256_file(path)
    if str(expected or "") != actual:
        raise ValueError(
            f"{field} digest mismatch: {path} (recorded={expected!r}, actual={actual})"
        )
    return actual


def _validated_phase_job(
    summary_path: Path, *, variant: str, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one completed Phase-8 job and return its lineage record.

    This is intentionally stricter than merely looking for ``summary.json``.
    A formal aggregate must be reproducible from immutable run artifacts, and
    must not silently combine a checkpoint/configuration from another variant,
    seed, mapping, or data release.
    """

    summary_path = summary_path.resolve()
    summary = _read_json(summary_path)
    expected_modalities = list(MODALITIES[variant])
    if (
        summary.get("schema_version") != TRAINING_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("test_manifest_read") is not False
        or summary.get("variant") != variant
        or summary.get("seed") != seed
        or summary.get("episode_contract") != "formal500h_training_episode_v3"
        or summary.get("enabled_modalities") != expected_modalities
    ):
        raise ValueError(f"invalid Phase-8 summary identity/seal: {summary_path}")

    summary_digest = sha256_file(summary_path)
    complete_path = summary_path.parent / "COMPLETE.json"
    complete = _read_json(complete_path)
    complete_summary = complete.get("summary")
    if _record_path(complete_summary, owner=complete_path, field="COMPLETE.summary") != summary_path:
        raise ValueError(f"COMPLETE marker points to another summary: {complete_path}")
    if (
        complete.get("schema_version") != TRAINING_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("test_manifest_read") is not False
        or complete.get("summary_sha256") != summary_digest
    ):
        raise ValueError(f"invalid COMPLETE marker/hash: {complete_path}")

    resolved_path = _record_path(
        summary.get("resolved_config"), owner=summary_path, field="resolved_config"
    )
    resolved_digest = _checked_digest(
        resolved_path,
        summary.get("resolved_config_sha256"),
        field="resolved_config",
    )
    resolved = _read_json(resolved_path)
    resolved_model = resolved.get("model")
    if not isinstance(resolved_model, dict):
        raise ValueError(f"resolved config lacks model block: {resolved_path}")
    if (
        resolved.get("schema_version") != TRAINING_SCHEMA
        or resolved.get("test_manifest_read") is not False
        or resolved.get("variant") != variant
        or resolved.get("seed") != seed
        or resolved.get("enabled_modalities") != expected_modalities
        or resolved_model.get("formal_variant") != variant
        or resolved_model.get("enabled_modalities") != expected_modalities
    ):
        raise ValueError(f"resolved config identity mismatch: {resolved_path}")

    # The phase release fixes the recipe.  Keep these checks here as well as
    # in the trainer so an aggregate cannot mix a hand-loss or batch variant.
    recipe_checks = {
        "precision": FROZEN_PRECISION,
        "batch_size_per_gpu": FROZEN_BATCH_SIZE,
        "num_workers": FROZEN_NUM_WORKERS,
        "epochs": FROZEN_EPOCHS,
    }
    for name, expected in recipe_checks.items():
        if resolved.get(name) != expected:
            raise ValueError(
                f"resolved recipe mismatch for {variant}/seed_{seed}: "
                f"{name}={resolved.get(name)!r}, expected={expected!r}"
            )
    if not math.isclose(
        float(resolved.get("learning_rate", float("nan"))),
        FROZEN_LEARNING_RATE,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError(f"resolved learning rate mismatch: {resolved_path}")
    if not math.isclose(
        float(resolved.get("hand_weight", float("nan"))),
        FROZEN_HAND_WEIGHT,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Phase-8 hand loss is not frozen at zero: {resolved_path}")
    if "hand_weight" in summary and not math.isclose(
        float(summary["hand_weight"]),
        FROZEN_HAND_WEIGHT,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"summary hand loss is not frozen at zero: {summary_path}")
    if resolved_model.get("hand_loss_enabled") is not False:
        raise ValueError(f"Phase-8 model unexpectedly enables hand loss: {resolved_path}")
    expected_actions = _num_actions(variant)
    expected_subtasks = PHASE8_COUNT if variant in HIER_VARIANTS else 0
    if (
        resolved_model.get("num_actions") != expected_actions
        or resolved_model.get("num_subtasks") != expected_subtasks
    ):
        raise ValueError(f"resolved label-space mismatch: {resolved_path}")

    implementation = resolved.get("implementation_release")
    if not isinstance(implementation, dict):
        raise ValueError(f"resolved config lacks implementation release: {resolved_path}")
    if implementation.get("schema_version") != TRAINING_SCHEMA:
        raise ValueError(f"implementation release schema mismatch: {resolved_path}")
    release_manifest_path = _record_path(
        implementation.get("release_manifest"),
        owner=resolved_path,
        field="implementation.release_manifest",
    )
    release_manifest_digest = _checked_digest(
        release_manifest_path,
        implementation.get("release_manifest_sha256"),
        field="implementation.release_manifest",
    )
    if summary.get("release_manifest_sha256") != release_manifest_digest:
        raise ValueError(f"summary/release-manifest lineage mismatch: {summary_path}")

    mapping = implementation.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError(f"implementation release lacks phase mapping: {resolved_path}")
    expected_mapping = [
        0, 1, 7, 2, 2, 3, 6, 4, 4, 4, 6,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    ]
    if (
        mapping.get("mapping_schema") != PHASE8_SCHEMA
        or mapping.get("group_count") != PHASE8_COUNT
        or mapping.get("class_to_group") != expected_mapping
        or len(mapping.get("group_ids") or []) != PHASE8_COUNT
    ):
        raise ValueError(f"implementation phase mapping is not the frozen 26-to-8 mapping: {resolved_path}")
    mapping_digest = _canonical_sha(
        {
            "ontology_sha256": mapping.get("ontology_sha256"),
            "group_ids": mapping.get("group_ids"),
            "class_to_group": mapping.get("class_to_group"),
        }
    )
    if mapping.get("mapping_sha256") != mapping_digest:
        raise ValueError(f"phase mapping digest mismatch: {resolved_path}")
    phase_mapping_digest = str(mapping.get("mapping_sha256"))
    if resolved_model.get("phase_mapping_sha256") != phase_mapping_digest:
        raise ValueError(f"model/phase mapping lineage mismatch: {resolved_path}")
    if summary.get("phase_mapping_sha256") not in {None, phase_mapping_digest}:
        raise ValueError(f"summary/phase mapping lineage mismatch: {summary_path}")

    # Bind the run to the frozen train/validation data summary and, through it,
    # to the protocol and audit.  Artifact metadata is compared rather than
    # re-hashing large manifests once per job.
    data_summary_path = _record_path(
        resolved.get("data_summary"), owner=resolved_path, field="data_summary"
    )
    data_summary_digest = _checked_digest(
        data_summary_path,
        resolved.get("data_summary_sha256"),
        field="data_summary",
    )
    data_summary = _read_json(data_summary_path)
    if (
        data_summary.get("schema_version") != WAVE1_DATA_SCHEMA
        or data_summary.get("status") != "complete"
        or data_summary.get("test_manifest_read") is not False
        or data_summary.get("class_count") != NUM_CLASSES
        or resolved.get("data_summary_sha256") != summary.get("data_summary_sha256")
        or resolved.get("source_protocol_sha256") != summary.get("source_protocol_sha256")
        or resolved.get("source_audit_sha256") != summary.get("source_audit_sha256")
        or data_summary.get("source_protocol_sha256") != summary.get("source_protocol_sha256")
        or data_summary.get("source_audit_sha256") != summary.get("source_audit_sha256")
    ):
        raise ValueError(f"data/protocol lineage mismatch: {summary_path}")
    for source_name, digest_name in (
        ("source_protocol", "source_protocol_sha256"),
        ("source_audit", "source_audit_sha256"),
    ):
        source_path = _record_path(
            data_summary.get(source_name),
            owner=data_summary_path,
            field=source_name,
        )
        _checked_digest(
            source_path,
            data_summary.get(digest_name),
            field=source_name,
        )

    data_artifacts = data_summary.get("artifacts")
    resolved_artifacts = resolved.get("artifacts")
    if not isinstance(data_artifacts, dict) or not isinstance(resolved_artifacts, dict):
        raise ValueError(f"data artifact lineage is missing: {summary_path}")
    for name, record in data_artifacts.items():
        other = resolved_artifacts.get(name)
        if not isinstance(record, dict) or not isinstance(other, dict):
            raise ValueError(f"data artifact lineage entry missing: {name}")
        expected_path = _record_path(record.get("path"), owner=data_summary_path, field=f"data artifact {name}")
        actual_path = _record_path(other.get("path"), owner=resolved_path, field=f"resolved artifact {name}")
        if expected_path != actual_path or record.get("sha256") != other.get("sha256"):
            raise ValueError(f"resolved/data artifact lineage mismatch for {name}: {summary_path}")

    best_path = _record_path(
        summary.get("best_checkpoint"), owner=summary_path, field="best_checkpoint"
    )
    best_digest = _checked_digest(
        best_path,
        summary.get("best_checkpoint_sha256"),
        field="best_checkpoint",
    )
    complete_best = complete.get("best_checkpoint")
    if _record_path(complete_best, owner=complete_path, field="COMPLETE.best_checkpoint") != best_path:
        raise ValueError(f"COMPLETE points to another best checkpoint: {complete_path}")
    if complete.get("best_checkpoint_sha256") != best_digest:
        raise ValueError(f"COMPLETE/best checkpoint digest mismatch: {complete_path}")
    # The trainer writes a sidecar for every checkpoint.  It is checked when
    # present so copied runs remain compatible with older self-contained jobs.
    sidecar = best_path.with_suffix(best_path.suffix + ".sha256")
    if sidecar.is_file():
        tokens = sidecar.read_text(encoding="utf-8").split()
        if not tokens or tokens[0] != best_digest:
            raise ValueError(f"best checkpoint sidecar digest mismatch: {sidecar}")

    calibration = summary.get("calibrated_validation")
    if not isinstance(calibration, dict) or calibration.get("test_manifest_read") is not False:
        raise ValueError(f"validation calibration is absent or unsealed: {summary_path}")
    if len(calibration.get("candidates") or []) != 96:
        raise ValueError(f"validation calibration grid is incomplete: {summary_path}")
    selected = calibration.get("selected") or {}
    policy = wave1.FORMAL_EVALUATION_POLICY["decode_calibration"]
    if (
        selected.get("boundary_threshold") not in policy["boundary_threshold_candidates"]
        or selected.get("boundary_min_distance_frames") not in policy["boundary_min_distance_frames_30hz"]
        or selected.get("min_segment_frames") not in policy["min_segment_frames_30hz"]
    ):
        raise ValueError(f"validation decoder selection is outside the frozen grid: {summary_path}")

    recipe = {
        "precision": resolved.get("precision"),
        "batch_size_per_gpu": resolved.get("batch_size_per_gpu"),
        "num_workers": resolved.get("num_workers"),
        "epochs": resolved.get("epochs"),
        "learning_rate": float(resolved.get("learning_rate")),
        "hand_weight": float(resolved.get("hand_weight")),
    }
    lineage = {
        "data_summary": str(data_summary_path),
        "data_summary_sha256": data_summary_digest,
        "source_protocol_sha256": str(summary["source_protocol_sha256"]),
        "source_audit_sha256": str(summary["source_audit_sha256"]),
        "phase_mapping_sha256": phase_mapping_digest,
        "release_manifest_sha256": release_manifest_digest,
    }
    record = {
        "variant": variant,
        "seed": int(seed),
        "summary": str(summary_path),
        "summary_sha256": summary_digest,
        "complete": str(complete_path.resolve()),
        "complete_sha256": sha256_file(complete_path),
        "resolved_config": str(resolved_path),
        "resolved_config_sha256": resolved_digest,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": best_digest,
        "enabled_modalities": expected_modalities,
        "num_actions": expected_actions,
        "num_subtasks": expected_subtasks,
        "hand_weight": 0.0,
        "recipe": recipe,
        "lineage": lineage,
        "checks": {
            "summary_identity": True,
            "complete_hash": True,
            "resolved_config_hash": True,
            "best_checkpoint_hash": True,
            "phase_mapping": True,
            "enabled_modalities": True,
            "hand_weight_zero": True,
            "data_protocol_lineage": True,
            "calibration_grid": True,
            "test_sealed": True,
        },
    }
    # Keep the full mapping in one compact per-job record; this gives an
    # auditor enough information to reproduce the digest without reopening a
    # test artifact.
    record["phase_mapping"] = {
        "mapping_sha256": phase_mapping_digest,
        "ontology_sha256": mapping.get("ontology_sha256"),
        "group_ids": list(mapping.get("group_ids") or []),
        "class_to_group": list(mapping.get("class_to_group") or []),
    }
    return summary, record


def aggregate_phase(
    *, runs_root: Path, output: Path, variants: Sequence[str], seeds: Sequence[int] = SEEDS
) -> dict[str, Any]:
    variants = tuple(str(value).upper() for value in variants)
    seeds = tuple(int(value) for value in seeds)
    if not variants or not seeds:
        raise ValueError("Phase-8 aggregate requires at least one variant and seed")
    if len(set(variants)) != len(variants):
        raise ValueError("Phase-8 aggregate variants must be unique")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Phase-8 aggregate seeds must be unique")
    unknown_seeds = sorted(set(seeds).difference(SEEDS))
    if unknown_seeds:
        raise ValueError(f"Phase-8 aggregate contains non-frozen seeds: {unknown_seeds}")
    summaries: dict[str, dict[int, dict[str, Any]]] = {}
    job_records: dict[str, dict[str, Any]] = {}
    for variant in variants:
        if variant not in MODALITIES:
            raise ValueError(f"unknown variant: {variant}")
        summaries[variant] = {}
        for seed in seeds:
            path = runs_root / variant / f"seed_{seed}" / "summary.json"
            value, record = _validated_phase_job(path, variant=variant, seed=seed)
            summaries[variant][seed] = value
            job_records[f"{variant}/seed_{seed}"] = record

    # Every job must refer to the same frozen data/protocol/mapping/release.
    # Modalities and label-space dimensions are intentionally allowed to vary
    # by variant, but are checked against the preregistered variant contract.
    lineage_fields = (
        "data_summary",
        "data_summary_sha256",
        "source_protocol_sha256",
        "source_audit_sha256",
        "phase_mapping_sha256",
        "release_manifest_sha256",
    )
    lineage_values = {
        tuple(record["lineage"][field] for field in lineage_fields)
        for record in job_records.values()
    }
    if len(lineage_values) != 1:
        raise ValueError("Phase-8 jobs do not share one frozen data/protocol/mapping lineage")
    common_lineage_tuple = next(iter(lineage_values))
    common_lineage = dict(zip(lineage_fields, common_lineage_tuple))

    # A seed comparison is meaningful only when each variant has the same
    # deterministic recipe and every requested seed is present exactly once.
    recipe_values = {
        tuple(
            (name, record["recipe"][name])
            for name in (
                "precision",
                "batch_size_per_gpu",
                "num_workers",
                "epochs",
                "learning_rate",
                "hand_weight",
            )
        )
        for record in job_records.values()
    }
    if len(recipe_values) != 1:
        raise ValueError("Phase-8 jobs do not share one frozen training recipe")
    parameter_counts: dict[str, int] = {}
    updates_by_variant: dict[str, int] = {}
    for variant in variants:
        per_variant = [summaries[variant][seed] for seed in seeds]
        counts = {int(value.get("parameter_count", -1)) for value in per_variant}
        updates = {int(value.get("updates", -1)) for value in per_variant}
        if len(counts) != 1 or len(updates) != 1 or next(iter(updates)) <= 0:
            raise ValueError(f"{variant} seeds are not parameter/update matched")
        parameter_counts[variant] = next(iter(counts))
        updates_by_variant[variant] = next(iter(updates))

    metrics = ("primary_composite", "collapsed_f1@50", "boundary_f1@250", "frame_accuracy", "tick_macro_f1")
    aggregate = {
        variant: {name: _mean_std([_metric(summaries[variant][seed], name) for seed in seeds]) for name in metrics}
        for variant in variants
    }
    paired: dict[str, Any] = {}
    pairs = (
        ("DIRECT8_VT", "DIRECT8_V"),
        ("HIER26P8_VT", "HIER26P8_V"),
        ("HIER26P8_V", "F26_H0_V"),
        ("HIER26P8_VT", "F26_H0_VT"),
        ("F26_H0_B2", "F26_H0_VT"),
        ("F26_H0_B2", "F26_H0_V"),
        ("F26_H0_VT", "F26_H0_V"),
    )
    for newer, older in pairs:
        if newer not in summaries or older not in summaries:
            continue
        key = f"{newer}_minus_{older}"
        paired[key] = {}
        for name in metrics:
            values = [_metric(summaries[newer][seed], name) - _metric(summaries[older][seed], name) for seed in seeds]
            paired[key][name] = {**_mean_std(values), "wins": sum(value > 0 for value in values)}
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "complete",
        "purpose": "validation_only_phase_training_comparison_test_sealed",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "variants": list(variants),
        "seeds": list(seeds),
        "jobs": len(variants) * len(seeds),
        "aggregate": aggregate,
        "paired": paired,
        "controls": {
            "same_data_and_split": True,
            "same_recipe": True,
            "hand_loss_weight": 0.0,
            "hand_loss_weight_zero": True,
            "summary_identity_validated": True,
            "complete_markers_validated": True,
            "resolved_config_hashes_validated": True,
            "best_checkpoint_hashes_validated": True,
            "phase_mapping_validated": True,
            "enabled_modalities_validated": True,
            "data_protocol_lineage_validated": True,
            "lineage_equal_across_jobs": True,
            "parameter_counts_by_variant": parameter_counts,
            "updates_by_variant": updates_by_variant,
            "enabled_modalities_by_variant": {
                variant: list(MODALITIES[variant]) for variant in variants
            },
            "frozen_seed_set": set(seeds) == set(SEEDS),
            "full_episode_validation": True,
            "test_sealed": True,
        },
        "lineage": {
            **common_lineage,
            "jobs": job_records,
            "phase_mapping": job_records[next(iter(job_records))]["phase_mapping"],
        },
        "job_validation": {
            key: value["checks"] for key, value in job_records.items()
        },
    }
    _atomic_json(output, result)
    _atomic_json(output.parent / "PHASE8_COMPLETE.json", {**result, "aggregate_sha256": sha256_file(output)})
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Formal500h direct Phase-8 and hierarchical trainer")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--ontology", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--phase-release", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--variant", choices=ALL_VARIANTS, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--precision", default=FROZEN_PRECISION)
    train.add_argument("--batch-size", type=int, default=FROZEN_BATCH_SIZE)
    train.add_argument("--num-workers", type=int, default=FROZEN_NUM_WORKERS)
    train.add_argument("--epochs", type=int, default=FROZEN_EPOCHS)
    train.add_argument("--learning-rate", type=float, default=FROZEN_LEARNING_RATE)
    train.add_argument("--hand-weight", type=float, default=0.0)
    train.add_argument("--calibration-workers", type=int, default=FROZEN_CALIBRATION_WORKERS)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--runs-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--variants", default=",".join(ALL_VARIANTS))
    aggregate.add_argument("--seeds", default=",".join(str(value) for value in SEEDS))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        result = build_phase8_release(data_dir=args.data_dir, ontology=args.ontology, output=args.output)
        print(json.dumps({"event": "phase8_release_complete", "status": result["status"], "output": str(args.output)}, ensure_ascii=False), flush=True)
    elif args.command == "train":
        result = train_phase_job(
            data_dir=args.data_dir,
            phase_release=args.phase_release,
            output_dir=args.output_dir,
            variant=args.variant,
            seed=args.seed,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hand_weight=args.hand_weight,
            calibration_workers=args.calibration_workers,
        )
        print(json.dumps({"event": "phase_training_complete", "variant": args.variant, "seed": args.seed, "status": result.get("status")}, ensure_ascii=False), flush=True)
    else:
        variants = tuple(value.strip() for value in args.variants.replace(" ", ",").split(",") if value.strip())
        seeds = tuple(int(value) for value in args.seeds.replace(" ", ",").split(",") if value.strip())
        result = aggregate_phase(runs_root=args.runs_root, output=args.output, variants=variants, seeds=seeds)
        print(json.dumps({"event": "phase_aggregate_complete", "status": result["status"], "jobs": result["jobs"], "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
