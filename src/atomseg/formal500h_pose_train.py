"""Formal500h no-video pose ablations.

This module provides two validation-only, 26-class controls:

``POSE_ONLY``
    skeleton + derived kinematics, with no video or tactile input;
``POSE_TACTILE``
    skeleton + derived kinematics + tactile, with no video input.

The HDF5 reader in this module deliberately never opens any ``vision/*``
dataset.  The model keeps the fixed ContactTAS parameter layout for a fair
comparison, but receives an all-invalid vision mask and a null visual token.
Training, full-episode validation, and decoder calibration are delegated to
the frozen Phase-8/Wave-1 implementation.  The sealed test manifest is never
read.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
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

from . import formal500h_phase_train as phase
from .formal500h_clock import clock_metadata, downsample_targets_60_to_30
from .formal500h_infra_runtime import collate_formal500h_fast
from .formal500h_protocol import NUM_CLASSES, sha256_file
from .formal500h_runtime import EPISODE_SCHEMA, _soft_boundary, Formal500hDataset
from .formal500h_wave1_data import WAVE1_DATA_SCHEMA, WINDOW_SCHEMA


POSE_SCHEMA = "formal500h_pose_training_release_v1"
POSE_TRAINING_SCHEMA = "formal500h_pose_training_v1"
POSE_AGGREGATE_SCHEMA = "formal500h_pose_aggregate_v1"
POSE_VARIANTS = ("POSE_ONLY", "POSE_TACTILE")
POSE_MODALITIES: dict[str, tuple[str, ...]] = {
    "POSE_ONLY": ("skeleton", "kinematics"),
    "POSE_TACTILE": ("tactile", "skeleton", "kinematics"),
}
SEEDS = (20260824, 20260825, 20260826)

FROZEN_PRECISION = "bf16"
FROZEN_BATCH_SIZE = 16
FROZEN_NUM_WORKERS = 20
FROZEN_EPOCHS = 3
FROZEN_LEARNING_RATE = 3.0e-4
FROZEN_WEIGHT_DECAY = 1.0e-4
FROZEN_HAND_WEIGHT = 0.0
FROZEN_CHECKPOINT_INTERVAL = 2000
FROZEN_CALIBRATION_WORKERS = 4

_POSE_RELEASE_MANIFEST: Path | None = None

# Keep references to phase functions before the per-job monkey-patch.  A
# separate process is used for every job by the shell runner, but restoring all
# globals also makes direct unit-test calls safe.
_BASE_PHASE_MODALITIES = dict(phase.MODALITIES)
_BASE_PHASE_H0_VARIANTS = phase.H0_VARIANTS
_BASE_PHASE_ALL_VARIANTS = phase.ALL_VARIANTS
_BASE_PHASE_TRAINING_SCHEMA = phase.TRAINING_SCHEMA
_BASE_PHASE_DATASET = phase._dataset
_BASE_PHASE_MODEL_CONFIG = phase._model_config
_BASE_PHASE_LOSS_CONFIG = phase._loss_config
_BASE_PHASE_IMPLEMENTATION = phase._implementation_release


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
    """Hash a JSON mapping deterministically for source-tree binding."""

    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


class PoseFormal500hDataset(Formal500hDataset):
    """Formal window reader that performs no video-cache I/O.

    Production episodes still contain the complete multimodal HDF5 contract;
    this view intentionally reads only targets, tactile (when enabled), and
    pose/kinematic datasets.  Vision tensors are shape-compatible placeholders
    with all validity bits false so the shared ContactTAS model cannot observe
    video through either values or masks.
    """

    def __init__(
        self,
        manifest: str | Path,
        statistics: str | Path,
        *,
        expected_statistics_schema: str,
        enabled_modalities: Sequence[str],
        hdf5_cache_size: int = 8,
    ) -> None:
        super().__init__(
            manifest,
            statistics,
            expected_statistics_schema=expected_statistics_schema,
        )
        enabled = frozenset(str(value).strip().lower() for value in enabled_modalities)
        allowed = frozenset({"tactile", "skeleton", "kinematics"})
        if not enabled.issubset(allowed):
            raise ValueError(
                "pose reader accepts tactile/skeleton/kinematics only; "
                f"got {sorted(enabled)}"
            )
        if not {"skeleton", "kinematics"}.issubset(enabled):
            raise ValueError("pose reader requires skeleton and kinematics")
        if hdf5_cache_size < 0:
            raise ValueError("hdf5_cache_size cannot be negative")
        self.enabled_modalities = enabled
        self.hdf5_cache_size = int(hdf5_cache_size)
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()

    def close(self) -> None:
        handles = getattr(self, "_handles", None)
        while handles:
            _, handle = handles.popitem(last=False)
            try:
                handle.close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        self.close()
        state = dict(self.__dict__)
        state["_handles"] = OrderedDict()
        return state

    @staticmethod
    def _validate_handle(handle: h5py.File, path: str) -> None:
        schema = handle.attrs.get("schema_version", "")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        if str(schema) != EPISODE_SCHEMA or not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"invalid Formal500h HDF5: {path}")

    def _open(self, path: str) -> tuple[h5py.File, bool]:
        if self.hdf5_cache_size == 0:
            handle = h5py.File(path, "r")
            self._validate_handle(handle, path)
            return handle, True
        handle = self._handles.pop(path, None)
        if handle is None or not bool(handle.id.valid):
            if handle is not None:
                handle.close()
            handle = h5py.File(path, "r")
            self._validate_handle(handle, path)
        self._handles[path] = handle
        while len(self._handles) > self.hdf5_cache_size:
            _, expired = self._handles.popitem(last=False)
            expired.close()
        return handle, False

    @staticmethod
    def _tensor(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(value)).to(dtype=dtype)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.entries[index]
        start = int(row["start_frame_60"])
        end = int(row["end_frame_60_exclusive"])
        if start % 2:
            raise ValueError("Formal500h windows must start on an even 60 Hz frame")
        selection = slice(start, end, 2)
        handle, close_after = self._open(str(row["cache_path"]))
        try:
            # No vision/* key is touched in this method.  This is the central
            # no-video guarantee and is tested with an HDF5 fixture lacking the
            # vision group entirely.
            time_ns = np.asarray(handle["timeline/time_ns"][selection], dtype=np.int64)
            # Targets are source-rate (60 Hz).  Use the same canonical
            # conversion as the video/multimodal readers so pose controls do
            # not receive a different boundary clock.
            action_60 = np.asarray(
                handle["targets/action_class"][start:end], dtype=np.int64
            )
            instance_60 = np.asarray(
                handle["targets/action_instance"][start:end], dtype=np.int64
            )
            boundary_instance_60 = np.asarray(
                handle["targets/boundary_instance"][start:end], dtype=np.bool_
            )
            boundary_class_change_60 = np.asarray(
                handle["targets/boundary_class_change"][start:end], dtype=np.bool_
            )
            targets = downsample_targets_60_to_30(
                action_class_60=action_60,
                action_instance_60=instance_60,
                boundary_instance_60=boundary_instance_60,
                boundary_class_change_60=boundary_class_change_60,
            )
            action = np.asarray(targets["action_class"], dtype=np.int64)
            hand = np.asarray(handle["targets/action_hand"][selection], dtype=np.int64)
            instance = np.asarray(targets["action_instance"], dtype=np.int64)
            boundary_hard = np.asarray(targets["boundary_instance"], dtype=np.bool_)
            boundary_change = np.asarray(
                targets["boundary_class_change"], dtype=np.bool_
            )
            steps = int(action.size)

            if "tactile" in self.enabled_modalities:
                tactile_valid = np.asarray(
                    handle["sensors/tactile/valid"][selection], dtype=np.bool_
                )
                tactile = self._normalized(
                    np.asarray(handle["sensors/tactile/value"][selection], dtype=np.float32),
                    self.tactile_mean,
                    self.tactile_std,
                    tactile_valid,
                )
                raw_quality = np.asarray(
                    handle["sensors/tactile/quality_raw460"][selection], dtype=np.float32
                )
                tactile_quality = np.stack(
                    [
                        raw_quality[..., 0],
                        raw_quality[..., 2],
                        raw_quality[..., 3],
                        raw_quality[..., 4],
                        np.zeros_like(raw_quality[..., 0]),
                    ],
                    axis=-1,
                ).astype(np.float32)
            else:
                tactile = np.zeros((steps, *self.tactile_mean.shape), dtype=np.float32)
                tactile_valid = np.zeros(
                    (steps, self.tactile_mean.shape[0]), dtype=np.bool_
                )
                tactile_quality = np.zeros(
                    (steps, self.tactile_mean.shape[0], 5), dtype=np.float32
                )

            skeleton_valid = np.asarray(
                handle["sensors/skeleton/valid"][selection], dtype=np.bool_
            )
            joint_valid = np.asarray(
                handle["sensors/skeleton/joint_valid"][selection], dtype=np.bool_
            )
            joint_valid &= skeleton_valid[..., None]
            skeleton = self._normalized(
                np.asarray(handle["sensors/skeleton/relative"][selection], dtype=np.float32),
                self.skeleton_mean,
                self.skeleton_std,
                joint_valid,
            )

            kinematics_valid = np.asarray(
                handle["sensors/kinematics/valid"][selection], dtype=np.bool_
            )
            kinematics = self._normalized(
                np.asarray(handle["sensors/kinematics/value"][selection], dtype=np.float32),
                self.kinematics_mean,
                self.kinematics_std,
                kinematics_valid,
            )
            pose_quality = np.asarray(
                handle["sensors/pose/quality"][selection], dtype=np.float32
            )

        finally:
            if close_after:
                handle.close()

        if np.any((action < 0) | (action >= NUM_CLASSES)):
            raise ValueError("Formal500h target escaped the frozen 26-class ontology")
        boundary_type = np.zeros(steps, dtype=np.int64)
        boundary_type[boundary_hard & boundary_change] = 1
        boundary_type[boundary_hard & ~boundary_change] = 2
        mask = np.ones(steps, dtype=np.bool_)
        global_frame_30 = start // 2 + np.arange(steps, dtype=np.int64)

        # Shape-compatible all-invalid visual placeholders.  The placeholders
        # are generated without consulting the HDF5 vision group.
        vision = np.zeros((1, 3, 1536), dtype=np.float32)
        vision_cache_valid = np.zeros((1, 3), dtype=np.bool_)
        vision_index = np.full((steps, 3), -1, dtype=np.int64)
        vision_valid = np.zeros((steps, 3), dtype=np.bool_)
        # Freshness is per output tick *and per view* in the shared model
        # contract (`[T, 3]`).  Keeping the second dimension here avoids an
        # accidental broadcast against the three-view score tensor even
        # though every visual view is intentionally invalid.
        vision_freshness = np.full((steps, 3), 1.0e6, dtype=np.float32)

        tensor = self._tensor
        return {
            "episode_id": str(row["episode_id"]),
            "time_ns": tensor(time_ns, torch.long),
            "global_frame_60": tensor(global_frame_30, torch.long),
            "vision_features": tensor(vision, torch.float32),
            "vision_cache_valid": tensor(vision_cache_valid, torch.bool),
            "vision_index_60": tensor(vision_index, torch.long),
            "vision_valid_60": tensor(vision_valid, torch.bool),
            "vision_freshness_ms": tensor(vision_freshness, torch.float32),
            "tactile": tensor(tactile, torch.float32),
            "tactile_valid": tensor(tactile_valid, torch.bool),
            "tactile_quality": tensor(tactile_quality, torch.float32),
            "skeleton": tensor(skeleton, torch.float32),
            "skeleton_valid": tensor(skeleton_valid, torch.bool),
            "skeleton_joint_valid": tensor(joint_valid, torch.bool),
            "kinematics": tensor(kinematics, torch.float32),
            "kinematics_valid": tensor(kinematics_valid, torch.bool),
            "pose_quality": tensor(pose_quality, torch.float32),
            "atomic_target": tensor(action, torch.long),
            "hand_target": tensor(hand, torch.long),
            "atomic_instance_id": tensor(instance, torch.long),
            "subtask_target": torch.full((steps,), -100, dtype=torch.long),
            "boundary_target": tensor(_soft_boundary(boundary_hard), torch.float32),
            "boundary_hard": tensor(boundary_hard, torch.bool),
            "boundary_type": tensor(boundary_type, torch.long),
            "boundary_eval_mask": tensor(mask, torch.bool),
            "core_mask": tensor(mask, torch.bool),
            "loss_mask": tensor(mask, torch.bool),
            "boundary_loss_mask": tensor(mask, torch.bool),
            "clock_metadata": dict(clock_metadata()),
            "read_start_step": start // 2,
            "steps": steps,
        }


def _pose_dataset(manifest: Path, statistics: Path) -> PoseFormal500hDataset:
    rows = _read_jsonl(manifest)
    if any(row.get("schema_version") != WINDOW_SCHEMA for row in rows):
        raise ValueError("Pose window manifest schema changed")
    active = phase._ACTIVE_VARIANT
    if active not in POSE_MODALITIES:
        raise RuntimeError("pose dataset requested outside an active pose job")
    return PoseFormal500hDataset(
        manifest,
        statistics,
        expected_statistics_schema=WAVE1_DATA_SCHEMA,
        enabled_modalities=POSE_MODALITIES[active],
        hdf5_cache_size=8,
    )


def _pose_model_config(variant: str, *, d_model: int, dropout: float) -> dict[str, Any]:
    if variant not in POSE_MODALITIES:
        raise ValueError(f"unknown pose variant: {variant}")
    # Construct this independently rather than calling the Phase-8 helper:
    # unit tests and release audits can inspect a pose config before extending
    # Phase-8's temporary variant table.
    return {
        "variant": "b2_early_fusion",
        "formal_variant": variant,
        "enabled_modalities": list(POSE_MODALITIES[variant]),
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
        "num_actions": NUM_CLASSES,
        "num_subtasks": 0,
        "use_boundary_head": True,
        "output_rate_hz": 30,
        "event_rate_hz": 30,
        "clock_metadata": dict(clock_metadata()),
        "phase_mapping_sha256": (phase._ACTIVE_MAPPING or {}).get("mapping_sha256"),
        "hand_weight": FROZEN_HAND_WEIGHT,
        "hand_loss_enabled": False,
        "video_input_disabled": True,
        "video_input_policy": "no_video_reader_all_visual_masks_false",
        "pose_ablation_schema": POSE_SCHEMA,
    }


def _pose_source_tree() -> dict[str, Path]:
    """Return the complete local import closure for the pose H0 runner."""

    module = Path(__file__).resolve()
    directory = module.parent
    return {
        "package_init": directory / "__init__.py",
        "pose_trainer": module,
        "phase_trainer": directory / "formal500h_phase_train.py",
        "wave1_trainer": directory / "formal500h_wave1_train.py",
        "wave1_data": directory / "formal500h_wave1_data.py",
        "runtime": directory / "formal500h_runtime.py",
        "infra_runtime": directory / "formal500h_infra_runtime.py",
        "evaluator": directory / "formal500h_evaluation.py",
        "protocol": directory / "formal500h_protocol.py",
        "training_release": directory / "formal500h_training_release.py",
        "inference": directory / "inference.py",
        "model": directory / "model.py",
        "baselines": directory / "baselines.py",
        "losses": directory / "losses.py",
        # phase_train imports wave5; wave5 in turn imports these Wave-4/3
        # helpers.  Keep the transitive closure explicit so an indirect
        # decoder/evaluator edit cannot bypass the H0 release gate.
        "wave5_robustness": directory / "formal500h_wave5_robustness.py",
        "wave4_metric_audit": directory / "formal500h_wave4_metric_audit.py",
        "wave4_crossarch": directory / "formal500h_wave4_crossarch.py",
        "wave3b_vt": directory / "formal500h_wave3b_vt.py",
        "wave3c_train": directory / "formal500h_wave3c_train.py",
    }


def _pose_source_records() -> dict[str, dict[str, str]]:
    paths = _pose_source_tree()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        details = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise ValueError(f"pose source tree is incomplete: {details}")
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    }


def _pose_source_tree_sha256(
    records: Mapping[str, Mapping[str, str]],
) -> str:
    """Hash source names and digests, independent of mount-point spelling."""

    return _canonical_sha(
        {name: str(record["sha256"]) for name, record in sorted(records.items())}
    )


def _pose_manifest_path() -> Path:
    """Resolve the checksum manifest used to gate the pose implementation."""

    module = Path(__file__).resolve()
    package_root = module.parents[2]
    if _POSE_RELEASE_MANIFEST is not None:
        manifest = _POSE_RELEASE_MANIFEST
    else:
        configured = os.environ.get("POSE_RELEASE_MANIFEST", "").strip()
        manifest = (
            Path(configured).expanduser().resolve()
            if configured
            else package_root
            / "configs"
            / "experimental"
            / "atomic_v1_pose_no_video_release_sha256.txt"
        )
    return manifest.resolve()


def _pose_implementation_release() -> dict[str, Any]:
    """Return a source/data lineage record for a pose H0 process."""

    manifest = _pose_manifest_path()
    if not manifest.is_file():
        raise ValueError(f"pose release manifest is missing: {manifest}")
    source_records = _pose_source_records()
    return {
        "schema_version": POSE_TRAINING_SCHEMA,
        "release_manifest": str(manifest),
        "release_manifest_sha256": sha256_file(manifest),
        "source_tree_sha256": _pose_source_tree_sha256(source_records),
        "no_video_input": True,
        "video_input_policy": "no_video_reader_all_visual_masks_false",
        "pose_modalities": list(POSE_MODALITIES.get(phase._ACTIVE_VARIANT or "POSE_ONLY", ())),
        "sources": source_records,
        # Retain the frozen Phase-8 mapping digest as an upstream lineage
        # anchor even though pose runs keep the fine 26-class target.
        "phase_mapping": phase._ACTIVE_MAPPING,
    }


def _validate_pose_hdf5(path: Path, *, row: Mapping[str, Any]) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        schema = handle.attrs.get("schema_version", "")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        if str(schema) != EPISODE_SCHEMA or not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"invalid pose-training HDF5: {path}")
        required = {
            "timeline/time_ns",
            "targets/action_class",
            "targets/action_hand",
            "targets/action_instance",
            "targets/boundary_instance",
            "targets/boundary_class_change",
            "sensors/tactile/value",
            "sensors/tactile/valid",
            "sensors/tactile/quality_raw460",
            "sensors/skeleton/relative",
            "sensors/skeleton/valid",
            "sensors/skeleton/joint_valid",
            "sensors/kinematics/value",
            "sensors/kinematics/valid",
            "sensors/pose/quality",
        }
        missing = sorted(name for name in required if name not in handle)
        if missing:
            raise ValueError(f"pose fields missing from {path}: {missing}")
        # The release audit iterates the episode manifest, whose rows cover
        # the complete HDF5 and therefore do not carry window bounds.  During
        # training, the same validator may receive a window row with explicit
        # 60-Hz bounds.  Validate bounds only in the latter case; treating an
        # episode row's absent fields as ``0:0`` incorrectly rejects every
        # episode during pose-release construction.
        has_start = "start_frame_60" in row
        has_end = "end_frame_60_exclusive" in row
        if has_start or has_end:
            if not (has_start and has_end):
                raise ValueError(f"pose window range incomplete for {path}")
            start = int(row["start_frame_60"])
            end = int(row["end_frame_60_exclusive"])
            if end <= start or end > int(handle["timeline/time_ns"].shape[0]):
                raise ValueError(f"pose window range invalid for {path}: {start}:{end}")
        shapes = {
            "skeleton": tuple(handle["sensors/skeleton/relative"].shape[1:]),
            "kinematics": tuple(handle["sensors/kinematics/value"].shape[1:]),
            "pose_quality": tuple(handle["sensors/pose/quality"].shape[1:]),
        }
        if shapes != {"skeleton": (2, 25, 7), "kinematics": (48,), "pose_quality": (6,)}:
            raise ValueError(f"pose field shapes changed for {path}: {shapes}")
    return {"path": str(path), "pose_fields_verified": True}


def build_pose_release(*, data_dir: Path, output: Path) -> dict[str, Any]:
    """Freeze pose-field availability against the already sealed data release."""
    implementation = _pose_implementation_release()
    # ``formal500h_phase_train`` delegates data validation to its imported
    # Wave-1 module; keep the same canonical validator here.
    summary, artifacts = phase.wave1._validated_data(data_dir)
    if summary.get("test_manifest_read") is not False:
        raise ValueError("pose release cannot be built from an unsealed data release")
    split_counts: dict[str, Counter[int]] = {"train": Counter(), "val": Counter()}
    split_rows: dict[str, int] = {}
    split_ticks: dict[str, int] = {}
    for split, artifact_name in (("train", "train_episodes"), ("val", "val_episodes")):
        rows = _read_jsonl(artifacts[artifact_name])
        split_rows[split] = len(rows)
        split_ticks[split] = 0
        for index, row in enumerate(rows, 1):
            path = Path(str(row["cache_path"]))
            _validate_pose_hdf5(path, row=row)
            with h5py.File(path, "r") as handle:
                action = downsample_targets_60_to_30(
                    action_class_60=np.asarray(
                        handle["targets/action_class"], dtype=np.int64
                    )
                )["action_class"]
            if np.any((action < 0) | (action >= NUM_CLASSES)):
                raise ValueError(f"invalid fine labels in {path}")
            split_counts[split].update(int(value) for value in np.unique(action))
            split_ticks[split] += int(action.size)
            if index % 1000 == 0:
                print(
                    json.dumps(
                        {"event": "pose_release_progress", "split": split, "episodes": index}
                    ),
                    flush=True,
                )
    checks = {
        "source_data_complete": summary.get("status") == "complete",
        "test_sealed": summary.get("test_manifest_read") is False,
        "pose_fields_present": all(split_rows.get(split, 0) > 0 for split in ("train", "val")),
        "all_26_train_support": all(split_counts["train"][i] > 0 for i in range(NUM_CLASSES)),
        "all_26_val_support": all(split_counts["val"][i] > 0 for i in range(NUM_CLASSES)),
    }
    result = {
        "schema_version": POSE_SCHEMA,
        "status": "complete" if all(checks.values()) else "fail",
        "purpose": "freeze_no_video_pose_ablation_before_validation",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "clock_metadata": dict(clock_metadata()),
        "source_data": {
            "summary": str((data_dir / "summary.json").resolve()),
            "summary_sha256": sha256_file(data_dir / "summary.json"),
            "source_protocol_sha256": summary["source_protocol_sha256"],
            "source_audit_sha256": summary["source_audit_sha256"],
            "train_episodes": split_rows["train"],
            "val_episodes": split_rows["val"],
        },
        "pose_contract": {
            "video_input_disabled": True,
            "reader_does_not_open": ["vision/features", "vision/valid", "vision/index_60", "vision/aligned_valid_60", "vision/freshness_ms_60"],
            "skeleton_shape_per_tick": [2, 25, 7],
            "kinematics_shape_per_tick": [48],
            "pose_quality_shape_per_tick": [6],
            "tactile_shape_per_tick": [2, 360],
        },
        # Bind the release to the exact executable source closure.  The
        # aliases make the contract easy to inspect from shell tooling while
        # the nested object is retained in every resolved training config.
        "implementation": implementation,
        "source_tree_sha256": implementation["source_tree_sha256"],
        "sources": implementation["sources"],
        "release_manifest_sha256": implementation["release_manifest_sha256"],
        "splits": {
            split: {
                "episodes": split_rows[split],
                "ticks_30": split_ticks[split],
                "class_support": [split_counts[split][i] for i in range(NUM_CLASSES)],
            }
            for split in ("train", "val")
        },
        "checks": checks,
    }
    _atomic_json(output, result)
    if result["status"] != "complete":
        raise RuntimeError(f"pose release audit failed: {checks}")
    return result


def _validated_pose_release(pose_release: Path, data_dir: Path) -> dict[str, Any]:
    release = _read_json(pose_release)
    if (
        release.get("schema_version") != POSE_SCHEMA
        or release.get("status") != "complete"
        or release.get("test_manifest_read") is not False
    ):
        raise ValueError("invalid or unsealed pose release")
    source = release.get("source_data") or {}
    summary_path = (data_dir / "summary.json").resolve()
    summary, _ = phase.wave1._validated_data(data_dir)
    if (
        Path(str(source.get("summary", ""))).resolve() != summary_path
        or source.get("summary_sha256") != sha256_file(summary_path)
        or source.get("source_protocol_sha256") != summary.get("source_protocol_sha256")
        or source.get("source_audit_sha256") != summary.get("source_audit_sha256")
    ):
        raise ValueError("pose release does not match frozen Formal500h data")
    contract = release.get("pose_contract") or {}
    if (
        contract.get("video_input_disabled") is not True
        or contract.get("skeleton_shape_per_tick") != [2, 25, 7]
        or contract.get("kinematics_shape_per_tick") != [48]
        or contract.get("pose_quality_shape_per_tick") != [6]
    ):
        raise ValueError("pose release field contract changed")
    implementation = release.get("implementation")
    current = _pose_implementation_release()
    recorded_sources = (
        implementation.get("sources")
        if isinstance(implementation, Mapping)
        else None
    )
    if not isinstance(recorded_sources, Mapping):
        recorded_sources = {}
    checks = {
        "implementation_schema": isinstance(implementation, Mapping)
        and implementation.get("schema_version") == POSE_TRAINING_SCHEMA,
        "manifest_hash": isinstance(implementation, Mapping)
        and implementation.get("release_manifest_sha256")
        == current["release_manifest_sha256"],
        "source_tree_hash": isinstance(implementation, Mapping)
        and implementation.get("source_tree_sha256")
        == current["source_tree_sha256"]
        and release.get("source_tree_sha256") == current["source_tree_sha256"],
        "source_keys_exact": set(recorded_sources) == set(current["sources"]),
        "release_aliases": release.get("sources") == recorded_sources
        and release.get("release_manifest_sha256")
        == current["release_manifest_sha256"],
    }
    for name, current_record in current["sources"].items():
        recorded = recorded_sources.get(name)
        checks[f"source:{name}"] = (
            isinstance(recorded, Mapping)
            and recorded.get("sha256") == current_record.get("sha256")
        )
    if not all(checks.values()):
        raise ValueError(f"stale or incompatible pose release: {checks}")
    return release


def train_pose_job(
    *,
    data_dir: Path,
    phase_release: Path,
    pose_release: Path,
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
    global _POSE_RELEASE_MANIFEST
    variant = variant.upper()
    if variant not in POSE_VARIANTS or seed not in SEEDS:
        raise ValueError("invalid pose variant or frozen seed")
    if (precision, batch_size, num_workers, epochs) != (
        FROZEN_PRECISION,
        FROZEN_BATCH_SIZE,
        FROZEN_NUM_WORKERS,
        FROZEN_EPOCHS,
    ) or not math.isclose(learning_rate, FROZEN_LEARNING_RATE, rel_tol=1e-12):
        raise ValueError("pose recipe is bf16, batch=16, workers=20, epochs=3, lr=3e-4")
    _validated_pose_release(pose_release, data_dir)
    phase_release = phase_release.resolve()
    if not phase_release.is_file():
        raise ValueError(f"Phase-8 release is missing: {phase_release}")

    # Extend, rather than rewrite, the Phase-8 globals.  The Phase-8 runner's
    # exact seven-variant matrix is untouched; this adapter is only imported by
    # the separate pose runner.
    prior_manifest = _POSE_RELEASE_MANIFEST
    prior = {
        "modalities": phase.MODALITIES,
        "h0": phase.H0_VARIANTS,
        "all": phase.ALL_VARIANTS,
        "schema": phase.TRAINING_SCHEMA,
        "dataset": phase._dataset,
        "model_config": phase._model_config,
        "implementation": phase._implementation_release,
    }
    phase.MODALITIES = {**phase.MODALITIES, **POSE_MODALITIES}
    phase.H0_VARIANTS = frozenset(set(phase.H0_VARIANTS).union(POSE_VARIANTS))
    phase.ALL_VARIANTS = tuple(dict.fromkeys((*phase.ALL_VARIANTS, *POSE_VARIANTS)))
    phase.TRAINING_SCHEMA = POSE_TRAINING_SCHEMA
    phase._dataset = _pose_dataset
    phase._model_config = _pose_model_config
    phase._implementation_release = _pose_implementation_release
    # ``_POSE_RELEASE_MANIFEST`` denotes the checksum manifest, not the
    # pose_release.json data record.  The old code accidentally pointed it at
    # pose_release.json, which made implementation lineage circular and
    # prevented a reliable source-tree check.
    configured_manifest = os.environ.get("POSE_RELEASE_MANIFEST", "").strip()
    _POSE_RELEASE_MANIFEST = (
        Path(configured_manifest).expanduser().resolve()
        if configured_manifest
        else None
    )
    try:
        return phase.train_phase_job(
            data_dir=data_dir,
            phase_release=phase_release,
            output_dir=output_dir,
            variant=variant,
            seed=seed,
            device=device,
            precision=precision,
            batch_size=batch_size,
            num_workers=num_workers,
            epochs=epochs,
            learning_rate=learning_rate,
            hand_weight=0.0,
            calibration_workers=calibration_workers,
        )
    finally:
        phase.MODALITIES = prior["modalities"]
        phase.H0_VARIANTS = prior["h0"]
        phase.ALL_VARIANTS = prior["all"]
        phase.TRAINING_SCHEMA = prior["schema"]
        phase._dataset = prior["dataset"]
        phase._model_config = prior["model_config"]
        phase._implementation_release = prior["implementation"]
        _POSE_RELEASE_MANIFEST = prior_manifest


def _record_path(raw: Any, *, owner: Path, field: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise ValueError(f"missing {field} path in {owner}")
    path = Path(value)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def _validate_pose_summary(path: Path, *, variant: str, seed: int) -> dict[str, Any]:
    summary = _read_json(path)
    expected_modalities = list(POSE_MODALITIES[variant])
    if (
        summary.get("schema_version") != POSE_TRAINING_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("test_manifest_read") is not False
        or summary.get("variant") != variant
        or summary.get("seed") != seed
        or summary.get("enabled_modalities") != expected_modalities
        # Older Wave-1 summary writers did not duplicate hand_weight at the
        # top level; the immutable resolved_config below is authoritative.
        # Accept an omitted field, but reject an explicit non-zero value.
        or (
            "hand_weight" in summary
            and not math.isclose(
                float(summary.get("hand_weight")),
                FROZEN_HAND_WEIGHT,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
    ):
        raise ValueError(f"invalid pose summary identity/seal: {path}")
    resolved_path = _record_path(summary.get("resolved_config"), owner=path, field="resolved_config")
    if sha256_file(resolved_path) != summary.get("resolved_config_sha256"):
        raise ValueError(f"resolved config digest mismatch: {resolved_path}")
    resolved = _read_json(resolved_path)
    model = resolved.get("model") or {}
    recipe = {
        "precision": FROZEN_PRECISION,
        "batch_size_per_gpu": FROZEN_BATCH_SIZE,
        "num_workers": FROZEN_NUM_WORKERS,
        "epochs": FROZEN_EPOCHS,
    }
    if any(resolved.get(name) != expected for name, expected in recipe.items()):
        raise ValueError(f"pose recipe mismatch: {resolved_path}")
    if not math.isclose(float(resolved.get("learning_rate", float("nan"))), FROZEN_LEARNING_RATE, rel_tol=1e-12):
        raise ValueError(f"pose learning rate mismatch: {resolved_path}")
    if not math.isclose(
        float(resolved.get("hand_weight", float("nan"))),
        FROZEN_HAND_WEIGHT,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"pose hand loss is not disabled: {resolved_path}")
    if (
        resolved.get("schema_version") != POSE_TRAINING_SCHEMA
        or resolved.get("test_manifest_read") is not False
        or resolved.get("variant") != variant
        or resolved.get("seed") != seed
        or resolved.get("enabled_modalities") != expected_modalities
        or model.get("video_input_disabled") is not True
        or model.get("enabled_modalities") != expected_modalities
        or model.get("num_actions") != NUM_CLASSES
        or model.get("hand_loss_enabled") is not False
    ):
        raise ValueError(f"pose resolved config identity mismatch: {resolved_path}")
    implementation = resolved.get("implementation_release") or {}
    release_path = _record_path(
        implementation.get("release_manifest"), owner=resolved_path, field="implementation.release_manifest"
    )
    release_digest = sha256_file(release_path)
    if release_digest != implementation.get("release_manifest_sha256"):
        raise ValueError(f"pose release manifest digest mismatch: {release_path}")
    if implementation.get("schema_version") != POSE_TRAINING_SCHEMA or implementation.get("no_video_input") is not True:
        raise ValueError(f"pose implementation lineage mismatch: {resolved_path}")
    current_implementation = _pose_implementation_release()
    recorded_sources = implementation.get("sources")
    if not isinstance(recorded_sources, Mapping):
        recorded_sources = {}
    source_checks = {
        "manifest": implementation.get("release_manifest_sha256")
        == current_implementation["release_manifest_sha256"],
        "tree": implementation.get("source_tree_sha256")
        == current_implementation["source_tree_sha256"],
        "keys": set(recorded_sources) == set(current_implementation["sources"]),
    }
    for name, current_record in current_implementation["sources"].items():
        recorded = recorded_sources.get(name)
        source_checks[f"source:{name}"] = (
            isinstance(recorded, Mapping)
            and recorded.get("sha256") == current_record.get("sha256")
        )
    if not all(source_checks.values()):
        raise ValueError(f"stale pose implementation source tree: {resolved_path}: {source_checks}")
    data_summary_path = _record_path(resolved.get("data_summary"), owner=resolved_path, field="data_summary")
    data_digest = sha256_file(data_summary_path)
    if data_digest != resolved.get("data_summary_sha256"):
        raise ValueError(f"pose data summary digest mismatch: {data_summary_path}")
    complete_path = path.parent / "COMPLETE.json"
    complete = _read_json(complete_path)
    if (
        complete.get("schema_version") != POSE_TRAINING_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("test_manifest_read") is not False
        or _record_path(complete.get("summary"), owner=complete_path, field="COMPLETE.summary") != path.resolve()
        or complete.get("summary_sha256") != sha256_file(path)
    ):
        raise ValueError(f"invalid pose completion marker: {complete_path}")
    best_path = _record_path(summary.get("best_checkpoint"), owner=path, field="best_checkpoint")
    if sha256_file(best_path) != summary.get("best_checkpoint_sha256"):
        raise ValueError(f"pose checkpoint digest mismatch: {best_path}")
    calibration = summary.get("calibrated_validation") or {}
    if calibration.get("test_manifest_read") is not False or len(calibration.get("candidates") or []) != 96:
        raise ValueError(f"pose validation calibration is incomplete: {path}")
    return {
        "variant": variant,
        "seed": seed,
        "summary": str(path.resolve()),
        "summary_sha256": sha256_file(path),
        "resolved_config": str(resolved_path),
        "resolved_config_sha256": sha256_file(resolved_path),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "enabled_modalities": expected_modalities,
        "lineage": {
            "data_summary": str(data_summary_path),
            "data_summary_sha256": data_digest,
            "source_protocol_sha256": str(summary.get("source_protocol_sha256")),
            "source_audit_sha256": str(summary.get("source_audit_sha256")),
            "release_manifest_sha256": release_digest,
            "source_tree_sha256": implementation.get("source_tree_sha256"),
        },
    }


def _metric(summary: Mapping[str, Any], name: str) -> float:
    return phase._metric(summary, name)


def _mean_std(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise ValueError("cannot aggregate empty/non-finite pose metrics")
    return {
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "values": numbers,
    }


def aggregate_pose(*, runs_root: Path, output: Path) -> dict[str, Any]:
    summaries: dict[str, dict[int, dict[str, Any]]] = {}
    records: dict[str, Any] = {}
    for variant in POSE_VARIANTS:
        summaries[variant] = {}
        for seed in SEEDS:
            path = runs_root / variant / f"seed_{seed}" / "summary.json"
            summary = _read_json(path)
            record = _validate_pose_summary(path, variant=variant, seed=seed)
            summaries[variant][seed] = summary
            records[f"{variant}/seed_{seed}"] = record
    lineage_fields = (
        "data_summary",
        "data_summary_sha256",
        "source_protocol_sha256",
        "source_audit_sha256",
        "release_manifest_sha256",
    )
    lineage_values = {
        tuple(record["lineage"][field] for field in lineage_fields)
        for record in records.values()
    }
    if len(lineage_values) != 1:
        raise ValueError("pose jobs do not share one frozen data/release lineage")
    # The two modality ablations must start from the same seeded parameter
    # state for a paired comparison.  Their input masks differ, but the fixed
    # ContactTAS parameter layout and optimizer/update budget do not.
    for seed in SEEDS:
        left = summaries["POSE_ONLY"][seed]
        right = summaries["POSE_TACTILE"][seed]
        if left.get("initial_model_sha256") != right.get("initial_model_sha256"):
            raise ValueError(
                f"pose variants are not paired at seed {seed}: initial model differs"
            )
        if left.get("parameter_count") != right.get("parameter_count"):
            raise ValueError(
                f"pose variants are not parameter matched at seed {seed}"
            )
        if left.get("updates") != right.get("updates"):
            raise ValueError(
                f"pose variants are not update matched at seed {seed}"
            )
    metrics = ("primary_composite", "collapsed_f1@50", "boundary_f1@250", "frame_accuracy", "tick_macro_f1")
    aggregate = {
        variant: {
            name: _mean_std([_metric(summaries[variant][seed], name) for seed in SEEDS])
            for name in metrics
        }
        for variant in POSE_VARIANTS
    }
    paired = {
        "POSE_TACTILE_minus_POSE_ONLY": {
            name: {
                **_mean_std(
                    [
                        _metric(summaries["POSE_TACTILE"][seed], name)
                        - _metric(summaries["POSE_ONLY"][seed], name)
                        for seed in SEEDS
                    ]
                ),
                "wins": sum(
                    _metric(summaries["POSE_TACTILE"][seed], name)
                    > _metric(summaries["POSE_ONLY"][seed], name)
                    for seed in SEEDS
                ),
            }
            for name in metrics
        }
    }
    common = dict(zip(lineage_fields, next(iter(lineage_values))))
    result = {
        "schema_version": POSE_AGGREGATE_SCHEMA,
        "status": "complete",
        "purpose": "validation_only_no_video_pose_tactile_comparison_test_sealed",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "variants": list(POSE_VARIANTS),
        "seeds": list(SEEDS),
        "jobs": len(records),
        "aggregate": aggregate,
        "paired": paired,
        "controls": {
            "same_data_and_split": True,
            "same_recipe": True,
            "paired_initialization_within_seed": True,
            "parameter_count_matched": True,
            "updates_matched": True,
            "same_fine_26_label_space": True,
            "no_video_reader": True,
            "visual_masks_false": True,
            "hand_loss_weight": 0.0,
            "full_episode_validation": True,
            "test_sealed": True,
            "lineage_equal_across_jobs": True,
        },
        "lineage": {**common, "jobs": records},
    }
    _atomic_json(output, result)
    _atomic_json(
        output.parent / "POSE_COMPLETE.json",
        {
            "schema_version": POSE_AGGREGATE_SCHEMA,
            "status": "complete",
            "aggregate": str(output.resolve()),
            "aggregate_sha256": sha256_file(output),
            "test_manifest_read": False,
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Formal500h no-video pose ablations")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    validate_release = sub.add_parser("validate-release")
    validate_release.add_argument("--data-dir", type=Path, required=True)
    validate_release.add_argument("--pose-release", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--phase-release", type=Path, required=True)
    train.add_argument("--pose-release", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--variant", choices=POSE_VARIANTS, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--precision", default=FROZEN_PRECISION)
    train.add_argument("--batch-size", type=int, default=FROZEN_BATCH_SIZE)
    train.add_argument("--num-workers", type=int, default=FROZEN_NUM_WORKERS)
    train.add_argument("--epochs", type=int, default=FROZEN_EPOCHS)
    train.add_argument("--learning-rate", type=float, default=FROZEN_LEARNING_RATE)
    train.add_argument("--calibration-workers", type=int, default=FROZEN_CALIBRATION_WORKERS)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--runs-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        result = build_pose_release(data_dir=args.data_dir, output=args.output)
        print(json.dumps({"event": "pose_release_complete", "status": result["status"], "output": str(args.output)}), flush=True)
    elif args.command == "validate-release":
        _validated_pose_release(args.pose_release, args.data_dir)
        print(
            json.dumps(
                {
                    "event": "pose_release_valid",
                    "status": "complete",
                    "test_manifest_read": False,
                }
            ),
            flush=True,
        )
    elif args.command == "train":
        result = train_pose_job(
            data_dir=args.data_dir,
            phase_release=args.phase_release,
            pose_release=args.pose_release,
            output_dir=args.output_dir,
            variant=args.variant,
            seed=args.seed,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        print(json.dumps({"event": "pose_training_complete", "variant": args.variant, "seed": args.seed, "status": result.get("status")}), flush=True)
    else:
        result = aggregate_pose(runs_root=args.runs_root, output=args.output)
        print(json.dumps({"event": "pose_aggregate_complete", "status": result["status"], "jobs": result["jobs"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
