from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .baselines import build_model
from .formal500h_clock import clock_metadata, downsample_targets_60_to_30
from .losses import ContactTASLoss


EPISODE_SCHEMA = "formal500h_training_episode_v3"
NUM_CLASSES = 26
OUTPUT_RATE_HZ = 30
REQUIRED_DATASETS = (
    "timeline/time_ns",
    "targets/action_class",
    "targets/action_hand",
    "targets/action_instance",
    "targets/boundary_instance",
    "targets/boundary_class_change",
    "vision/features",
    "vision/valid",
    "vision/index_60",
    "vision/aligned_valid_60",
    "vision/freshness_ms_60",
    "sensors/tactile/value",
    "sensors/tactile/valid",
    "sensors/tactile/quality_raw460",
    "sensors/skeleton/relative",
    "sensors/skeleton/valid",
    "sensors/skeleton/joint_valid",
    "sensors/kinematics/value",
    "sensors/kinematics/valid",
    "sensors/pose/quality",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty Formal500h window manifest: {path}")
    return rows


def _soft_boundary(hard: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    output = hard.astype(np.float32)
    radius = max(1, int(math.ceil(4.0 * sigma)))
    for center in np.flatnonzero(hard):
        start = max(0, int(center) - radius)
        end = min(len(output), int(center) + radius + 1)
        distance = np.arange(start, end, dtype=np.float32) - float(center)
        output[start:end] = np.maximum(
            output[start:end], np.exp(-0.5 * (distance / sigma) ** 2)
        )
    return output


class Formal500hDataset(Dataset[dict[str, Any]]):
    """Direct reader for the frozen Formal500h episode-v3 HDF5 contract."""

    def __init__(
        self,
        manifest: str | Path,
        statistics: str | Path,
        *,
        expected_statistics_schema: str,
    ):
        self.manifest_path = Path(manifest)
        self.entries = _read_jsonl(self.manifest_path)
        with np.load(statistics, allow_pickle=False) as values:
            schema = str(values["schema_version"].item())
            if schema != expected_statistics_schema:
                raise ValueError(f"unexpected Formal500h statistics schema: {schema}")
            self.tactile_mean = values["tactile_mean"].astype(np.float32)
            self.tactile_std = values["tactile_std"].astype(np.float32)
            self.skeleton_mean = values["skeleton_mean"].astype(np.float32)
            self.skeleton_std = values["skeleton_std"].astype(np.float32)
            self.kinematics_mean = values["kinematics_mean"].astype(np.float32)
            self.kinematics_std = values["kinematics_std"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _normalized(
        value: np.ndarray, mean: np.ndarray, standard: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        output = np.clip((value - mean) / standard, -10.0, 10.0).astype(np.float32)
        weight = valid.astype(np.float32)
        while weight.ndim < output.ndim:
            weight = weight[..., None]
        return output * weight

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.entries[index]
        start = int(row["start_frame_60"])
        end = int(row["end_frame_60_exclusive"])
        if start % 2:
            raise ValueError("Formal500h windows must start on an even 60 Hz frame")
        selection = slice(start, end, 2)
        with h5py.File(row["cache_path"], "r") as handle:
            schema = handle.attrs.get("schema_version", "")
            if isinstance(schema, bytes):
                schema = schema.decode("utf-8")
            if str(schema) != EPISODE_SCHEMA or not bool(handle.attrs.get("complete", False)):
                raise ValueError(f"invalid Formal500h HDF5: {row['cache_path']}")
            # Targets are stored at 60 Hz.  Convert all target streams through
            # the canonical clock helper before any loss/evaluation code sees
            # them.  In particular, boundaries use ceiling-bin/OR semantics;
            # the historical pair-and-reshape trick silently moved odd-frame
            # events to the preceding tick.
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
            instance = np.asarray(targets["action_instance"], dtype=np.int64)
            boundary_hard = np.asarray(targets["boundary_instance"], dtype=np.bool_)
            boundary_change = np.asarray(
                targets["boundary_class_change"], dtype=np.bool_
            )
            hand = np.asarray(
                handle["targets/action_hand"][selection], dtype=np.int64
            )
            steps = int(action.size)
            raw_index = np.asarray(handle["vision/index_60"][selection], dtype=np.int64)
            aligned_valid = np.asarray(handle["vision/aligned_valid_60"][selection], dtype=np.bool_)
            valid_indices = raw_index[raw_index >= 0]
            if valid_indices.size:
                vision_start = int(valid_indices.min())
                vision_end = int(valid_indices.max()) + 1
            else:
                vision_start = 0
                vision_end = min(1, int(handle["vision/features"].shape[0]))
            vision = np.asarray(handle["vision/features"][vision_start:vision_end], dtype=np.float32)
            vision_cache_valid = np.asarray(handle["vision/valid"][vision_start:vision_end], dtype=np.bool_)
            if not len(vision):
                vision = np.zeros((1, 3, 1536), dtype=np.float32)
                vision_cache_valid = np.zeros((1, 3), dtype=np.bool_)
                vision_start = 0
            vision_index = np.where(raw_index >= 0, raw_index - vision_start, -1)
            vision_index[(vision_index < 0) | (vision_index >= len(vision))] = -1
            vision_valid = aligned_valid & (vision_index >= 0)
            vision_freshness = np.asarray(handle["vision/freshness_ms_60"][selection], dtype=np.float32)
            vision_freshness = np.nan_to_num(vision_freshness, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)

            tactile_valid = np.asarray(handle["sensors/tactile/valid"][selection], dtype=np.bool_)
            tactile = self._normalized(
                np.asarray(handle["sensors/tactile/value"][selection], dtype=np.float32),
                self.tactile_mean,
                self.tactile_std,
                tactile_valid,
            )
            raw_quality = np.asarray(handle["sensors/tactile/quality_raw460"][selection], dtype=np.float32)
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

            skeleton_valid = np.asarray(handle["sensors/skeleton/valid"][selection], dtype=np.bool_)
            joint_valid = np.asarray(handle["sensors/skeleton/joint_valid"][selection], dtype=np.bool_)
            joint_valid &= skeleton_valid[..., None]
            skeleton = self._normalized(
                np.asarray(handle["sensors/skeleton/relative"][selection], dtype=np.float32),
                self.skeleton_mean,
                self.skeleton_std,
                joint_valid,
            )
            kinematics_valid = np.asarray(handle["sensors/kinematics/valid"][selection], dtype=np.bool_)
            kinematics = self._normalized(
                np.asarray(handle["sensors/kinematics/value"][selection], dtype=np.float32),
                self.kinematics_mean,
                self.kinematics_std,
                kinematics_valid,
            )
            pose_quality = np.asarray(handle["sensors/pose/quality"][selection], dtype=np.float32)
            time_ns = np.asarray(handle["timeline/time_ns"][selection], dtype=np.int64)

        if np.any((action < 0) | (action >= NUM_CLASSES)):
            raise ValueError("Formal500h target escaped the frozen 26-class ontology")
        boundary_type = np.zeros(steps, dtype=np.int64)
        boundary_type[boundary_hard & boundary_change] = 1
        boundary_type[boundary_hard & ~boundary_change] = 2
        mask = np.ones(steps, dtype=np.bool_)
        global_frame_30 = start // 2 + np.arange(steps, dtype=np.int64)

        def tensor(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(value)).to(dtype=dtype)

        return {
            "episode_id": str(row["episode_id"]),
            "time_ns": tensor(time_ns, torch.long),
            # Kept under the generic stitching field name; values are the
            # contiguous formal 30 Hz evaluation ticks.
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


def collate_formal500h(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty Formal500h batch")
    batch_size = len(samples)
    max_steps = max(int(sample["steps"]) for sample in samples)
    max_vision = max(int(sample["vision_features"].shape[0]) for sample in samples)
    _, views, vision_dim = samples[0]["vision_features"].shape
    _, hands, tactile_dim = samples[0]["tactile"].shape
    _, skeleton_hands, joints, joint_dim = samples[0]["skeleton"].shape
    if (views, hands, skeleton_hands) != (3, 2, 2):
        raise ValueError("Formal500h requires three views and two hands")
    quality_dim = int(samples[0]["tactile_quality"].shape[-1])
    kinematic_dim = int(samples[0]["kinematics"].shape[-1])
    pose_dim = int(samples[0]["pose_quality"].shape[-1])
    batch: dict[str, Any] = {
        "episode_id": [sample["episode_id"] for sample in samples],
        "read_start_step": [int(sample["read_start_step"]) for sample in samples],
        "clock_metadata": dict(clock_metadata()),
        "time_ns": torch.zeros(batch_size, max_steps, dtype=torch.long),
        "global_frame_60": torch.full((batch_size, max_steps), -1, dtype=torch.long),
        "vision_features": torch.zeros(batch_size, max_vision, 3, vision_dim),
        "vision_cache_valid": torch.zeros(batch_size, max_vision, 3, dtype=torch.bool),
        "vision_index_60": torch.full((batch_size, max_steps, 3), -1, dtype=torch.long),
        "vision_valid_60": torch.zeros(batch_size, max_steps, 3, dtype=torch.bool),
        "vision_freshness_ms": torch.zeros(batch_size, max_steps, 3),
        "tactile": torch.zeros(batch_size, max_steps, 2, tactile_dim),
        "tactile_valid": torch.zeros(batch_size, max_steps, 2, dtype=torch.bool),
        "tactile_quality": torch.zeros(batch_size, max_steps, 2, quality_dim),
        "skeleton": torch.zeros(batch_size, max_steps, 2, joints, joint_dim),
        "skeleton_valid": torch.zeros(batch_size, max_steps, 2, dtype=torch.bool),
        "skeleton_joint_valid": torch.zeros(batch_size, max_steps, 2, joints, dtype=torch.bool),
        "kinematics": torch.zeros(batch_size, max_steps, kinematic_dim),
        "kinematics_valid": torch.zeros(batch_size, max_steps, dtype=torch.bool),
        "pose_quality": torch.zeros(batch_size, max_steps, pose_dim),
        "atomic_target": torch.full((batch_size, max_steps), -100, dtype=torch.long),
        "hand_target": torch.full((batch_size, max_steps), -100, dtype=torch.long),
        "atomic_instance_id": torch.full((batch_size, max_steps), -100, dtype=torch.long),
        "subtask_target": torch.full((batch_size, max_steps), -100, dtype=torch.long),
        "boundary_target": torch.zeros(batch_size, max_steps),
        "boundary_hard": torch.zeros(batch_size, max_steps, dtype=torch.bool),
        "boundary_type": torch.zeros(batch_size, max_steps, dtype=torch.long),
        "boundary_eval_mask": torch.zeros(batch_size, max_steps, dtype=torch.bool),
        "core_mask": torch.zeros(batch_size, max_steps, dtype=torch.bool),
        "loss_mask": torch.zeros(batch_size, max_steps, dtype=torch.bool),
        "boundary_loss_mask": torch.zeros(batch_size, max_steps, dtype=torch.bool),
        "sequence_mask": torch.zeros(batch_size, max_steps, dtype=torch.bool),
    }
    expected_clock = dict(clock_metadata())
    for sample in samples:
        sample_clock = sample.get("clock_metadata")
        if sample_clock is not None and sample_clock != expected_clock:
            raise ValueError("Formal500h sample clock metadata is inconsistent")
    temporal_names = (
        "time_ns", "global_frame_60", "vision_index_60", "vision_valid_60",
        "vision_freshness_ms", "tactile", "tactile_valid", "tactile_quality",
        "skeleton", "skeleton_valid", "skeleton_joint_valid", "kinematics",
        "kinematics_valid", "pose_quality", "atomic_target", "hand_target",
        "atomic_instance_id", "subtask_target", "boundary_target", "boundary_hard",
        "boundary_type", "boundary_eval_mask", "core_mask", "loss_mask",
        "boundary_loss_mask",
    )
    for row, sample in enumerate(samples):
        steps = int(sample["steps"])
        vision_steps = int(sample["vision_features"].shape[0])
        for name in temporal_names:
            batch[name][row, :steps] = sample[name]
        batch["vision_features"][row, :vision_steps] = sample["vision_features"]
        batch["vision_cache_valid"][row, :vision_steps] = sample["vision_cache_valid"]
        batch["sequence_mask"][row, :steps] = True
    batch["loss_mask"] &= batch["sequence_mask"]
    batch["boundary_loss_mask"] &= batch["sequence_mask"]
    return batch


class Formal500hModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.backbone = build_model(config)
        self.hand_head = nn.Linear(int(config["decoder_channels"]), 3)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        output = self.backbone(batch, apply_modality_dropout=False)
        output["hand_logits"] = self.hand_head(output["decoder_features"])
        return output


def formal500h_loss_terms(
    criterion: ContactTASLoss,
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    hand_weight: float,
    hand_loss_enabled: bool | None = None,
) -> dict[str, torch.Tensor]:
    terms = criterion.supervised(output, batch)
    # H0 keeps the optional hand output for checkpoint/interface
    # compatibility, but does not compute an auxiliary objective.  Apart
    # from being faster, this prevents an apparently harmless zero-weight CE
    # from being reported as a trained signal.  Legacy H1 callers can retain
    # the old behavior by passing a non-zero weight (or explicitly enabling
    # the loss).
    if hand_loss_enabled is None:
        hand_loss_enabled = not math.isclose(float(hand_weight), 0.0, abs_tol=1.0e-15)
    if not hand_loss_enabled:
        terms["hand"] = terms["loss"].new_zeros(())
        return terms
    hand_mask = batch["loss_mask"] & (batch["hand_target"] >= 0)
    safe_hand = batch["hand_target"].masked_fill(~hand_mask, -100)
    hand_loss = F.cross_entropy(
        output["hand_logits"].transpose(1, 2),
        safe_hand,
        ignore_index=-100,
        reduction="none",
    )
    hand_loss = (hand_loss * hand_mask.to(hand_loss.dtype)).sum() / hand_mask.sum().clamp_min(1)
    terms["hand"] = hand_loss
    terms["loss"] = terms["loss"] + float(hand_weight) * hand_loss
    return terms


def formal500h_class_weights(statistics: Path) -> list[float]:
    with np.load(statistics, allow_pickle=False) as values:
        count = values["action_count"].astype(np.float64)
    if count.shape != (NUM_CLASSES,) or np.any(count <= 0):
        raise ValueError("Formal500h train statistics must contain all 26 classes")
    weight = 1.0 / np.sqrt(count)
    weight /= weight.mean()
    return weight.astype(np.float32).tolist()
