from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch

from .formal500h_runtime import (
    EPISODE_SCHEMA,
    NUM_CLASSES,
    Formal500hDataset,
    _soft_boundary,
    collate_formal500h,
)
from .formal500h_clock import clock_metadata, downsample_targets_60_to_30


ALL_MODALITIES = frozenset({"vision", "tactile", "skeleton", "kinematics"})


class VariantAwareFormal500hDataset(Formal500hDataset):
    """Formal500h reader that avoids HDF5 reads for disabled model modalities.

    Disabled tensors retain the exact public batch contract but contain zeros and
    false validity.  The frozen B0/B1/B2 models already force those validity masks
    to false, so this removes I/O without changing any enabled model input.

    The optional handle cache is process-local.  DataLoader workers never share an
    h5py handle; pickling clears the cache and each worker opens its own bounded LRU.
    """

    def __init__(
        self,
        manifest: str | Path,
        statistics: str | Path,
        *,
        expected_statistics_schema: str,
        enabled_modalities: Sequence[str],
        hdf5_cache_size: int = 8,
    ):
        super().__init__(
            manifest,
            statistics,
            expected_statistics_schema=expected_statistics_schema,
        )
        enabled = frozenset(str(value).strip().lower() for value in enabled_modalities)
        unknown = enabled.difference(ALL_MODALITIES)
        if unknown or "vision" not in enabled:
            raise ValueError(
                "variant-aware Formal500h requires vision and only known modalities; "
                f"got enabled={sorted(enabled)}, unknown={sorted(unknown)}"
            )
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
            time_ns = np.asarray(handle["timeline/time_ns"][selection], dtype=np.int64)
            # Keep target conversion identical to the legacy reader.  The
            # source HDF5 is 60 Hz, while the model contract is 30 Hz; event
            # markers must use the canonical ceiling-bin/OR mapping rather
            # than the old pair-and-reshape floor convention.
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

            raw_index = np.asarray(handle["vision/index_60"][selection], dtype=np.int64)
            aligned_valid = np.asarray(
                handle["vision/aligned_valid_60"][selection], dtype=np.bool_
            )
            valid_indices = raw_index[raw_index >= 0]
            if valid_indices.size:
                vision_start = int(valid_indices.min())
                vision_end = int(valid_indices.max()) + 1
            else:
                vision_start = 0
                vision_end = min(1, int(handle["vision/features"].shape[0]))
            vision = np.asarray(
                handle["vision/features"][vision_start:vision_end], dtype=np.float32
            )
            vision_cache_valid = np.asarray(
                handle["vision/valid"][vision_start:vision_end], dtype=np.bool_
            )
            if not len(vision):
                vision = np.zeros((1, 3, 1536), dtype=np.float32)
                vision_cache_valid = np.zeros((1, 3), dtype=np.bool_)
                vision_start = 0
            vision_index = np.where(raw_index >= 0, raw_index - vision_start, -1)
            vision_index[(vision_index < 0) | (vision_index >= len(vision))] = -1
            vision_valid = aligned_valid & (vision_index >= 0)
            vision_freshness = np.asarray(
                handle["vision/freshness_ms_60"][selection], dtype=np.float32
            )
            vision_freshness = np.nan_to_num(
                vision_freshness, nan=1.0e6, posinf=1.0e6, neginf=1.0e6
            )

            if "tactile" in self.enabled_modalities:
                tactile_valid = np.asarray(
                    handle["sensors/tactile/valid"][selection], dtype=np.bool_
                )
                tactile = self._normalized(
                    np.asarray(
                        handle["sensors/tactile/value"][selection], dtype=np.float32
                    ),
                    self.tactile_mean,
                    self.tactile_std,
                    tactile_valid,
                )
                raw_quality = np.asarray(
                    handle["sensors/tactile/quality_raw460"][selection],
                    dtype=np.float32,
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
                tactile_valid = np.zeros((steps, self.tactile_mean.shape[0]), dtype=np.bool_)
                tactile_quality = np.zeros(
                    (steps, self.tactile_mean.shape[0], 5), dtype=np.float32
                )

            if "skeleton" in self.enabled_modalities:
                skeleton_valid = np.asarray(
                    handle["sensors/skeleton/valid"][selection], dtype=np.bool_
                )
                joint_valid = np.asarray(
                    handle["sensors/skeleton/joint_valid"][selection], dtype=np.bool_
                )
                joint_valid &= skeleton_valid[..., None]
                skeleton = self._normalized(
                    np.asarray(
                        handle["sensors/skeleton/relative"][selection], dtype=np.float32
                    ),
                    self.skeleton_mean,
                    self.skeleton_std,
                    joint_valid,
                )
            else:
                skeleton = np.zeros((steps, *self.skeleton_mean.shape), dtype=np.float32)
                skeleton_valid = np.zeros(
                    (steps, self.skeleton_mean.shape[0]), dtype=np.bool_
                )
                joint_valid = np.zeros(
                    (steps, *self.skeleton_mean.shape[:-1]), dtype=np.bool_
                )

            if "kinematics" in self.enabled_modalities:
                kinematics_valid = np.asarray(
                    handle["sensors/kinematics/valid"][selection], dtype=np.bool_
                )
                kinematics = self._normalized(
                    np.asarray(
                        handle["sensors/kinematics/value"][selection], dtype=np.float32
                    ),
                    self.kinematics_mean,
                    self.kinematics_std,
                    kinematics_valid,
                )
                pose_quality = np.asarray(
                    handle["sensors/pose/quality"][selection], dtype=np.float32
                )
            else:
                kinematics = np.zeros(
                    (steps, *self.kinematics_mean.shape), dtype=np.float32
                )
                kinematics_valid = np.zeros(steps, dtype=np.bool_)
                pose_quality = np.zeros((steps, 6), dtype=np.float32)

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


_TEMPORAL_NAMES = (
    "time_ns",
    "global_frame_60",
    "vision_index_60",
    "vision_valid_60",
    "vision_freshness_ms",
    "tactile",
    "tactile_valid",
    "tactile_quality",
    "skeleton",
    "skeleton_valid",
    "skeleton_joint_valid",
    "kinematics",
    "kinematics_valid",
    "pose_quality",
    "atomic_target",
    "hand_target",
    "atomic_instance_id",
    "subtask_target",
    "boundary_target",
    "boundary_hard",
    "boundary_type",
    "boundary_eval_mask",
    "core_mask",
    "loss_mask",
    "boundary_loss_mask",
)


def collate_formal500h_fast(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack equal-shape windows; retain the frozen padding path for tails."""

    if not samples:
        raise ValueError("cannot collate an empty Formal500h batch")
    first = samples[0]
    tensor_names = (*_TEMPORAL_NAMES, "vision_features", "vision_cache_valid")
    if any(
        tuple(sample[name].shape) != tuple(first[name].shape)
        for sample in samples[1:]
        for name in tensor_names
    ):
        return collate_formal500h(samples)
    batch: dict[str, Any] = {
        "episode_id": [sample["episode_id"] for sample in samples],
        "read_start_step": [int(sample["read_start_step"]) for sample in samples],
        "clock_metadata": dict(clock_metadata()),
    }
    expected_clock = dict(clock_metadata())
    for sample in samples:
        sample_clock = sample.get("clock_metadata")
        if sample_clock is not None and sample_clock != expected_clock:
            raise ValueError("Formal500h sample clock metadata is inconsistent")
    for name in tensor_names:
        batch[name] = torch.stack([sample[name] for sample in samples], dim=0)
    batch["sequence_mask"] = torch.ones(
        len(samples), int(first["steps"]), dtype=torch.bool
    )
    batch["loss_mask"] &= batch["sequence_mask"]
    batch["boundary_loss_mask"] &= batch["sequence_mask"]
    return batch
