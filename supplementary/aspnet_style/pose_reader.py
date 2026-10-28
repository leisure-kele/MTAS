"""Frozen Pose reader adaptation: pure P performs no V or K HDF5 reads."""
from __future__ import annotations
import numpy as np
import torch
from atomseg.formal500h_pose_train import PoseFormal500hDataset
from atomseg.formal500h_runtime import _soft_boundary
from atomseg.formal500h_protocol import NUM_CLASSES
from atomseg.formal500h_clock import clock_metadata, downsample_targets_60_to_30

class PurePoseDataset(PoseFormal500hDataset):
    def __init__(self, manifest, statistics, *, expected_statistics_schema, enabled_modalities, hdf5_cache_size=8):
        modalities = frozenset(enabled_modalities)
        if "skeleton" not in modalities or not modalities <= {"skeleton", "kinematics", "tactile"}:
            raise ValueError("invalid pure pose reader modalities")
        super().__init__(manifest, statistics, expected_statistics_schema=expected_statistics_schema,
                         enabled_modalities=sorted(modalities | {"kinematics"}), hdf5_cache_size=hdf5_cache_size)
        self.enabled_modalities = modalities

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
            origin = int(handle["timeline/time_ns"][0])
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

            if "kinematics" in self.enabled_modalities:
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
            else:
                kinematics_valid = np.zeros(steps, dtype=np.bool_)
                kinematics = np.zeros((steps, 48), dtype=np.float32)
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
            "episode_start_time_ns": origin,
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
