from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .artifacts import (
    artifact_record,
    atomic_write_json,
    require_sha256,
    sha256_file,
    sha256_json,
    verify_artifact_record,
)
from .mcap_io import (
    JSON_SENSOR_TOPICS,
    assert_dual_source_labels_equal,
    iter_episode_json_streams,
    read_dual_source_labels,
    read_first_json_by_topic,
)
from .ontology import FrozenOntology, load_frozen_ontology


SENSOR_CACHE_SCHEMA_VERSION = "legacy20h_sensor_cache_v1"
DEFAULT_QUARANTINE_EPISODES = frozenset({"T001_00095", "T008_00352"})
POSE_TOPICS = ("/pose/head", "/pose/wrist/left", "/pose/wrist/right")
HAND_BY_TOPIC = {
    "/tactile/raw/left": 0,
    "/tactile/raw/right": 1,
    "/hand_skeleton/left": 0,
    "/hand_skeleton/right": 1,
}


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - optional production dependency
        raise RuntimeError("HDF5 export requires h5py") from error
    return h5py


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    safe = np.where(norm > 1.0e-8, quaternion / np.maximum(norm, 1.0e-8), 0.0)
    safe[..., 3] = np.where(norm[..., 0] > 1.0e-8, safe[..., 3], 1.0)
    return safe.astype(np.float32)


def _continuous_quaternion(quaternion: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = _normalize_quaternion(quaternion)
    for index in range(1, len(output)):
        if valid[index] and valid[index - 1] and np.dot(output[index - 1], output[index]) < 0:
            output[index] *= -1.0
    return output


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    output = quaternion.copy()
    output[..., :3] *= -1.0
    return output


def _quat_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(first, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(second, -1, 0)
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector_quaternion = np.concatenate(
        [vector, np.zeros(vector.shape[:-1] + (1,), dtype=vector.dtype)], axis=-1
    )
    return _quat_multiply(
        _quat_multiply(quaternion, vector_quaternion), _quat_conjugate(quaternion)
    )[..., :3]


def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = _normalize_quaternion(quaternion)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _rotation_6d(quaternion: np.ndarray) -> np.ndarray:
    matrix = _quat_to_matrix(quaternion)
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,)).astype(np.float32)


def _backward_difference(
    values: np.ndarray, time_ns: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros_like(values, dtype=np.float32)
    output_valid = np.zeros(len(values), dtype=np.bool_)
    if len(values) < 2:
        return output, output_valid
    dt = np.diff(time_ns.astype(np.float64)) / 1.0e9
    pair_valid = valid[1:] & valid[:-1] & np.isfinite(dt) & (dt > 1.0e-6)
    difference = values[1:] - values[:-1]
    output[1:][pair_valid] = (
        difference[pair_valid] / dt[pair_valid].reshape((-1,) + (1,) * (values.ndim - 1))
    )
    output_valid[1:] = pair_valid
    return output, output_valid


def _angular_velocity(
    quaternion: np.ndarray, time_ns: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((len(quaternion), 3), dtype=np.float32)
    output_valid = np.zeros(len(quaternion), dtype=np.bool_)
    if len(quaternion) < 2:
        return output, output_valid
    delta = _quat_multiply(_quat_conjugate(quaternion[:-1]), quaternion[1:])
    delta = _normalize_quaternion(delta)
    delta[delta[:, 3] < 0] *= -1.0
    vector = delta[:, :3]
    vector_norm = np.linalg.norm(vector, axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[:, 3], -1.0, 1.0))
    axis = np.zeros_like(vector)
    nonzero = vector_norm > 1.0e-8
    axis[nonzero] = vector[nonzero] / vector_norm[nonzero, None]
    dt = np.diff(time_ns.astype(np.float64)) / 1.0e9
    pair_valid = valid[1:] & valid[:-1] & np.isfinite(dt) & (dt > 1.0e-6)
    output[1:][pair_valid] = axis[pair_valid] * (
        angle[pair_valid] / dt[pair_valid]
    )[:, None]
    output_valid[1:] = pair_valid
    return output, output_valid


def _soft_boundary_target(hard: np.ndarray, sigma_ticks: float) -> np.ndarray:
    if sigma_ticks <= 0:
        return hard.astype(np.float32)
    target = np.zeros(len(hard), dtype=np.float32)
    radius = max(1, int(math.ceil(4.0 * sigma_ticks)))
    for center in np.flatnonzero(hard):
        start = max(0, int(center) - radius)
        end = min(len(hard), int(center) + radius + 1)
        distance = np.arange(start, end, dtype=np.float32) - float(center)
        target[start:end] = np.maximum(
            target[start:end], np.exp(-0.5 * (distance / sigma_ticks) ** 2)
        )
    return target


def build_label_arrays(
    segments: Sequence[Mapping[str, Any]],
    *,
    num_ticks: int,
    ontology: FrozenOntology,
    boundary_sigma_ticks: float,
    require_dense: bool = True,
) -> dict[str, np.ndarray]:
    ignore_index = ontology.ignore_index
    target = np.full(num_ticks, ignore_index, dtype=np.int64)
    instance = np.full(num_ticks, ignore_index, dtype=np.int64)
    boundary_hard = np.zeros(num_ticks, dtype=np.bool_)
    boundary_type = np.zeros(num_ticks, dtype=np.int8)
    previous_end = 0
    previous_index: int | None = None

    for instance_id, raw_segment in enumerate(segments):
        name = str(raw_segment["name"])
        if name not in ontology.index_by_name:
            raise ValueError(f"Unknown legacy20h action {name!r}")
        start = int(raw_segment["start_frame_60"])
        end = int(raw_segment["end_frame_60_exclusive"])
        if not 0 <= start < end <= num_ticks:
            raise ValueError(f"Atomic interval [{start},{end}) is outside T={num_ticks}")
        if start < previous_end:
            raise ValueError(f"Overlapping/non-monotonic atomic interval starts at {start}")
        if require_dense and start != previous_end:
            raise ValueError(f"Atomic labels contain a gap [{previous_end},{start})")
        action_index = ontology.index_by_name[name]
        target[start:end] = action_index
        instance[start:end] = instance_id
        if instance_id > 0:
            boundary_hard[start] = True
            boundary_type[start] = 2 if action_index == previous_index else 1
        previous_end = end
        previous_index = action_index

    if require_dense and segments and np.any(target[:previous_end] == ignore_index):
        raise ValueError("Atomic labels are not dense inside the annotated interval")
    label_mask = target != ignore_index
    boundary_eval_mask = label_mask.copy()
    boundary_loss_mask = label_mask.copy()
    valid_indices = np.flatnonzero(label_mask)
    if valid_indices.size:
        first = int(valid_indices[0])
        last = int(valid_indices[-1])
        boundary_eval_mask[first] = False
        boundary_eval_mask[last] = False
        boundary_loss_mask[first] = False
        boundary_loss_mask[last] = False
    boundary_target = _soft_boundary_target(boundary_hard, boundary_sigma_ticks)
    boundary_target *= boundary_loss_mask.astype(np.float32)
    return {
        "atomic_target": target,
        "atomic_instance_id": instance,
        "boundary_target": boundary_target,
        "boundary_hard": boundary_hard,
        "boundary_type": boundary_type,
        "boundary_eval_mask": boundary_eval_mask,
        "label_mask": label_mask,
        "boundary_loss_mask": boundary_loss_mask,
        "core_mask": np.ones(num_ticks, dtype=np.bool_),
    }


@dataclass
class RawEpisodeSensors:
    episode_id: str
    num_ticks: int
    time_ns: np.ndarray
    tactile: np.ndarray
    tactile_valid: np.ndarray
    tactile_quality: np.ndarray
    skeleton: np.ndarray
    skeleton_valid: np.ndarray
    skeleton_joint_valid: np.ndarray
    kinematics: np.ndarray
    kinematics_valid: np.ndarray
    pose_quality: np.ndarray


def _frame_index(message: Mapping[str, Any], log_time_ns: int, output_rate_hz: float) -> int:
    value = message.get("frame_60")
    if isinstance(value, int):
        return value
    return int(round(log_time_ns * output_rate_hz / 1.0e9))


def _build_tactile_quality(
    tactile: np.ndarray,
    valid: np.ndarray,
    *,
    jump_threshold: float,
    saturation_threshold: float | None,
) -> np.ndarray:
    # [packet_valid, zero_fraction, unchanged_fraction, jump_fraction, saturation_fraction]
    quality = np.zeros(tactile.shape[:2] + (5,), dtype=np.float32)
    quality[..., 0] = valid.astype(np.float32)
    quality[..., 1] = np.mean(tactile == 0, axis=-1)
    delta = np.zeros_like(tactile)
    delta[1:] = tactile[1:] - tactile[:-1]
    quality[..., 2] = np.mean(np.abs(delta) <= 1.0e-6, axis=-1)
    quality[..., 3] = np.mean(np.abs(delta) > jump_threshold, axis=-1)
    pair_valid = np.zeros_like(valid)
    pair_valid[1:] = valid[1:] & valid[:-1]
    quality[..., 2] *= pair_valid.astype(np.float32)
    quality[..., 3] *= pair_valid.astype(np.float32)
    if saturation_threshold is not None:
        quality[..., 4] = np.mean(np.abs(tactile) >= saturation_threshold, axis=-1)
    quality[~valid] = 0.0
    return quality


def _relative_skeleton(
    skeleton_world: np.ndarray, joint_valid: np.ndarray
) -> np.ndarray:
    output = skeleton_world.copy()
    for hand in range(2):
        root_position = skeleton_world[:, hand, 0, :3]
        root_quaternion = _continuous_quaternion(
            skeleton_world[:, hand, 0, 3:7], joint_valid[:, hand, 0]
        )
        inverse = _quat_conjugate(root_quaternion)
        relative_position = skeleton_world[:, hand, :, :3] - root_position[:, None, :]
        output[:, hand, :, :3] = _quat_rotate(
            inverse[:, None, :], relative_position
        )
        joint_quaternion = skeleton_world[:, hand, :, 3:7]
        output[:, hand, :, 3:7] = _normalize_quaternion(
            _quat_multiply(inverse[:, None, :], joint_quaternion)
        )
        output[:, hand] *= joint_valid[:, hand, :, None].astype(np.float32)
    return output.astype(np.float32)


def _build_kinematics(
    pose_position: np.ndarray,
    pose_quaternion: np.ndarray,
    pose_valid: np.ndarray,
    skeleton_relative: np.ndarray,
    joint_valid: np.ndarray,
    time_ns: np.ndarray,
    *,
    pose_jump_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for stream in range(3):
        pose_quaternion[:, stream] = _continuous_quaternion(
            pose_quaternion[:, stream], pose_valid[:, stream]
        )
    head_position = pose_position[:, 0]
    head_quaternion = pose_quaternion[:, 0]
    head_inverse = _quat_conjugate(head_quaternion)
    per_hand: list[np.ndarray] = []
    hand_valids: list[np.ndarray] = []
    relative_positions: list[np.ndarray] = []
    linear_velocities: list[np.ndarray] = []

    for hand, pose_stream in enumerate((1, 2)):
        valid = pose_valid[:, 0] & pose_valid[:, pose_stream]
        relative_position = _quat_rotate(
            head_inverse, pose_position[:, pose_stream] - head_position
        ).astype(np.float32)
        relative_quaternion = _normalize_quaternion(
            _quat_multiply(head_inverse, pose_quaternion[:, pose_stream])
        )
        relative_quaternion = _continuous_quaternion(relative_quaternion, valid)
        rotation_6d = _rotation_6d(relative_quaternion)
        velocity, velocity_valid = _backward_difference(relative_position, time_ns, valid)
        angular_velocity, angular_valid = _angular_velocity(
            relative_quaternion, time_ns, valid
        )
        acceleration, acceleration_valid = _backward_difference(
            velocity, time_ns, velocity_valid
        )
        jerk, jerk_valid = _backward_difference(
            acceleration, time_ns, acceleration_valid
        )

        # The exporter preserves flat-460 touch and only derives geometry from
        # verified skeleton joints.  Indices 4/9 are thumb/index tips in the
        # audited 25-joint ordering; validity prevents a missing joint becoming 0.
        thumb_index = min(4, skeleton_relative.shape[2] - 1)
        index_index = min(9, skeleton_relative.shape[2] - 1)
        aperture_valid = joint_valid[:, hand, thumb_index] & joint_valid[:, hand, index_index]
        aperture = np.linalg.norm(
            skeleton_relative[:, hand, thumb_index, :3]
            - skeleton_relative[:, hand, index_index, :3],
            axis=-1,
        ).astype(np.float32)
        aperture[~aperture_valid] = 0.0
        aperture_speed, aperture_speed_valid = _backward_difference(
            aperture[:, None], time_ns, aperture_valid
        )
        full_valid = (
            valid
            & velocity_valid
            & angular_valid
            & acceleration_valid
            & jerk_valid
            & aperture_valid
            & aperture_speed_valid
        )
        features = np.concatenate(
            [
                relative_position,
                rotation_6d,
                velocity,
                angular_velocity,
                acceleration,
                jerk,
                aperture[:, None],
                aperture_speed,
            ],
            axis=-1,
        )
        if features.shape[1] != 23:
            raise AssertionError(f"Per-hand kinematics must have 23 dims, got {features.shape}")
        features[~valid] = 0.0
        per_hand.append(features.astype(np.float32))
        hand_valids.append(full_valid)
        relative_positions.append(relative_position)
        linear_velocities.append(velocity)

    bilateral_vector = relative_positions[1] - relative_positions[0]
    bilateral_distance = np.linalg.norm(bilateral_vector, axis=-1)
    bilateral_relative_speed = np.linalg.norm(
        linear_velocities[1] - linear_velocities[0], axis=-1
    )
    kinematics = np.concatenate(
        [per_hand[0], per_hand[1], bilateral_distance[:, None], bilateral_relative_speed[:, None]],
        axis=-1,
    ).astype(np.float32)
    if kinematics.shape[1] != 48:
        raise AssertionError(f"Kinematics contract must be 48-D, got {kinematics.shape}")
    kinematics_valid = hand_valids[0] & hand_valids[1]

    dt = np.zeros(len(time_ns), dtype=np.float32)
    if len(time_ns) > 1:
        dt[1:] = np.diff(time_ns.astype(np.float64)) / 1.0e9
    expected_dt = 1.0 / 60.0
    dt_valid = np.abs(dt - expected_dt) <= 0.25 * expected_dt
    dt_valid[0] = False
    pose_step = np.zeros((len(time_ns), 3), dtype=np.float32)
    pose_step[1:] = np.linalg.norm(np.diff(pose_position, axis=0), axis=-1)
    jump_score = np.max(pose_step, axis=1) / max(pose_jump_m, 1.0e-6)
    freeze_score = np.mean(pose_step <= 1.0e-7, axis=1).astype(np.float32)
    pose_quality = np.stack(
        [
            pose_valid[:, 0],
            pose_valid[:, 1],
            pose_valid[:, 2],
            dt_valid,
            np.clip(jump_score, 0.0, 1.0),
            freeze_score,
        ],
        axis=-1,
    ).astype(np.float32)
    return kinematics, kinematics_valid, pose_quality


def extract_sensor_arrays(
    mcap_path: str | Path,
    *,
    num_ticks: int,
    tactile_dim: int = 460,
    joints: int = 25,
    joint_dim: int = 7,
    output_rate_hz: float = 60.0,
    tactile_jump_threshold: float = 1000.0,
    tactile_saturation_threshold: float | None = None,
    pose_jump_m: float = 0.25,
) -> RawEpisodeSensors:
    if joint_dim != 7:
        raise ValueError("The formal exporter currently writes xyz+xyzw skeleton_joint_dim=7")
    source = Path(mcap_path)
    episode_id = source.name.removesuffix(".annotated.mcap")
    time_ns = np.rint(
        np.arange(num_ticks, dtype=np.float64) * (1.0e9 / output_rate_hz)
    ).astype(np.int64)
    tactile = np.zeros((num_ticks, 2, tactile_dim), dtype=np.float32)
    tactile_valid = np.zeros((num_ticks, 2), dtype=np.bool_)
    skeleton_world = np.zeros((num_ticks, 2, joints, joint_dim), dtype=np.float32)
    skeleton_valid = np.zeros((num_ticks, 2), dtype=np.bool_)
    joint_valid = np.zeros((num_ticks, 2, joints), dtype=np.bool_)
    pose_position = np.zeros((num_ticks, 3, 3), dtype=np.float32)
    pose_quaternion = np.zeros((num_ticks, 3, 4), dtype=np.float32)
    pose_quaternion[..., 3] = 1.0
    pose_valid = np.zeros((num_ticks, 3), dtype=np.bool_)
    pose_index = {topic: index for index, topic in enumerate(POSE_TOPICS)}

    for topic, log_time_ns, message in iter_episode_json_streams(source, JSON_SENSOR_TOPICS):
        frame = _frame_index(message, log_time_ns, output_rate_hz)
        if not 0 <= frame < num_ticks:
            raise ValueError(f"{episode_id}: {topic} frame {frame} is outside T={num_ticks}")
        session_time = message.get("session_time_ns")
        if isinstance(session_time, int):
            time_ns[frame] = session_time
        if topic.startswith("/tactile/raw/"):
            hand = HAND_BY_TOPIC[topic]
            values = np.asarray(message.get("values"), dtype=np.float32)
            if values.shape != (tactile_dim,):
                raise ValueError(
                    f"{episode_id}: {topic} frame {frame} has tactile shape {values.shape}"
                )
            tactile[frame, hand] = values
            tactile_valid[frame, hand] = bool(message.get("valid", False))
        elif topic.startswith("/hand_skeleton/"):
            hand = HAND_BY_TOPIC[topic]
            rows = message.get("joints")
            if not isinstance(rows, list) or len(rows) != joints:
                raise ValueError(
                    f"{episode_id}: {topic} frame {frame} has {len(rows) if isinstance(rows, list) else 'invalid'} joints"
                )
            skeleton_valid[frame, hand] = bool(message.get("valid", False))
            for joint_index, row in enumerate(rows):
                position = np.asarray(row.get("position_m"), dtype=np.float32)
                quaternion = np.asarray(row.get("orientation_xyzw"), dtype=np.float32)
                if position.shape != (3,) or quaternion.shape != (4,):
                    raise ValueError(
                        f"{episode_id}: invalid skeleton joint at frame {frame}, joint {joint_index}"
                    )
                skeleton_world[frame, hand, joint_index] = np.concatenate(
                    [position, quaternion]
                )
                joint_valid[frame, hand, joint_index] = bool(row.get("valid", False))
        elif topic in pose_index:
            stream = pose_index[topic]
            position = np.asarray(message.get("position_m"), dtype=np.float32)
            quaternion = np.asarray(message.get("orientation_xyzw"), dtype=np.float32)
            if position.shape != (3,) or quaternion.shape != (4,):
                raise ValueError(f"{episode_id}: invalid pose on {topic} frame {frame}")
            pose_position[frame, stream] = position
            pose_quaternion[frame, stream] = quaternion
            pose_valid[frame, stream] = bool(message.get("valid", False))

    if not np.all(np.diff(time_ns) > 0):
        raise ValueError(f"{episode_id}: unified time_ns is not strictly increasing")
    tactile_quality = _build_tactile_quality(
        tactile,
        tactile_valid,
        jump_threshold=tactile_jump_threshold,
        saturation_threshold=tactile_saturation_threshold,
    )
    skeleton_relative = _relative_skeleton(skeleton_world, joint_valid)
    kinematics, kinematics_valid, pose_quality = _build_kinematics(
        pose_position,
        pose_quaternion,
        pose_valid,
        skeleton_relative,
        joint_valid,
        time_ns,
        pose_jump_m=pose_jump_m,
    )
    return RawEpisodeSensors(
        episode_id=episode_id,
        num_ticks=num_ticks,
        time_ns=time_ns,
        tactile=tactile,
        tactile_valid=tactile_valid,
        tactile_quality=tactile_quality,
        skeleton=skeleton_relative,
        skeleton_valid=skeleton_valid,
        skeleton_joint_valid=joint_valid,
        kinematics=kinematics,
        kinematics_valid=kinematics_valid,
        pose_quality=pose_quality,
    )


def _array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def write_sensor_cache(
    output_path: str | Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    compression: str | None = "lzf",
    overwrite: bool = False,
) -> dict[str, Any]:
    h5py = _require_h5py()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite sensor cache without --overwrite: {output}"
        )
    partial = output.with_name(f".{output.name}.partial")
    partial.unlink(missing_ok=True)
    with h5py.File(partial, "w") as handle:
        handle.attrs["schema_version"] = SENSOR_CACHE_SCHEMA_VERSION
        handle.attrs["metadata_json"] = json.dumps(
            dict(metadata), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        for name, value in arrays.items():
            data = np.asarray(value)
            chunks = None
            if data.ndim > 0 and data.shape[0] > 0:
                chunk_first = min(data.shape[0], 600)
                chunks = (chunk_first,) + data.shape[1:]
            handle.create_dataset(
                name,
                data=data,
                chunks=chunks,
                compression=compression if data.size > 0 else None,
                shuffle=bool(compression and data.dtype.kind in "fiu"),
            )
        handle.flush()
    os.replace(partial, output)
    payload_sha256 = sha256_file(output)
    content_sha256 = _array_content_sha256(arrays)
    release_inputs = {}
    if metadata.get("corpus_release_sha256") not in (None, ""):
        release_inputs["corpus_release_sha256"] = str(
            metadata["corpus_release_sha256"]
        )
    record = artifact_record(
        artifact_type="legacy20h_sensor_cache",
        schema_version=SENSOR_CACHE_SCHEMA_VERSION,
        payload_sha256=payload_sha256,
        config_sha256=str(metadata["export_config_sha256"]),
        inputs={
            "mcap_sha256": str(metadata["mcap_sha256"]),
            "ontology_sha256": str(metadata["ontology_sha256"]),
            **release_inputs,
        },
        extra={
            "episode_id": metadata["episode_id"],
            "array_content_sha256": content_sha256,
            "path": output.name,
            "status": "implemented_not_experimental_result",
        },
    )
    atomic_write_json(output.with_suffix(output.suffix + ".meta.json"), record)
    return record


def export_episode(
    mcap_path: str | Path,
    output_dir: str | Path,
    *,
    ontology: FrozenOntology,
    config: Mapping[str, Any],
    mcap_sha256: str | None = None,
    corpus_release: str | None = None,
    corpus_release_sha256: str | None = None,
    allow_quarantined: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(mcap_path).expanduser().resolve()
    episode_id = source.name.removesuffix(".annotated.mcap")
    quarantine = set(config.get("quarantine_episode_ids", DEFAULT_QUARANTINE_EPISODES))
    if episode_id in quarantine and not allow_quarantined:
        raise ValueError(
            f"{episode_id} is quarantined for dropped atomic segments; "
            "publish a new reviewed manifest before export"
        )
    first = read_first_json_by_topic(source, ["/meta/session", "/meta/block", "/meta/data_contract"])
    if "/meta/session" not in first:
        raise ValueError(f"{episode_id}: missing /meta/session")
    session = first["/meta/session"]
    num_ticks = int(session["num_frames_60"])
    output_rate_hz = float(config.get("output_rate_hz", 60.0))
    if output_rate_hz != 60.0:
        raise ValueError("legacy20h formal output clock is frozen at 60 Hz")

    attachment, attachment_segments, topic_segments = read_dual_source_labels(source)
    assert_dual_source_labels_equal(attachment_segments, topic_segments, source=source)
    if not attachment_segments:
        raise ValueError(f"{episode_id}: semantic_review contains no atomic segments")
    label_arrays = build_label_arrays(
        attachment_segments,
        num_ticks=num_ticks,
        ontology=ontology,
        boundary_sigma_ticks=float(config.get("boundary_sigma_ticks", 5.0)),
        require_dense=bool(config.get("require_dense_labels", True)),
    )
    sensors = extract_sensor_arrays(
        source,
        num_ticks=num_ticks,
        tactile_dim=int(config.get("tactile_dim", 460)),
        joints=int(config.get("skeleton_joints", 25)),
        joint_dim=int(config.get("skeleton_joint_dim", 7)),
        output_rate_hz=output_rate_hz,
        tactile_jump_threshold=float(config.get("tactile_jump_threshold", 1000.0)),
        tactile_saturation_threshold=config.get("tactile_saturation_threshold"),
        pose_jump_m=float(config.get("pose_jump_m", 0.25)),
    )
    arrays: dict[str, np.ndarray] = {
        "tactile": sensors.tactile,
        "tactile_valid": sensors.tactile_valid,
        "tactile_quality": sensors.tactile_quality,
        "skeleton": sensors.skeleton,
        "skeleton_valid": sensors.skeleton_valid,
        "skeleton_joint_valid": sensors.skeleton_joint_valid,
        "kinematics": sensors.kinematics,
        "kinematics_valid": sensors.kinematics_valid,
        "pose_quality": sensors.pose_quality,
        "time_ns": sensors.time_ns,
        "global_frame_60": np.arange(num_ticks, dtype=np.int64),
        **label_arrays,
    }
    export_config_sha256 = sha256_json(config)
    source_sha256 = mcap_sha256 or sha256_file(source)
    # The exact release identity must come from a verified audit manifest, not
    # merely from the reusable producer YAML. Direct function calls without it
    # remain diagnostic/backward-compatible and do not claim a frozen release.
    effective_corpus_release = corpus_release
    if (effective_corpus_release is None) != (corpus_release_sha256 is None):
        raise ValueError(
            "corpus_release and corpus_release_sha256 must be supplied together"
        )
    if corpus_release_sha256 is not None:
        corpus_release_sha256 = require_sha256(
            corpus_release_sha256, "corpus_release_sha256"
        )
    metadata = {
        "episode_id": episode_id,
        "source_path": str(source),
        "mcap_sha256": source_sha256,
        "exporter_schema_version": SENSOR_CACHE_SCHEMA_VERSION,
        "export_config_sha256": export_config_sha256,
        "annotation_source": "semantic_review.json+/semantic/atomic_state",
        "annotation_version": "legacy20h_v1",
        "annotation_sha256": sha256_json(attachment_segments),
        "dual_source_labels_equal": True,
        "ontology_name": ontology.version,
        "ontology_sha256": ontology.sha256,
        "action_names": list(ontology.actions),
        "output_rate_hz": output_rate_hz,
        "num_ticks": num_ticks,
        "num_instances": len(attachment_segments),
        "num_class_change_boundaries": int(np.sum(label_arrays["boundary_type"] == 1)),
        "num_same_label_boundaries": int(np.sum(label_arrays["boundary_type"] == 2)),
        "tail_extension_ticks": int(num_ticks - int(attachment_segments[-1]["end_frame_60_exclusive"])),
        "coordinate_profile": "wrist-relative-skeleton+head-relative-wrist-v1",
        "kinematics_profile": "offline-backward-derived-v1",
        "kinematic_feature_names": [
            *(f"{hand}_{name}" for hand in ("left", "right") for name in (
                "rel_pos_x", "rel_pos_y", "rel_pos_z",
                "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
                "vel_x", "vel_y", "vel_z", "angvel_x", "angvel_y", "angvel_z",
                "acc_x", "acc_y", "acc_z", "jerk_x", "jerk_y", "jerk_z",
                "aperture", "aperture_speed",
            )),
            "bimanual_wrist_distance",
            "bimanual_relative_speed",
        ],
        "tactile_quality_names": [
            "packet_valid", "zero_fraction", "unchanged_fraction", "jump_fraction", "saturation_fraction"
        ],
        "pose_quality_names": [
            "head_valid", "left_wrist_valid", "right_wrist_valid", "dt_valid", "jump_score", "freeze_score"
        ],
        "subtask_target_status": "not_exported_until_global_subtask_registry_is_frozen",
        "experimental_result": False,
    }
    if effective_corpus_release is not None:
        metadata["corpus_release"] = str(effective_corpus_release)
        metadata["corpus_release_sha256"] = corpus_release_sha256
    if metadata["tail_extension_ticks"] < 0:
        raise ValueError(f"{episode_id}: labels extend past num_frames_60")
    if metadata["tail_extension_ticks"] > int(config.get("max_tail_extension_ticks", 3)):
        raise ValueError(
            f"{episode_id}: unexpected tail extension {metadata['tail_extension_ticks']} ticks"
        )
    output = Path(output_dir) / f"{episode_id}.sensor60.h5"
    return write_sensor_cache(
        output,
        arrays,
        metadata,
        compression=config.get("hdf5_compression", "lzf"),
        overwrite=overwrite,
    )


def _verified_existing_cache(
    path: Path,
    *,
    expected_config_sha256: str,
    expected_mcap_sha256: str,
    expected_ontology_sha256: str,
    expected_corpus_release: str | None = None,
    expected_corpus_release_sha256: str | None = None,
) -> dict[str, Any]:
    if (expected_corpus_release is None) != (
        expected_corpus_release_sha256 is None
    ):
        raise ValueError(
            "expected_corpus_release and expected_corpus_release_sha256 "
            "must be supplied together"
        )
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Existing cache lacks sidecar: {sidecar}")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    verify_artifact_record(record)
    if record.get("payload_sha256") != sha256_file(path):
        raise ValueError(f"Existing cache payload hash mismatch: {path}")
    expected_config = require_sha256(
        expected_config_sha256, "expected_config_sha256"
    )
    expected_mcap = require_sha256(expected_mcap_sha256, "expected_mcap_sha256")
    expected_ontology = require_sha256(
        expected_ontology_sha256, "expected_ontology_sha256"
    )
    if record.get("config_sha256") != expected_config:
        raise ValueError(f"Existing sensor cache is stale for current config: {path}")
    inputs = record.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"Existing sensor cache sidecar lacks inputs: {path}")
    if inputs.get("mcap_sha256") != expected_mcap:
        raise ValueError(f"Existing sensor cache is stale for current MCAP: {path}")
    if inputs.get("ontology_sha256") != expected_ontology:
        raise ValueError(f"Existing sensor cache is stale for current ontology: {path}")
    if expected_corpus_release_sha256 is not None:
        expected_release_hash = require_sha256(
            expected_corpus_release_sha256, "expected_corpus_release_sha256"
        )
        if inputs.get("corpus_release_sha256") != expected_release_hash:
            raise ValueError(
                f"Existing sensor cache is stale for current corpus release: {path}"
            )
    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        raw = handle.attrs.get("metadata_json")
    if raw is None:
        raise ValueError(f"Existing sensor cache lacks metadata_json: {path}")
    metadata = json.loads(raw)
    expected_metadata = {
        "export_config_sha256": expected_config,
        "mcap_sha256": expected_mcap,
        "ontology_sha256": expected_ontology,
    }
    if expected_corpus_release is not None:
        expected_metadata["corpus_release"] = expected_corpus_release
        expected_metadata["corpus_release_sha256"] = expected_release_hash
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"Existing sensor cache metadata {field} is stale: {path}"
            )
    return record


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return value


def _paths_from_arguments(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value).expanduser().resolve() for value in args.mcap]
    if args.episodes_parquet:
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("--episodes-parquet requires pandas and pyarrow") from error
        frame = pd.read_parquet(args.episodes_parquet)
        source_column = "source_path" if "source_path" in frame else "mcap_path"
        if source_column not in frame:
            raise ValueError("episodes parquet must contain source_path or mcap_path")
        if "qa_status" in frame:
            # audit20h writes "pass"; older reviewed registries used "accepted".
            frame = frame[frame["qa_status"].isin(["pass", "accepted"])]
        paths.extend(Path(value).expanduser().resolve() for value in frame[source_column])
    unique = sorted(set(paths), key=lambda path: str(path))
    return unique[: args.limit] if args.limit is not None else unique


def _registry_hashes(args: argparse.Namespace, paths: Sequence[Path]) -> dict[Path, str]:
    if args.hash_mode != "registry":
        return {}
    if not args.episodes_parquet:
        raise ValueError("--hash-mode registry requires --episodes-parquet")
    audit_manifest_value = getattr(args, "audit_manifest", None)
    if not audit_manifest_value:
        raise ValueError("--hash-mode registry requires --audit-manifest")
    audit_manifest_path = Path(audit_manifest_value).expanduser().resolve()
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    manifest_record = audit_manifest.get("manifest_metadata")
    if not isinstance(manifest_record, dict):
        raise ValueError("Audit manifest lacks manifest_metadata")
    verify_artifact_record(manifest_record)
    unhashed_manifest = dict(audit_manifest)
    unhashed_manifest.pop("manifest_metadata", None)
    if manifest_record.get("payload_sha256") != sha256_json(unhashed_manifest):
        raise ValueError("Audit manifest content hash mismatch")
    if audit_manifest.get("selection", {}).get("release_eligible") is not True:
        raise ValueError("Registry hash mode requires a release-eligible full audit")
    if audit_manifest.get("selection", {}).get("checksum_mode") != "sha256":
        raise ValueError("Registry hash mode requires audit checksum_mode=sha256")
    episodes_record = audit_manifest.get("artifacts", {}).get("episodes")
    if not isinstance(episodes_record, dict):
        raise ValueError("Audit manifest lacks the episodes artifact record")
    verify_artifact_record(episodes_record)
    episodes_path = Path(args.episodes_parquet).expanduser().resolve()
    if episodes_record.get("payload_sha256") != sha256_file(episodes_path):
        raise ValueError("episodes.parquet does not match the frozen audit manifest")
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Registry hash mode requires pandas and pyarrow") from error
    frame = pd.read_parquet(args.episodes_parquet)
    source_column = "source_path" if "source_path" in frame else "mcap_path"
    if source_column not in frame or "source_sha256" not in frame:
        raise ValueError(
            "Registry hash mode requires source_path/mcap_path and source_sha256 columns"
        )
    mapping: dict[Path, str] = {}
    for _, row in frame.iterrows():
        value = row.get("source_sha256")
        if not isinstance(value, str) or len(value) != 64:
            continue
        try:
            int(value, 16)
        except ValueError:
            continue
        mapping[Path(str(row[source_column])).expanduser().resolve()] = value.lower()
    missing = [str(path) for path in paths if path not in mapping]
    if missing:
        raise ValueError(
            "Registry hash mode has no formal source_sha256 for: "
            f"{missing[:10]}"
        )
    return {path: mapping[path] for path in paths}


def _audit_release_identity(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    """Resolve the corpus release only from a release-eligible audit artifact.

    The producer YAML intentionally stays reusable across a frozen corpus release;
    the exact release hash comes from the same audit manifest that authenticates
    the per-MCAP SHA256 registry.
    """

    configured_release = config.get("corpus_release")
    if configured_release in (None, ""):
        return None, None
    if args.hash_mode != "registry" or not getattr(args, "audit_manifest", None):
        raise ValueError(
            "A versioned corpus_release requires --hash-mode registry and "
            "--audit-manifest"
        )
    audit_path = Path(args.audit_manifest).expanduser().resolve()
    audit_manifest = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit_manifest.get("selection", {}).get("release_eligible") is not True:
        raise ValueError("Corpus release export requires a release-eligible audit")
    observed_release = audit_manifest.get("corpus_release")
    if observed_release != configured_release:
        raise ValueError(
            "Export config/audit corpus release mismatch: "
            f"{configured_release!r} != {observed_release!r}"
        )
    release_sha256 = require_sha256(
        audit_manifest.get("corpus_release_sha256"),
        "audit_manifest.corpus_release_sha256",
    )
    return str(observed_release), release_sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export read-only legacy20h MCAP sensor/label caches"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mcap", action="append", default=[])
    parser.add_argument("--episodes-parquet", type=Path)
    parser.add_argument(
        "--audit-manifest",
        type=Path,
        help="Required trust anchor for --hash-mode registry",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-quarantined", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--hash-mode", choices=("sha256", "registry"), default="sha256",
        help="Formal mode hashes the source; registry uses episodes.parquet mcap_sha256",
    )
    args = parser.parse_args()
    if args.overwrite and args.skip_existing:
        parser.error("--overwrite and --skip-existing are mutually exclusive")
    config = _load_yaml(args.config)
    ontology = load_frozen_ontology(
        args.ontology, expected_sha256=config.get("ontology_sha256")
    )
    paths = _paths_from_arguments(args)
    if not paths:
        parser.error("Provide --mcap or --episodes-parquet")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_hashes = _registry_hashes(args, paths)
    corpus_release, corpus_release_sha256 = _audit_release_identity(args, config)
    export_config_sha256 = sha256_json(config)
    records = []
    for path in paths:
        episode_id = path.name.removesuffix(".annotated.mcap")
        cache_path = output_dir / f"{episode_id}.sensor60.h5"
        source_sha256 = registry_hashes.get(path) or sha256_file(path)
        if cache_path.exists() and args.skip_existing:
            record = _verified_existing_cache(
                cache_path,
                expected_config_sha256=export_config_sha256,
                expected_mcap_sha256=source_sha256,
                expected_ontology_sha256=ontology.sha256,
                expected_corpus_release=corpus_release,
                expected_corpus_release_sha256=corpus_release_sha256,
            )
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            continue
        record = export_episode(
            path,
            output_dir,
            ontology=ontology,
            config=config,
            mcap_sha256=source_sha256,
            corpus_release=corpus_release,
            corpus_release_sha256=corpus_release_sha256,
            allow_quarantined=args.allow_quarantined,
            overwrite=args.overwrite,
        )
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    index = {
        "schema_version": "legacy20h_sensor_cache_index_v1",
        "config_sha256": export_config_sha256,
        "ontology_sha256": ontology.sha256,
        "corpus_release": corpus_release,
        "corpus_release_sha256": corpus_release_sha256,
        "episodes": records,
        "experimental_result": False,
    }
    index["index_sha256"] = sha256_json(index)
    atomic_write_json(output_dir / "sensor_cache_index.json", index)


if __name__ == "__main__":
    main()
