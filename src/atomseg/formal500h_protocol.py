from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import h5py
import numpy as np

from .formal500h_training_release import RELEASE_SCHEMA


PROTOCOL_SCHEMA = "formal500h_experiment_protocol_v2"
SPLIT_SCHEMA = "formal500h_group_split_v1"
INVENTORY_SCHEMA = "formal500h_episode_inventory_v1"
EPISODE_SCHEMA = "formal500h_training_episode_v3"
GOOD_HDF5_STATUSES = {"written", "skipped_verified"}
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)
NUM_CLASSES = 26
WAVE0_DIAGNOSTIC_METRIC_POLICY = {
    "reportability": "diagnostic_only_not_a_paper_result",
    "output_rate_hz": 30,
    "segment_iou_thresholds": [0.10, 0.25, 0.50],
    "boundary_probability_threshold": 0.5,
    "boundary_tolerance_frames": 8,
    "boundary_tolerance_ms": 8000.0 / 30.0,
    "boundary_min_distance_frames": 2,
    "composite": "0.45*f1@50 + 0.30*edit + 0.25*boundary_f1",
}
FORMAL_EVALUATION_POLICY = {
    "evaluation_unit": "full_episode_after_overlap_stitching",
    "output_rate_hz": 30,
    "stitch_weighting": "uniform",
    "require_full_coverage": True,
    "segment_iou_thresholds": [0.10, 0.25, 0.50],
    "reported_views": ["raw_instance", "collapsed_semantic"],
    "primary_view": "collapsed_semantic",
    "boundary_types": ["all", "class_change", "same_label_repeat"],
    "primary_boundary_type": "class_change",
    "boundary_tolerances_ms": [50, 100, 250, 500],
    "primary_boundary_tolerance_ms": 250,
    "primary_composite": (
        "0.50*collapsed_semantic_f1@50 + "
        "0.50*class_change_boundary_f1@250ms"
    ),
    "checkpoint_selection": "maximum validation primary_composite only",
    "decode_calibration": {
        "split": "validation_only",
        "boundary_threshold_candidates": [0.50, 0.60, 0.70, 0.80, 0.85, 0.90],
        "boundary_min_distance_frames_30hz": [3, 6, 9, 15],
        "min_segment_frames_30hz": [6, 9, 12, 18],
        "tie_break": (
            "max(primary_composite,collapsed_semantic_f1@50,"
            "class_change_boundary_f1@250ms,raw_instance_f1@50,"
            "-min_segment,-nms,-abs(threshold-0.5),-threshold)"
        ),
    },
    "implementation_gate": (
        "evaluator config and source hashes must be frozen and embedded in every "
        "formal run before formal main-model training; test remains sealed"
    ),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


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
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, path)


def _capture_key(row: Mapping[str, Any]) -> str:
    value = row.get("capture_key")
    if value:
        return str(value)
    mcap_path = row.get("mcap_path")
    if mcap_path:
        return Path(str(mcap_path)).stem
    raise ValueError("training row lacks capture_key and mcap_path")


def _validate_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("training manifest is empty")
    seen_capture: set[str] = set()
    seen_mcap: set[str] = set()
    seen_cache: set[str] = set()
    invalid: list[str] = []
    for index, row in enumerate(rows):
        try:
            capture = _capture_key(row)
            mcap = str(row["mcap_path"])
            cache = str(row["cache_path"])
            group = str(row["split_group_id"])
            duration = float(row["duration_s"])
            ticks = int(row["annotation_frame_count"])
            swap = row.get("tactile_swap_left_right")
            expected_correction = "swap_left_right" if swap is True else "keep"
            checks = (
                row.get("training_release_schema") == RELEASE_SCHEMA,
                row.get("tactile_qc_keep") is True,
                row.get("hdf5_status") in GOOD_HDF5_STATUSES,
                bool(mcap),
                bool(cache),
                bool(group),
                math.isfinite(duration) and duration > 0,
                ticks > 0,
                type(swap) is bool,
                row.get("tactile_hand_correction") == expected_correction,
            )
            if not all(checks):
                raise ValueError("row violates the frozen training-release contract")
            if capture in seen_capture:
                raise ValueError(f"duplicate capture_key {capture}")
            if mcap in seen_mcap:
                raise ValueError(f"duplicate mcap_path {mcap}")
            if cache in seen_cache:
                raise ValueError(f"duplicate cache_path {cache}")
            seen_capture.add(capture)
            seen_mcap.add(mcap)
            seen_cache.add(cache)
        except (KeyError, TypeError, ValueError) as error:
            invalid.append(f"row {index}: {error}")
    if invalid:
        raise ValueError(
            f"training manifest has {len(invalid)} invalid rows: {invalid[:5]}"
        )


def _inspect_episode_job(payload: tuple[int, Mapping[str, Any]]) -> tuple[int, dict[str, Any]]:
    index, row = payload
    path = Path(str(row["cache_path"]))
    if not path.is_file():
        raise ValueError(f"HDF5 does not exist: {path}")
    with h5py.File(path, "r") as handle:
        schema = _decode_attr(handle.attrs.get("schema_version", ""))
        if schema != EPISODE_SCHEMA:
            raise ValueError(f"unexpected HDF5 schema {schema!r}: {path}")
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"incomplete HDF5: {path}")
        required = (
            "targets/action_class",
            "annotations/atomic_segments/start_frame_60",
            "annotations/atomic_segments/end_frame_60_exclusive",
            "annotations/atomic_segments/class_index",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"HDF5 lacks {missing}: {path}")
        ticks = int(handle["targets/action_class"].shape[0])
        starts = np.asarray(
            handle["annotations/atomic_segments/start_frame_60"], dtype=np.int64
        )
        ends = np.asarray(
            handle["annotations/atomic_segments/end_frame_60_exclusive"],
            dtype=np.int64,
        )
        labels = np.asarray(
            handle["annotations/atomic_segments/class_index"], dtype=np.int64
        )
        if not (starts.shape == ends.shape == labels.shape):
            raise ValueError(f"atomic-segment arrays have different shapes: {path}")
        if labels.size == 0:
            raise ValueError(f"episode has no atomic segments: {path}")
        if labels.min() < 0 or labels.max() >= NUM_CLASSES:
            raise ValueError(f"atomic class is outside [0,{NUM_CLASSES}): {path}")
        if np.any(starts < 0) or np.any(ends <= starts) or np.any(ends > ticks):
            raise ValueError(f"invalid atomic segment interval: {path}")
        if len(starts) > 1 and np.any(starts[1:] < ends[:-1]):
            raise ValueError(f"overlapping atomic segments: {path}")
        declared_ticks = int(row["annotation_frame_count"])
        if ticks != declared_ticks:
            raise ValueError(
                f"target ticks {ticks} != manifest annotation_frame_count "
                f"{declared_ticks}: {path}"
            )
        declared_seconds = float(row["duration_s"])
        if abs(ticks / 60.0 - declared_seconds) > 2.0 / 60.0:
            raise ValueError(
                f"duration {declared_seconds} differs from {ticks}/60 Hz: {path}"
            )
        metadata_raw = handle.attrs.get("metadata_json")
        try:
            metadata = json.loads(_decode_attr(metadata_raw))
        except (TypeError, json.JSONDecodeError):
            raise ValueError(f"invalid metadata_json: {path}") from None
        expected_metadata = {
            "source_mcap_path": str(row["mcap_path"]),
            "tactile_hand_correction": str(row["tactile_hand_correction"]),
            "tactile_swap_left_right": bool(row["tactile_swap_left_right"]),
            "tactile_hand_order": ["left", "right"],
        }
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(f"HDF5 metadata mismatch for {name}: {path}")

    segment_count = np.bincount(labels, minlength=NUM_CLASSES).astype(np.int64)
    frame_count = np.bincount(
        labels, weights=(ends - starts), minlength=NUM_CLASSES
    ).astype(np.int64)
    inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "capture_key": _capture_key(row),
        "episode_id": str(row.get("episode_id") or path.stem),
        "task_id": str(row.get("task_id") or metadata.get("task_id") or ""),
        "split_group_id": str(row["split_group_id"]),
        "split_group_basis": str(row.get("split_group_basis") or "unknown"),
        "mcap_path": str(row["mcap_path"]),
        "cache_path": str(path),
        "ticks_60": ticks,
        "duration_s": ticks / 60.0,
        "class_indices": np.flatnonzero(segment_count).astype(int).tolist(),
        "class_segment_count": segment_count.astype(int).tolist(),
        "class_frame_count_60": frame_count.astype(int).tolist(),
        "atomic_segment_count": int(labels.size),
        "hdf5_schema": EPISODE_SCHEMA,
        "hdf5_complete": True,
        "tactile_hand_order": ["left", "right"],
        "tactile_hand_correction": str(row["tactile_hand_correction"]),
    }
    return index, inventory


def inspect_training_manifest(
    rows: Sequence[Mapping[str, Any]], *, workers: int
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    output: list[dict[str, Any] | None] = [None] * len(rows)
    # Do not pickle the nested tactile-QC payload into every process job.  The
    # inventory worker only needs this compact, immutable lineage subset.
    jobs = [
        (
            index,
            {
                name: row.get(name)
                for name in (
                    "capture_key",
                    "episode_id",
                    "task_id",
                    "mcap_path",
                    "cache_path",
                    "duration_s",
                    "annotation_frame_count",
                    "split_group_id",
                    "split_group_basis",
                    "tactile_swap_left_right",
                    "tactile_hand_correction",
                )
            },
        )
        for index, row in enumerate(rows)
    ]
    if workers == 1:
        iterator = map(_inspect_episode_job, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_inspect_episode_job, jobs, chunksize=16)
    try:
        for completed, (index, inventory) in enumerate(iterator, start=1):
            output[index] = inventory
            if completed % 1000 == 0 or completed == len(rows):
                print(
                    json.dumps(
                        {
                            "event": "formal_split_inventory_progress",
                            "episodes": completed,
                            "total_episodes": len(rows),
                        }
                    ),
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if any(value is None for value in output):
        raise AssertionError("episode inventory is incomplete")
    return [value for value in output if value is not None]


@dataclass(frozen=True)
class GroupStats:
    name: str
    duration_s: float
    episodes: int
    segments: np.ndarray
    frames: np.ndarray
    tasks: frozenset[str]
    task_duration_s: Mapping[str, float]


def _aggregate_groups(inventory: Sequence[Mapping[str, Any]]) -> list[GroupStats]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in inventory:
        grouped[str(row["split_group_id"])].append(row)
    groups: list[GroupStats] = []
    for name in sorted(grouped):
        rows = grouped[name]
        task_duration_s: dict[str, float] = defaultdict(float)
        for row in rows:
            task = str(row["task_id"])
            if task:
                task_duration_s[task] += float(row["duration_s"])
        groups.append(
            GroupStats(
                name=name,
                duration_s=sum(float(row["duration_s"]) for row in rows),
                episodes=len(rows),
                segments=np.sum(
                    np.asarray(
                        [row["class_segment_count"] for row in rows], dtype=np.int64
                    ),
                    axis=0,
                ),
                frames=np.sum(
                    np.asarray(
                        [row["class_frame_count_60"] for row in rows], dtype=np.int64
                    ),
                    axis=0,
                ),
                tasks=frozenset(str(row["task_id"]) for row in rows if row["task_id"]),
                task_duration_s=dict(task_duration_s),
            )
        )
    if len(groups) < 3:
        raise ValueError("formal split requires at least three source groups")
    return groups


def _assignment_score(
    assignment: np.ndarray,
    *,
    duration: np.ndarray,
    episodes: np.ndarray,
    frames: np.ndarray,
    presence: np.ndarray,
    task_duration: np.ndarray,
    task_presence: np.ndarray,
    fractions: np.ndarray,
    required_eval_classes: np.ndarray,
    all_classes: np.ndarray,
    required_eval_tasks: np.ndarray,
    all_tasks: np.ndarray,
    minimum_groups: int,
) -> float:
    group_count = np.bincount(assignment, minlength=3).astype(np.float64)
    duration_by_split = np.bincount(
        assignment, weights=duration, minlength=3
    ).astype(np.float64)
    episode_by_split = np.bincount(
        assignment, weights=episodes, minlength=3
    ).astype(np.float64)
    frame_by_split = np.stack(
        [frames[assignment == index].sum(axis=0) for index in range(3)], axis=0
    )
    group_presence = np.stack(
        [presence[assignment == index].sum(axis=0) for index in range(3)], axis=0
    )
    task_duration_by_split = np.stack(
        [task_duration[assignment == index].sum(axis=0) for index in range(3)], axis=0
    )
    task_group_presence = np.stack(
        [task_presence[assignment == index].sum(axis=0) for index in range(3)], axis=0
    )
    duration_fraction = duration_by_split / duration.sum()
    episode_fraction = episode_by_split / episodes.sum()
    group_fraction = group_count / len(assignment)
    score = 2000.0 * float(
        np.square((duration_fraction - fractions) / np.maximum(fractions, 0.05)).sum()
    )
    score += 100.0 * float(
        np.square((episode_fraction - fractions) / np.maximum(fractions, 0.05)).sum()
    )
    score += 10.0 * float(
        np.square((group_fraction - fractions) / np.maximum(fractions, 0.05)).sum()
    )
    total_frames = frames.sum(axis=0)
    present_frames = total_frames > 0
    if present_frames.any():
        expected = fractions[:, None] * total_frames[None, :]
        score += 5.0 * float(
            (
                np.abs(frame_by_split[:, present_frames] - expected[:, present_frames])
                / np.maximum(expected[:, present_frames], 1.0)
            ).mean()
        )
    total_task_duration = task_duration.sum(axis=0)
    present_tasks = total_task_duration > 0
    if present_tasks.any():
        expected_task_duration = fractions[:, None] * total_task_duration[None, :]
        score += 5.0 * float(
            (
                np.abs(
                    task_duration_by_split[:, present_tasks]
                    - expected_task_duration[:, present_tasks]
                )
                / np.maximum(expected_task_duration[:, present_tasks], 1.0)
            ).mean()
        )
    score += 1.0e9 * float(np.maximum(minimum_groups - group_count, 0).sum())
    score += 1.0e8 * float((group_presence[0, all_classes] == 0).sum())
    if required_eval_classes.size:
        score += 1.0e8 * float(
            (group_presence[1, required_eval_classes] == 0).sum()
            + (group_presence[2, required_eval_classes] == 0).sum()
        )
    score += 1.0e8 * float((task_group_presence[0, all_tasks] == 0).sum())
    if required_eval_tasks.size:
        score += 1.0e8 * float(
            (task_group_presence[1, required_eval_tasks] == 0).sum()
            + (task_group_presence[2, required_eval_tasks] == 0).sum()
        )
    return score


def _duration_repair_diagnostics(
    assignment: np.ndarray,
    *,
    duration: np.ndarray,
    fractions: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    duration_by_split = np.bincount(
        assignment, weights=duration, minlength=3
    ).astype(np.float64)
    achieved = duration_by_split / duration.sum()
    deviation = np.abs(achieved - fractions)
    excess = np.maximum(deviation - tolerance, 0.0)
    return {
        "achieved_duration_fractions": achieved.tolist(),
        "absolute_deviation": deviation.tolist(),
        "tolerance": float(tolerance),
        "maximum_excess": float(excess.max()),
        "passes": bool(np.all(excess <= 1.0e-12)),
    }


def _repair_duration_assignment(
    assignment: np.ndarray,
    *,
    duration: np.ndarray,
    fractions: np.ndarray,
    tolerance: float,
    score: Callable[[np.ndarray], float],
    hard_valid: Callable[[np.ndarray], bool],
    maximum_rounds: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministically satisfy the frozen duration audit when locally feasible.

    The simulated-annealing objective also balances episode counts, class
    frames, task durations and group counts.  Its lowest *aggregate* score can
    therefore narrowly miss the later maximum-deviation audit even when a
    nearby hard-valid assignment passes.  This repair enumerates all one-group
    moves and pairwise cross-split swaps, never accepting a candidate that
    breaks the class/task/group hard checks.  The audit tolerance itself is not
    changed.
    """

    if maximum_rounds < 1:
        raise ValueError("maximum_rounds must be positive")
    current = np.asarray(assignment, dtype=np.int8).copy()
    if current.ndim != 1 or current.shape != duration.shape:
        raise ValueError("assignment and duration must be aligned vectors")
    if not hard_valid(current):
        raise ValueError("duration repair requires a hard-valid starting assignment")

    before = _duration_repair_diagnostics(
        current, duration=duration, fractions=fractions, tolerance=tolerance
    )

    def quality(value: np.ndarray) -> tuple[int, float, float]:
        diagnostics = _duration_repair_diagnostics(
            value, duration=duration, fractions=fractions, tolerance=tolerance
        )
        return (
            0 if diagnostics["passes"] else 1,
            float(diagnostics["maximum_excess"]),
            float(score(value)),
        )

    current_quality = quality(current)
    rounds = 0
    moves = 0
    for _ in range(maximum_rounds):
        best = current
        best_quality = current_quality
        for index in range(len(current)):
            source = int(current[index])
            for destination in range(3):
                if destination == source:
                    continue
                proposal = current.copy()
                proposal[index] = destination
                if not hard_valid(proposal):
                    continue
                proposal_quality = quality(proposal)
                if proposal_quality < best_quality:
                    best = proposal
                    best_quality = proposal_quality
        for left in range(len(current)):
            for right in range(left + 1, len(current)):
                if current[left] == current[right]:
                    continue
                proposal = current.copy()
                proposal[left], proposal[right] = proposal[right], proposal[left]
                if not hard_valid(proposal):
                    continue
                proposal_quality = quality(proposal)
                if proposal_quality < best_quality:
                    best = proposal
                    best_quality = proposal_quality
        rounds += 1
        if np.array_equal(best, current):
            break
        current = best
        current_quality = best_quality
        moves += 1

    after = _duration_repair_diagnostics(
        current, duration=duration, fractions=fractions, tolerance=tolerance
    )
    return current, {
        "method": "hard_valid_exhaustive_move_swap_v1",
        "maximum_rounds": maximum_rounds,
        "rounds_executed": rounds,
        "accepted_moves": moves,
        "before": before,
        "after": after,
    }


def optimize_group_assignment(
    groups: Sequence[GroupStats],
    *,
    fractions: Sequence[float],
    seed: int,
    restarts: int,
    iterations: int,
    minimum_groups: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    fraction_array = np.asarray(fractions, dtype=np.float64)
    if fraction_array.shape != (3,) or np.any(fraction_array <= 0):
        raise ValueError("split fractions must contain three positive values")
    if not math.isclose(float(fraction_array.sum()), 1.0, abs_tol=1.0e-9):
        raise ValueError("split fractions must sum to one")
    if minimum_groups < 1 or len(groups) < 3 * minimum_groups:
        raise ValueError("not enough source groups for minimum_groups_per_split")
    if restarts < 1 or iterations < 1:
        raise ValueError("optimizer restarts and iterations must be positive")

    duration = np.asarray([group.duration_s for group in groups], dtype=np.float64)
    episodes = np.asarray([group.episodes for group in groups], dtype=np.float64)
    frames = np.stack([group.frames for group in groups], axis=0)
    segments = np.stack([group.segments for group in groups], axis=0)
    presence = segments > 0
    source_group_support = presence.sum(axis=0)
    all_classes = np.flatnonzero(source_group_support > 0)
    required_eval_classes = np.flatnonzero(source_group_support >= 3)
    task_names = sorted({task for group in groups for task in group.tasks})
    task_duration = np.asarray(
        [
            [group.task_duration_s.get(task, 0.0) for task in task_names]
            for group in groups
        ],
        dtype=np.float64,
    )
    task_presence = task_duration > 0
    source_task_group_support = task_presence.sum(axis=0)
    all_tasks = np.flatnonzero(source_task_group_support > 0)
    required_eval_tasks = np.flatnonzero(source_task_group_support >= 3)
    rng = random.Random(seed)
    best_assignment: np.ndarray | None = None
    best_score = float("inf")

    def score(value: np.ndarray) -> float:
        return _assignment_score(
            value,
            duration=duration,
            episodes=episodes,
            frames=frames,
            presence=presence,
            task_duration=task_duration,
            task_presence=task_presence,
            fractions=fraction_array,
            required_eval_classes=required_eval_classes,
            all_classes=all_classes,
            required_eval_tasks=required_eval_tasks,
            all_tasks=all_tasks,
            minimum_groups=minimum_groups,
        )

    def hard_valid(value: np.ndarray) -> bool:
        group_count = np.bincount(value, minlength=3)
        if np.any(group_count < minimum_groups):
            return False
        value_group_presence = np.stack(
            [presence[value == index].sum(axis=0) for index in range(3)], axis=0
        )
        value_task_presence = np.stack(
            [task_presence[value == index].sum(axis=0) for index in range(3)],
            axis=0,
        )
        return bool(
            np.all(value_group_presence[0, all_classes] > 0)
            and np.all(value_group_presence[1, required_eval_classes] > 0)
            and np.all(value_group_presence[2, required_eval_classes] > 0)
            and np.all(value_task_presence[0, all_tasks] > 0)
            and np.all(value_task_presence[1, required_eval_tasks] > 0)
            and np.all(value_task_presence[2, required_eval_tasks] > 0)
        )

    for restart in range(restarts):
        jitter = np.asarray([rng.random() for _ in groups])
        order = np.lexsort((jitter, -duration))
        current = np.full(len(groups), -1, dtype=np.int8)
        current_duration = np.zeros(3, dtype=np.float64)
        current_count = np.zeros(3, dtype=np.int64)
        for position, group_index in enumerate(order):
            groups_left = len(groups) - position
            short = [
                split
                for split in range(3)
                if current_count[split] < minimum_groups
                and groups_left
                <= sum(max(0, minimum_groups - int(value)) for value in current_count)
            ]
            candidates = short or [0, 1, 2]
            chosen = min(
                candidates,
                key=lambda split: (
                    (
                        (current_duration[split] + duration[group_index])
                        / (duration.sum() * fraction_array[split])
                    ),
                    current_count[split] / (len(groups) * fraction_array[split]),
                    rng.random(),
                ),
            )
            current[group_index] = chosen
            current_duration[chosen] += duration[group_index]
            current_count[chosen] += 1
        current_score = score(current)
        temperature_start = max(10.0, current_score * 0.01)
        for step in range(iterations):
            proposal = current.copy()
            if rng.random() < 0.65:
                index = rng.randrange(len(groups))
                proposal[index] = rng.randrange(3)
            else:
                left, right = rng.sample(range(len(groups)), 2)
                proposal[left], proposal[right] = proposal[right], proposal[left]
            proposal_score = score(proposal)
            progress = step / max(iterations - 1, 1)
            temperature = temperature_start * (1.0 - progress) ** 2 + 1.0e-6
            delta = proposal_score - current_score
            if delta <= 0 or rng.random() < math.exp(-min(delta / temperature, 700.0)):
                current = proposal
                current_score = proposal_score
            if current_score < best_score:
                best_assignment = current.copy()
                best_score = current_score
        print(
            json.dumps(
                {
                    "event": "formal_split_optimizer_progress",
                    "restart": restart + 1,
                    "restarts": restarts,
                    "best_score": best_score,
                }
            ),
            flush=True,
        )
    if best_assignment is None:
        raise AssertionError("split optimizer did not produce an assignment")
    duration_tolerance = max(
        0.03, float(duration.max() / duration.sum()) + 0.01
    )
    best_assignment, duration_repair = _repair_duration_assignment(
        best_assignment,
        duration=duration,
        fractions=fraction_array,
        tolerance=duration_tolerance,
        score=score,
        hard_valid=hard_valid,
    )
    best_score = score(best_assignment)
    print(
        json.dumps(
            {
                "event": "formal_split_duration_repair",
                **duration_repair,
                "best_score_after_repair": best_score,
            }
        ),
        flush=True,
    )
    group_presence = np.stack(
        [presence[best_assignment == index].sum(axis=0) for index in range(3)], axis=0
    )
    task_group_presence = np.stack(
        [
            task_presence[best_assignment == index].sum(axis=0)
            for index in range(3)
        ],
        axis=0,
    )
    hard_checks = {
        "minimum_groups_per_split": bool(
            np.all(np.bincount(best_assignment, minlength=3) >= minimum_groups)
        ),
        "all_source_classes_in_train": bool(
            np.all(group_presence[0, all_classes] > 0)
        ),
        "three_way_coverable_classes_in_val": bool(
            np.all(group_presence[1, required_eval_classes] > 0)
        ),
        "three_way_coverable_classes_in_test": bool(
            np.all(group_presence[2, required_eval_classes] > 0)
        ),
        "all_source_tasks_in_train": bool(
            np.all(task_group_presence[0, all_tasks] > 0)
        ),
        "three_way_coverable_tasks_in_val": bool(
            np.all(task_group_presence[1, required_eval_tasks] > 0)
        ),
        "three_way_coverable_tasks_in_test": bool(
            np.all(task_group_presence[2, required_eval_tasks] > 0)
        ),
    }
    if not all(hard_checks.values()):
        raise RuntimeError(
            f"could not construct a valid formal split: {hard_checks}; "
            "increase optimizer restarts/iterations or audit the group definition"
        )
    assignment = {
        group.name: SPLIT_NAMES[int(best_assignment[index])]
        for index, group in enumerate(groups)
    }
    diagnostics = {
        "optimizer": "deterministic_group_simulated_annealing_v1",
        "seed": seed,
        "restarts": restarts,
        "iterations_per_restart": iterations,
        "best_score": best_score,
        "duration_repair": duration_repair,
        "hard_checks": hard_checks,
        "source_group_support_by_class": source_group_support.astype(int).tolist(),
        "three_way_coverable_classes": required_eval_classes.astype(int).tolist(),
        "not_three_way_coverable_classes": np.flatnonzero(
            (source_group_support > 0) & (source_group_support < 3)
        ).astype(int).tolist(),
        "task_names": task_names,
        "source_group_support_by_task": source_task_group_support.astype(int).tolist(),
        "three_way_coverable_tasks": [task_names[index] for index in required_eval_tasks],
        "not_three_way_coverable_tasks": [
            task_names[index]
            for index in np.flatnonzero(
                (source_task_group_support > 0) & (source_task_group_support < 3)
            )
        ],
    }
    return assignment, diagnostics


def _split_summary(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups = sorted({str(row["split_group_id"]) for row in inventory})
    tasks = sorted({str(row["task_id"]) for row in inventory if row["task_id"]})
    task_episode_count = Counter(str(row["task_id"]) for row in inventory if row["task_id"])
    task_duration_s: dict[str, float] = defaultdict(float)
    task_groups: dict[str, set[str]] = defaultdict(set)
    for row in inventory:
        task = str(row["task_id"])
        if not task:
            continue
        task_duration_s[task] += float(row["duration_s"])
        task_groups[task].add(str(row["split_group_id"]))
    segments = np.sum(
        np.asarray([row["class_segment_count"] for row in inventory], dtype=np.int64),
        axis=0,
    )
    frames = np.sum(
        np.asarray([row["class_frame_count_60"] for row in inventory], dtype=np.int64),
        axis=0,
    )
    groups_per_class = []
    for class_index in range(NUM_CLASSES):
        groups_per_class.append(
            len(
                {
                    str(row["split_group_id"])
                    for row in inventory
                    if int(row["class_segment_count"][class_index]) > 0
                }
            )
        )
    return {
        "name": name,
        "episodes": len(rows),
        "groups": len(groups),
        "group_ids": groups,
        "tasks": len(tasks),
        "task_ids": tasks,
        "task_episode_count": dict(sorted(task_episode_count.items())),
        "task_duration_s": dict(sorted(task_duration_s.items())),
        "task_group_count": {
            task: len(task_groups[task]) for task in sorted(task_groups)
        },
        "duration_s": sum(float(row["duration_s"]) for row in inventory),
        "hours": sum(float(row["duration_s"]) for row in inventory) / 3600.0,
        "class_indices": np.flatnonzero(segments).astype(int).tolist(),
        "class_segment_count": segments.astype(int).tolist(),
        "class_frame_count_60": frames.astype(int).tolist(),
        "class_group_count": groups_per_class,
    }


def freeze_formal_protocol(
    *,
    training_manifest: Path,
    output_dir: Path,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    seed: int = 20260825,
    workers: int = 16,
    restarts: int = 32,
    iterations: int = 4000,
    minimum_groups_per_split: int = 3,
    overwrite: bool = False,
) -> dict[str, Any]:
    training_manifest = training_manifest.resolve()
    output_dir = output_dir.resolve()
    rows = _read_jsonl(training_manifest)
    _validate_manifest_rows(rows)
    reserved = (
        "protocol.json",
        "audit.json",
        "episode_inventory.jsonl",
        "group_assignments.jsonl",
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
    )
    existing = [output_dir / name for name in reserved if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"formal protocol already exists; refusing to overwrite: {existing[0]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = inspect_training_manifest(rows, workers=workers)
    groups = _aggregate_groups(inventory)
    assignment, optimizer = optimize_group_assignment(
        groups,
        fractions=fractions,
        seed=seed,
        restarts=restarts,
        iterations=iterations,
        minimum_groups=minimum_groups_per_split,
    )
    inventory_by_capture = {str(row["capture_key"]): row for row in inventory}
    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    split_inventory: dict[str, list[dict[str, Any]]] = {
        name: [] for name in SPLIT_NAMES
    }
    for row in rows:
        capture = _capture_key(row)
        episode = inventory_by_capture[capture]
        split_name = assignment[str(episode["split_group_id"])]
        frozen = {
            **row,
            "formal_split_schema": SPLIT_SCHEMA,
            "formal_split_name": split_name,
            "formal_split_seed": seed,
            "formal_split_group_id": str(episode["split_group_id"]),
            "formal_episode_summary": {
                "ticks_60": episode["ticks_60"],
                "duration_s": episode["duration_s"],
                "class_indices": episode["class_indices"],
                "class_segment_count": episode["class_segment_count"],
                "class_frame_count_60": episode["class_frame_count_60"],
            },
        }
        split_rows[split_name].append(frozen)
        split_inventory[split_name].append(episode)
    input_captures = {_capture_key(row) for row in rows}
    output_captures = {
        _capture_key(row) for name in SPLIT_NAMES for row in split_rows[name]
    }
    groups_by_split = {
        name: {str(row["formal_split_group_id"]) for row in split_rows[name]}
        for name in SPLIT_NAMES
    }
    overlaps = {
        "train_val": sorted(groups_by_split["train"] & groups_by_split["val"]),
        "train_test": sorted(groups_by_split["train"] & groups_by_split["test"]),
        "val_test": sorted(groups_by_split["val"] & groups_by_split["test"]),
    }
    summaries = {
        name: _split_summary(name, split_rows[name], split_inventory[name])
        for name in SPLIT_NAMES
    }
    total_duration = sum(value["duration_s"] for value in summaries.values())
    achieved = [summaries[name]["duration_s"] / total_duration for name in SPLIT_NAMES]
    largest_group_fraction = max(group.duration_s for group in groups) / total_duration
    tolerance = max(0.03, largest_group_fraction + 0.01)
    source_class_segments = np.sum(
        np.asarray(
            [row["class_segment_count"] for row in inventory], dtype=np.int64
        ),
        axis=0,
    )
    source_classes = set(np.flatnonzero(source_class_segments).astype(int).tolist())
    three_way = set(optimizer["three_way_coverable_classes"])
    source_tasks = {
        str(row["task_id"]) for row in inventory if str(row["task_id"])
    }
    three_way_tasks = set(optimizer["three_way_coverable_tasks"])
    checks = {
        "input_manifest_nonempty": bool(rows),
        "episode_inventory_complete": len(inventory) == len(rows),
        "episode_partition_exact": input_captures == output_captures
        and sum(len(value) for value in split_rows.values()) == len(rows),
        "group_partition_disjoint": not any(overlaps.values()),
        "all_splits_nonempty": all(split_rows[name] for name in SPLIT_NAMES),
        "all_source_classes_in_train": source_classes.issubset(
            summaries["train"]["class_indices"]
        ),
        "three_way_coverable_classes_in_val": three_way.issubset(
            summaries["val"]["class_indices"]
        ),
        "three_way_coverable_classes_in_test": three_way.issubset(
            summaries["test"]["class_indices"]
        ),
        "all_source_tasks_in_train": source_tasks.issubset(
            summaries["train"]["task_ids"]
        ),
        "three_way_coverable_tasks_in_val": three_way_tasks.issubset(
            summaries["val"]["task_ids"]
        ),
        "three_way_coverable_tasks_in_test": three_way_tasks.issubset(
            summaries["test"]["task_ids"]
        ),
        "duration_fraction_within_group_tolerance": all(
            abs(actual - target) <= tolerance
            for actual, target in zip(achieved, fractions)
        ),
        "hdf5_hand_order_verified": all(
            row["tactile_hand_order"] == ["left", "right"] for row in inventory
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"formal protocol audit failed: {checks}")

    paths = {
        "episode_inventory": output_dir / "episode_inventory.jsonl",
        "group_assignments": output_dir / "group_assignments.jsonl",
        "train": output_dir / "train.jsonl",
        "val": output_dir / "val.jsonl",
        "test": output_dir / "test.jsonl",
    }
    _atomic_jsonl(paths["episode_inventory"], inventory)
    group_rows = []
    for group in groups:
        group_rows.append(
            {
                "schema_version": SPLIT_SCHEMA,
                "split_group_id": group.name,
                "split": assignment[group.name],
                "episodes": group.episodes,
                "duration_s": group.duration_s,
                "hours": group.duration_s / 3600.0,
                "class_indices": np.flatnonzero(group.segments).astype(int).tolist(),
                "class_segment_count": group.segments.astype(int).tolist(),
                "class_frame_count_60": group.frames.astype(int).tolist(),
                "task_ids": sorted(group.tasks),
            }
        )
    _atomic_jsonl(paths["group_assignments"], group_rows)
    for name in SPLIT_NAMES:
        _atomic_jsonl(paths[name], split_rows[name])
    artifacts = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    reportable = [
        class_index
        for class_index in range(NUM_CLASSES)
        if summaries["train"]["class_segment_count"][class_index] >= 50
        and summaries["test"]["class_segment_count"][class_index] >= 20
        and summaries["test"]["class_group_count"][class_index] >= 5
    ]
    audit = {
        "schema_version": PROTOCOL_SCHEMA,
        "status": "pass",
        "checks": checks,
        "target_duration_fractions": dict(zip(SPLIT_NAMES, fractions)),
        "achieved_duration_fractions": dict(zip(SPLIT_NAMES, achieved)),
        "duration_fraction_tolerance": tolerance,
        "largest_source_group_fraction": largest_group_fraction,
        "group_overlap": overlaps,
        "splits": summaries,
        "class_policy": {
            "all_source_classes": sorted(source_classes),
            "three_way_coverable_classes": sorted(three_way),
            "not_three_way_coverable_classes": optimizer[
                "not_three_way_coverable_classes"
            ],
            "reportable_macro_rule": {
                "minimum_train_segments": 50,
                "minimum_test_segments": 20,
                "minimum_test_groups": 5,
            },
            "reportable_macro_classes": reportable,
            "nonreportable_macro_classes": sorted(source_classes - set(reportable)),
        },
        "task_policy": {
            "all_source_tasks": sorted(source_tasks),
            "three_way_coverable_tasks": sorted(three_way_tasks),
            "not_three_way_coverable_tasks": optimizer[
                "not_three_way_coverable_tasks"
            ],
            "balancing": "task duration distribution plus group-level presence",
        },
        "evaluation_policy": {
            "wave0_diagnostic": WAVE0_DIAGNOSTIC_METRIC_POLICY,
            "formal_reportable": FORMAL_EVALUATION_POLICY,
        },
        "optimizer": optimizer,
        "artifacts": artifacts,
    }
    audit_path = output_dir / "audit.json"
    _atomic_json(audit_path, audit)
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "status": "frozen",
        "purpose": "formal_reportable_experiment",
        "created_unix_s": time.time(),
        "hostname": socket.gethostname(),
        "training_manifest": str(training_manifest),
        "training_manifest_sha256": sha256_file(training_manifest),
        "episodes": len(rows),
        "source_groups": len(groups),
        "seed": seed,
        "split_unit": "split_group_id",
        "split_unit_claim": (
            "conservative collection-code/capture-directory group; not a verified "
            "participant identity"
        ),
        "target_duration_fractions": dict(zip(SPLIT_NAMES, fractions)),
        "test_policy": (
            "sealed; never used for checkpoint, hyperparameter, threshold, NMS, "
            "minimum-duration, or model selection"
        ),
        "protocol_scope": (
            "data split, test seal, metric semantics, and calibration search space; "
            "formal evaluator implementation hashes are a required pre-main gate"
        ),
        "evaluation_policy": {
            "wave0_diagnostic": WAVE0_DIAGNOSTIC_METRIC_POLICY,
            "formal_reportable": FORMAL_EVALUATION_POLICY,
        },
        "hand_order_policy": (
            "HDF5 is already [left,right] after formal correction; training must not "
            "swap tactile hands again"
        ),
        "audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "artifacts": artifacts,
    }
    protocol_path = output_dir / "protocol.json"
    _atomic_json(protocol_path, protocol)
    print(
        json.dumps(
            {
                "event": "formal500h_protocol_frozen",
                "status": "PASS",
                "output_dir": str(output_dir),
                "episodes": len(rows),
                "source_groups": len(groups),
                "hours": {name: summaries[name]["hours"] for name in SPLIT_NAMES},
                "groups": {name: summaries[name]["groups"] for name in SPLIT_NAMES},
                "reportable_macro_classes": reportable,
                "protocol_sha256": sha256_file(protocol_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and audit the formal500h group-disjoint experiment split"
    )
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--optimizer-restarts", type=int, default=32)
    parser.add_argument("--optimizer-iterations", type=int, default=4000)
    parser.add_argument("--minimum-groups-per-split", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    freeze_formal_protocol(
        training_manifest=Path(arguments.training_manifest),
        output_dir=Path(arguments.output_dir),
        fractions=(
            arguments.train_fraction,
            arguments.val_fraction,
            arguments.test_fraction,
        ),
        seed=arguments.seed,
        workers=arguments.workers,
        restarts=arguments.optimizer_restarts,
        iterations=arguments.optimizer_iterations,
        minimum_groups_per_split=arguments.minimum_groups_per_split,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
