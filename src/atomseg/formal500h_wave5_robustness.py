"""Formal500h Wave-5 upper-level and visual-degradation audits.

This module is deliberately validation-only.  It replays frozen checkpoints,
applies deterministic missing-vision masks before inference, and scores both
the frozen 26-class task and a deterministic upper-level ontology view.  The
upper-level view is post-hoc reporting: it never recalibrates a decoder,
selects a checkpoint, changes a label, or reads the sealed test manifest.

The number of upper-level groups is read from the supplied ontology.  The
current Formal500h secondary analysis uses the preregistered eight-phase
ontology; accepting an arbitrary declared mapping keeps the evaluator useful
for historical artifacts without baking a legacy label count into the code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .formal500h_evaluation import (
    GroundTruthEpisode,
    decode_evaluable_prediction,
    evaluate_dataset,
    ground_truth_segments,
)
from .formal500h_protocol import NUM_CLASSES, sha256_file
from .formal500h_wave4_metric_audit import (
    PREDICTION_SCHEMA,
    load_predictions,
)
from .inference import StitchedEpisode, WindowPrediction, stitch_window_predictions
from . import formal500h_wave1_train as wave1
from . import formal500h_wave4_crossarch as crossarch


AUDIT_SCHEMA = "formal500h_wave5_upper_visual_audit_v1"
AUDIT_COMPLETE_SCHEMA = "formal500h_wave5_upper_visual_audit_complete_v1"
AGGREGATE_SCHEMA = "formal500h_wave5_upper_visual_aggregate_v1"
AGGREGATE_COMPLETE_SCHEMA = "formal500h_wave5_upper_visual_complete_v1"
HAND_AGGREGATE_SCHEMA = "formal500h_wave5_hand_loss_aggregate_v1"
HAND_COMPLETE_SCHEMA = "formal500h_wave5_hand_loss_complete_v1"
SAVED_HIERARCHY_SCHEMA = "formal500h_saved_hierarchy_aggregate_v1"

BOUNDARY_TOLERANCES_MS = (250, 500, 1000)
OCCLUSION_HEAD = "ASRF"
OCCLUSION_REPRESENTATIONS = ("V", "VT")


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
        raise ValueError(f"expected JSON object: {path}")
    return value


def _mean_std(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("cannot aggregate an empty metric")
    return {
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "values": numbers,
    }


def _valid_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
    if not indices.size:
        return []
    output: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        index = int(raw)
        if index != previous + 1:
            output.append((start, previous + 1))
            start = index
        previous = index
    output.append((start, previous + 1))
    return output


def load_group_mapping(ontology_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    groups = raw.get("groups") if isinstance(raw, dict) else None
    actions = raw.get("actions") if isinstance(raw, dict) else None
    if not isinstance(groups, list) or not isinstance(actions, list):
        raise ValueError("ontology lacks groups/actions")
    group_ids = [str(value["group_id"]) for value in groups]
    group_count = len(group_ids)
    if group_count < 2 or len(set(group_ids)) != group_count:
        raise ValueError("upper-level view requires at least two unique groups")
    group_index = {name: index for index, name in enumerate(group_ids)}
    mapping = np.full(NUM_CLASSES, -1, dtype=np.int64)
    class_rows: list[dict[str, Any]] = []
    for row in actions:
        source = int(row["class_index"])
        if not 0 <= source < NUM_CLASSES or mapping[source] >= 0:
            raise ValueError("ontology class indices are incomplete or duplicated")
        target = group_index[str(row["group_id"])]
        mapping[source] = target
        class_rows.append(
            {
                "class_index": source,
                "label_id": str(row["label_id"]),
                "action_name": str(row["action_name"]),
                "group_index": target,
                "group_id": group_ids[target],
            }
        )
    if np.any(mapping < 0):
        raise ValueError("ontology does not map all 26 classes")
    if not np.array_equal(np.unique(mapping), np.arange(group_count)):
        raise ValueError("ontology declares an unused or non-contiguous group")
    return {
        "ontology_path": str(ontology_path.resolve()),
        "ontology_sha256": sha256_file(ontology_path),
        "ontology_version": str(raw.get("ontology_version", "")),
        "group_count": group_count,
        "group_ids": group_ids,
        "group_display_names_zh": [str(value["display_name_zh"]) for value in groups],
        "class_to_group": mapping.tolist(),
        "actions": sorted(class_rows, key=lambda value: value["class_index"]),
    }


def upper_view_key(mapping_record: Mapping[str, Any]) -> str:
    """Return the stable result key for a declared upper-level ontology.

    Result files use ``fine26`` for the native task and ``upperN`` for the
    ontology supplied to the command, where *N* is its declared group count.
    Keeping this derivation in one place prevents a historical ontology from
    silently changing the meaning of a current phase report.
    """

    try:
        group_count = int(mapping_record["group_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("mapping record lacks a valid group_count") from exc
    if group_count < 2:
        raise ValueError("upper-level view requires at least two groups")
    return f"upper{group_count}"


def coarse_ground_truth(
    truth: Mapping[str, GroundTruthEpisode],
    class_to_group: Sequence[int],
) -> dict[str, GroundTruthEpisode]:
    mapping = np.asarray(class_to_group, dtype=np.int64)
    if mapping.shape != (NUM_CLASSES,):
        raise ValueError("invalid 26-to-upper class mapping")
    group_count = int(mapping.max()) + 1
    if group_count < 2 or np.any(
        (mapping < 0) | (mapping >= group_count)
    ) or not np.array_equal(np.unique(mapping), np.arange(group_count)):
        raise ValueError("invalid 26-to-upper class mapping")
    output: dict[str, GroundTruthEpisode] = {}
    for episode_id, source in truth.items():
        valid = source.label_mask & (source.atomic_target >= 0)
        target = np.full(source.num_ticks, -100, dtype=np.int64)
        target[valid] = mapping[source.atomic_target[valid]]
        instance = np.full(source.num_ticks, -100, dtype=np.int64)
        boundary_hard = np.zeros(source.num_ticks, dtype=np.bool_)
        boundary_type = np.zeros(source.num_ticks, dtype=np.int8)
        next_instance = 0
        for start, end in _valid_runs(valid):
            instance[start] = next_instance
            for index in range(start + 1, end):
                if target[index] != target[index - 1]:
                    next_instance += 1
                    boundary_hard[index] = True
                    boundary_type[index] = 1
                instance[index] = next_instance
            next_instance += 1
        output[episode_id] = GroundTruthEpisode(
            episode_id=episode_id,
            global_frame_60=source.global_frame_60.copy(),
            atomic_target=target,
            atomic_instance_id=instance,
            boundary_hard=boundary_hard,
            boundary_type=boundary_type,
            label_mask=source.label_mask.copy(),
            boundary_eval_mask=source.boundary_eval_mask.copy(),
            time_ns=None if source.time_ns is None else source.time_ns.copy(),
            metadata={**source.metadata, "view": f"ontology_group_{group_count}"},
        )
    return output


def _decoder(summary: Mapping[str, Any]) -> dict[str, Any]:
    selected = (summary.get("calibrated_validation") or {}).get("selected") or {}
    required = {
        "boundary_threshold",
        "boundary_min_distance_frames",
        "min_segment_frames",
    }
    if not required.issubset(selected):
        raise ValueError("source summary lacks frozen validation decoder")
    return {key: selected[key] for key in required}


def _prediction_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        schema = handle.attrs.get("schema_version", "")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        if schema != PREDICTION_SCHEMA or not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"invalid persisted prediction artifact: {path}")
        raw = handle.attrs.get("metadata_json")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError("prediction artifact lacks metadata_json")
    metadata = json.loads(raw)
    if metadata.get("test_manifest_read") is not False:
        raise ValueError("Wave-5 accepts validation predictions only")
    decoder = metadata.get("fixed_decoder")
    if not isinstance(decoder, dict):
        raise ValueError("prediction metadata lacks fixed_decoder")
    return metadata


def _decoded_fine_labels(
    predictions: Mapping[str, StitchedEpisode],
    truth: Mapping[str, GroundTruthEpisode],
    *,
    decoder: Mapping[str, Any],
    use_boundary_head: bool,
) -> dict[str, np.ndarray]:
    labels: dict[str, np.ndarray] = {}
    for episode_id in sorted(truth):
        prediction = predictions[episode_id]
        target = truth[episode_id]
        if not np.array_equal(prediction.global_frame_60, target.global_frame_60):
            raise ValueError(f"prediction/GT clock mismatch: {episode_id}")
        evaluation_mask, segments, _ = decode_evaluable_prediction(
            prediction.atomic_logits,
            prediction.boundary_probability,
            target,
            prediction.coverage_mask,
            boundary_threshold=float(decoder["boundary_threshold"]),
            boundary_min_distance_frames=int(decoder["boundary_min_distance_frames"]),
            min_segment_frames=int(decoder["min_segment_frames"]),
            use_boundary_head=use_boundary_head,
            include_action_changes=True,
        )
        value = np.full(target.num_ticks, -100, dtype=np.int64)
        for segment in segments:
            value[segment.start_index : segment.end_index] = int(segment.label)
        if np.any(evaluation_mask & (value < 0)):
            raise RuntimeError(f"decoded labels do not cover evaluation ticks: {episode_id}")
        labels[episode_id] = value
    return labels


def _synthetic_group_predictions(
    fine_labels: Mapping[str, np.ndarray],
    coarse_truth: Mapping[str, GroundTruthEpisode],
    class_to_group: Sequence[int],
) -> dict[str, StitchedEpisode]:
    mapping = np.asarray(class_to_group, dtype=np.int64)
    group_count = int(mapping.max()) + 1
    output: dict[str, StitchedEpisode] = {}
    for episode_id in sorted(coarse_truth):
        target = coarse_truth[episode_id]
        fine = np.asarray(fine_labels[episode_id], dtype=np.int64)
        valid = target.label_mask & (fine >= 0) & (fine < NUM_CLASSES)
        grouped = np.full(target.num_ticks, -100, dtype=np.int64)
        grouped[valid] = mapping[fine[valid]]
        logits = np.zeros((target.num_ticks, group_count), dtype=np.float32)
        indices = np.flatnonzero(valid)
        logits[indices, grouped[indices]] = 1.0
        output[episode_id] = StitchedEpisode(
            episode_id=episode_id,
            global_frame_60=target.global_frame_60,
            atomic_logits=logits,
            boundary_probability=np.zeros(target.num_ticks, dtype=np.float32),
            coverage_mask=valid,
            contribution_count=valid.astype(np.int64),
            weight_sum=valid.astype(np.float32),
            time_ns=target.time_ns,
        )
    return output


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    primary = metrics["primary"]
    return {
        "primary_composite": float(primary["checkpoint_composite"]),
        "collapsed_f1@10": float(metrics["collapsed"]["f1@10"]),
        "collapsed_f1@25": float(metrics["collapsed"]["f1@25"]),
        "collapsed_f1@50": float(metrics["collapsed"]["f1@50"]),
        "collapsed_edit": float(metrics["collapsed"]["edit"]),
        "class_change_bf1@250ms": float(
            metrics["boundary"]["class_change"]["250ms"]["f1"]
        ),
        "class_change_bf1@500ms": float(
            metrics["boundary"]["class_change"]["500ms"]["f1"]
        ),
        "class_change_bf1@1000ms": float(
            metrics["boundary"]["class_change"]["1000ms"]["f1"]
        ),
        "frame_accuracy": float(metrics["frame_accuracy"]),
        "tick_macro_f1": float(metrics["tick_macro_f1"]),
    }


def _audit_upper_view(value: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    """Validate and return the upper-level key declared by an audit row."""

    mapping = value.get("ontology_group_view")
    if not isinstance(mapping, Mapping):
        raise ValueError("audit lacks ontology_group_view")
    key = upper_view_key(mapping)
    conditions = value.get("conditions")
    clean = conditions.get("clean") if isinstance(conditions, Mapping) else None
    if not isinstance(clean, Mapping) or key not in clean:
        raise ValueError(f"audit lacks declared upper-level result {key}")
    return key, mapping


def _audit_matches_mapping(
    value: Mapping[str, Any], mapping: Mapping[str, Any]
) -> bool:
    """Return whether an existing audit was produced with ``mapping``."""

    try:
        key, stored = _audit_upper_view(value)
    except (TypeError, ValueError):
        return False
    return (
        key == upper_view_key(mapping)
        and stored.get("ontology_sha256") == mapping.get("ontology_sha256")
        and tuple(stored.get("class_to_group") or ())
        == tuple(mapping.get("class_to_group") or ())
    )


def evaluate_views(
    predictions: Mapping[str, StitchedEpisode],
    truth: Mapping[str, GroundTruthEpisode],
    *,
    decoder: Mapping[str, Any],
    use_boundary_head: bool,
    mapping_record: Mapping[str, Any],
) -> dict[str, Any]:
    fine = evaluate_dataset(
        predictions,
        truth,
        num_actions=NUM_CLASSES,
        segment_iou_thresholds=(0.10, 0.25, 0.50),
        boundary_tolerances_ms=BOUNDARY_TOLERANCES_MS,
        primary_boundary_tolerance_ms=250,
        output_rate_hz=30.0,
        boundary_threshold=float(decoder["boundary_threshold"]),
        boundary_min_distance_frames=int(decoder["boundary_min_distance_frames"]),
        min_segment_frames=int(decoder["min_segment_frames"]),
        use_boundary_head=use_boundary_head,
        include_action_changes=True,
        strict=True,
        collect_artifacts=False,
    ).metrics
    coarse_truth = coarse_ground_truth(truth, mapping_record["class_to_group"])
    fine_labels = _decoded_fine_labels(
        predictions,
        truth,
        decoder=decoder,
        use_boundary_head=use_boundary_head,
    )
    coarse_predictions = _synthetic_group_predictions(
        fine_labels,
        coarse_truth,
        mapping_record["class_to_group"],
    )
    group_count = int(mapping_record["group_count"])
    coarse = evaluate_dataset(
        coarse_predictions,
        coarse_truth,
        num_actions=group_count,
        segment_iou_thresholds=(0.10, 0.25, 0.50),
        boundary_tolerances_ms=BOUNDARY_TOLERANCES_MS,
        primary_boundary_tolerance_ms=250,
        output_rate_hz=30.0,
        boundary_threshold=1.0,
        boundary_min_distance_frames=1,
        min_segment_frames=1,
        use_boundary_head=False,
        include_action_changes=True,
        strict=True,
        collect_artifacts=False,
    ).metrics
    fine_segments = sum(
        len(ground_truth_segments(value, strict=True)) for value in truth.values()
    )
    coarse_segments = sum(
        len(ground_truth_segments(value, strict=True)) for value in coarse_truth.values()
    )
    upper_key = f"upper{group_count}"
    return {
        "fine26": _compact_metrics(fine),
        upper_key: _compact_metrics(coarse),
        "ground_truth_segment_accounting": {
            "fine26_raw_instances": fine_segments,
            "upper_group_count": group_count,
            "upper_segments": coarse_segments,
            "upper_survival_pct": 100.0 * coarse_segments / max(fine_segments, 1),
            "merged_or_removed_pct": 100.0 * (fine_segments - coarse_segments) / max(fine_segments, 1),
        },
    }


def corruption_conditions() -> tuple[dict[str, Any], ...]:
    """Return the historical v1 missingness plan.

    This function is intentionally retained for audit replay.  The field named
    ``global_frame_60`` in the Clock-v2 runtime actually contains dense 30-Hz
    output ticks, so the historical 60/240-tick blocks correspond to roughly
    2/8 seconds.  New experiments must use
    :func:`corruption_conditions_time_ns_v2` instead.
    """
    return (
        {"id": "clean", "kind": "clean"},
        {"id": "drop_head", "kind": "views", "views": [0]},
        {"id": "drop_left", "kind": "views", "views": [1]},
        {"id": "drop_right", "kind": "views", "views": [2]},
        {"id": "drop_head_left", "kind": "views", "views": [0, 1]},
        {"id": "drop_head_right", "kind": "views", "views": [0, 2]},
        {"id": "drop_left_right", "kind": "views", "views": [1, 2]},
        # Historical condition encoding.  Kept byte-for-byte at the plan level
        # so old validation evidence remains reproducible and auditable.
        {"id": "block1s_p25", "kind": "blocks", "block_frames_60": 60, "rate": 0.25},
        {"id": "block1s_p50", "kind": "blocks", "block_frames_60": 60, "rate": 0.50},
        {"id": "block1s_p75", "kind": "blocks", "block_frames_60": 60, "rate": 0.75},
        {"id": "block4s_p50", "kind": "blocks", "block_frames_60": 240, "rate": 0.50},
        {"id": "drop_all", "kind": "views", "views": [0, 1, 2]},
    )


def corruption_conditions_time_ns_v2() -> tuple[dict[str, Any], ...]:
    """Return the corrected plan on the episode-relative nanosecond clock.

    ``time_ns`` is used instead of the legacy, misleadingly named
    ``global_frame_60`` field.  Subtracting the immutable episode origin makes
    the block assignment both episode-relative and identical in overlapping
    windows from the same episode.
    """
    return (
        {"id": "clean", "kind": "clean"},
        {"id": "drop_head", "kind": "views", "views": [0]},
        {"id": "drop_left", "kind": "views", "views": [1]},
        {"id": "drop_right", "kind": "views", "views": [2]},
        {"id": "drop_head_left", "kind": "views", "views": [0, 1]},
        {"id": "drop_head_right", "kind": "views", "views": [0, 2]},
        {"id": "drop_left_right", "kind": "views", "views": [1, 2]},
        {
            "id": "block1s_p25",
            "kind": "blocks",
            "clock_field": "time_ns",
            "origin_field": "episode_start_time_ns",
            "block_duration_ns": 1_000_000_000,
            "rate": 0.25,
        },
        {
            "id": "block1s_p50",
            "kind": "blocks",
            "clock_field": "time_ns",
            "origin_field": "episode_start_time_ns",
            "block_duration_ns": 1_000_000_000,
            "rate": 0.50,
        },
        {
            "id": "block1s_p75",
            "kind": "blocks",
            "clock_field": "time_ns",
            "origin_field": "episode_start_time_ns",
            "block_duration_ns": 1_000_000_000,
            "rate": 0.75,
        },
        {
            "id": "block4s_p50",
            "kind": "blocks",
            "clock_field": "time_ns",
            "origin_field": "episode_start_time_ns",
            "block_duration_ns": 4_000_000_000,
            "rate": 0.50,
        },
        {"id": "drop_all", "kind": "views", "views": [0, 1, 2]},
    )


def _block_selected(*, seed: int, episode_id: str, block_index: int, rate: float) -> bool:
    payload = f"{seed}|{episode_id}|{block_index}".encode("utf-8")
    number = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return number / float(2**64) < rate


def _drop_mask(batch: Mapping[str, Any], condition: Mapping[str, Any], *, seed: int) -> torch.Tensor:
    valid = batch["vision_valid_60"]
    mask = torch.zeros_like(valid, dtype=torch.bool)
    kind = str(condition["kind"])
    if kind == "clean":
        return mask
    if kind == "views":
        mask[..., [int(value) for value in condition["views"]]] = True
        return mask
    if kind != "blocks":
        raise ValueError(f"unknown corruption kind: {kind}")
    if "block_duration_ns" in condition:
        if condition.get("clock_field") != "time_ns":
            raise ValueError("nanosecond block conditions must declare clock_field=time_ns")
        if condition.get("origin_field") != "episode_start_time_ns":
            raise ValueError(
                "nanosecond block conditions must declare "
                "origin_field=episode_start_time_ns"
            )
        duration_ns = int(condition["block_duration_ns"])
        if duration_ns <= 0:
            raise ValueError("block_duration_ns must be positive")
        clock_values = batch["time_ns"].detach().cpu().numpy()
        origins = batch["episode_start_time_ns"].detach().cpu().numpy()
        if origins.ndim != 1 or origins.shape[0] != clock_values.shape[0]:
            raise ValueError("episode_start_time_ns must contain one origin per batch row")
        block_ids_by_row = np.floor_divide(
            clock_values - origins[:, None], duration_ns
        )
    else:
        # Backward-compatible replay of the historical v1 plan.  Despite its
        # name, global_frame_60 is a dense 30-Hz output-tick index in Clock-v2.
        frames = batch["global_frame_60"].detach().cpu().numpy()
        block_frames_60 = int(condition["block_frames_60"])
        if block_frames_60 <= 0:
            raise ValueError("block_frames_60 must be positive")
        block_ids_by_row = np.floor_divide(frames, block_frames_60)
    rate = float(condition["rate"])
    if not 0.0 <= rate <= 1.0:
        raise ValueError("block missingness rate must lie in [0, 1]")
    for row, episode_id in enumerate(batch["episode_id"]):
        block_ids = block_ids_by_row[row]
        selected = np.asarray(
            [
                _block_selected(
                    seed=seed,
                    episode_id=str(episode_id),
                    block_index=int(value),
                    rate=rate,
                )
                for value in block_ids
            ],
            dtype=np.bool_,
        )
        mask[row, :, :] = torch.from_numpy(selected)[:, None]
    return mask


def _corrupt_gpu_batch(
    batch: Mapping[str, Any],
    drop_mask_cpu: torch.Tensor,
) -> dict[str, Any]:
    output = dict(batch)
    device = batch["vision_valid_60"].device
    drop = drop_mask_cpu.to(device=device)
    output["vision_valid_60"] = batch["vision_valid_60"] & ~drop
    cache_valid = batch["vision_cache_valid"].clone()
    indices = batch["vision_index_60"]
    for view in range(3):
        affected = drop[..., view] & (indices[..., view] >= 0)
        for row in range(affected.shape[0]):
            selected = indices[row, :, view][affected[row]].unique()
            if selected.numel():
                cache_valid[row, selected, view] = False
    output["vision_cache_valid"] = cache_valid
    return output


@torch.no_grad()
def predict_conditions(
    model: torch.nn.Module,
    dataset: crossarch.VariantAwareFormal500hDataset,
    *,
    conditions: Sequence[Mapping[str, Any]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    precision: str,
    seed: int,
    cell: str,
    collate_fn: Callable[[Sequence[dict[str, Any]]], dict[str, Any]] | None = None,
) -> dict[str, dict[str, StitchedEpisode]]:
    resolved_collate = (
        crossarch.collate_formal500h_fast if collate_fn is None else collate_fn
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=resolved_collate,
    )
    windows: dict[str, list[WindowPrediction]] = {
        str(value["id"]): [] for value in conditions
    }
    amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    model.eval()
    for batch_index, cpu_batch in enumerate(loader, start=1):
        masks = {
            str(condition["id"]): _drop_mask(cpu_batch, condition, seed=seed)
            for condition in conditions
        }
        batch = wave1._to_device(cpu_batch, device)
        for condition in conditions:
            condition_id = str(condition["id"])
            model_batch = _corrupt_gpu_batch(batch, masks[condition_id])
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                output = model(model_batch)
            atomic = output["atomic_logits"].float().detach().cpu().numpy()
            boundary = (
                torch.sigmoid(output["boundary_logits"].float()).detach().cpu().numpy()
            )
            sequence = cpu_batch["sequence_mask"].numpy().astype(bool)
            frames = cpu_batch["global_frame_60"].numpy()
            time_ns = cpu_batch["time_ns"].numpy()
            for row, episode_id in enumerate(cpu_batch["episode_id"]):
                used = sequence[row]
                windows[condition_id].append(
                    WindowPrediction(
                        episode_id=str(episode_id),
                        global_frame_60=frames[row, used],
                        atomic_logits=atomic[row, used],
                        boundary_probability=boundary[row, used],
                        time_ns=time_ns[row, used],
                        source_id=(
                            f"{condition_id}:{episode_id}:"
                            f"{int(frames[row, used][0])}"
                        ),
                    )
                )
        if batch_index % 100 == 0:
            print(
                json.dumps(
                    {
                        "event": "wave5_audit_inference_progress",
                        "cell": cell,
                        "seed": seed,
                        "batches": batch_index,
                        "conditions": len(conditions),
                    }
                ),
                flush=True,
            )
    return {
        condition_id: stitch_window_predictions(
            values, weighting="uniform", require_full_coverage=True
        )
        for condition_id, values in windows.items()
    }


def _validated_wave4(exp_root: Path) -> dict[str, Any]:
    aggregate_path = exp_root / "wave4/cross_arch/aggregate.json"
    complete_path = exp_root / "wave4/cross_arch/WAVE4_CROSSARCH_COMPLETE.json"
    aggregate = _read_json(aggregate_path)
    complete = _read_json(complete_path)
    checks = {
        "aggregate_complete": aggregate.get("status") == "complete",
        "completion_complete": complete.get("status") == "complete",
        "aggregate_hash": complete.get("aggregate_sha256") == sha256_file(aggregate_path),
        "jobs_27": aggregate.get("jobs") == 27 and complete.get("jobs") == 27,
        "test_sealed": aggregate.get("test_manifest_read") is False
        and complete.get("test_manifest_read") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid Wave-4 release: {checks}")
    return {
        "aggregate": str(aggregate_path.resolve()),
        "aggregate_sha256": sha256_file(aggregate_path),
        "complete": str(complete_path.resolve()),
        "complete_sha256": sha256_file(complete_path),
    }


def _source_truth(data_dir: Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, GroundTruthEpisode]]:
    data_summary, artifacts = wave1._validated_data(data_dir)
    if data_summary.get("test_manifest_read") is not False:
        raise ValueError("Wave-5 data release is not test-sealed")
    truth = wave1._ground_truth(wave1._read_jsonl(artifacts["val_episodes"]))
    return data_summary, artifacts, truth


def audit_crossarch_cell(
    *,
    exp_root: Path,
    data_dir: Path,
    ontology_path: Path,
    output_dir: Path,
    head: str,
    representation: str,
    seed: int,
    device_name: str,
    batch_size: int,
    num_workers: int,
    visual_corruptions: bool,
) -> dict[str, Any]:
    started = time.time()
    head, representation = head.upper(), representation.upper()
    if head not in crossarch.HEADS or representation not in crossarch.REPRESENTATIONS:
        raise ValueError("unknown Wave-4 cell")
    if seed not in crossarch.SEEDS:
        raise ValueError("unknown Wave-4 seed")
    if visual_corruptions and (
        head != OCCLUSION_HEAD or representation not in OCCLUSION_REPRESENTATIONS
    ):
        raise ValueError("visual corruption phase is pre-registered for ASRF V/VT")
    mapping = load_group_mapping(ontology_path)
    summary_path = output_dir / "summary.json"
    complete_path = output_dir / "COMPLETE.json"
    if summary_path.is_file() and complete_path.is_file():
        result = _read_json(summary_path)
        complete = _read_json(complete_path)
        if (
            result.get("status") == "complete"
            and complete.get("summary_sha256") == sha256_file(summary_path)
            and _audit_matches_mapping(result, mapping)
        ):
            print(json.dumps({"event": "wave5_audit_reused", "output": str(output_dir)}))
            return result
        raise ValueError(
            "existing Wave-5 audit is incomplete or uses a different ontology: "
            f"{output_dir}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"partial Wave-5 output requires inspection: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    wave4 = _validated_wave4(exp_root)
    run_dir = (
        exp_root
        / "wave4/cross_arch/runs"
        / head
        / representation
        / f"seed_{seed}"
    )
    summary, resolved = crossarch.validate_run(
        run_dir, head=head, representation=representation, seed=seed
    )
    data_summary, artifacts, truth = _source_truth(data_dir)
    if summary["data_summary_sha256"] != sha256_file(data_dir / "summary.json"):
        raise ValueError("Wave-4 checkpoint and Wave-5 data differ")
    checkpoint_path = Path(str(summary["best_checkpoint"]))
    # Wave-4 checkpoints intentionally use the cross-architecture training
    # schema.  _checkpoint_load is shared with Wave-1 and validates against
    # its module-level schema, so bind the expected Wave-4 schema only for
    # this integrity-checked load.  Without this, every Wave-5 replay rejects
    # valid Wave-4 checkpoints as a Wave-1 schema mismatch.
    original_training_schema = wave1.TRAINING_SCHEMA
    wave1.TRAINING_SCHEMA = crossarch.TRAINING_SCHEMA
    try:
        checkpoint = wave1._checkpoint_load(checkpoint_path, torch.device("cpu"))
    finally:
        wave1.TRAINING_SCHEMA = original_training_schema
    if checkpoint.get("resolved_config") != resolved:
        raise ValueError("checkpoint resolved configuration changed")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    spec = crossarch.CELL_BY_ID[f"{head}_{representation}"]
    model = crossarch.CrossArchFormal500hModel(resolved["model"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    dataset = crossarch.VariantAwareFormal500hDataset(
        artifacts["val_windows"],
        artifacts["train_statistics"],
        expected_statistics_schema=crossarch.WAVE1_DATA_SCHEMA,
        enabled_modalities=spec.modalities,
        hdf5_cache_size=8,
    )
    conditions = corruption_conditions() if visual_corruptions else ({"id": "clean", "kind": "clean"},)
    predictions = predict_conditions(
        model,
        dataset,
        conditions=conditions,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        precision=str(resolved["precision"]),
        seed=seed,
        cell=spec.cell,
    )
    decoder = _decoder(summary)
    upper_key = upper_view_key(mapping)
    by_condition: dict[str, Any] = {}
    for condition in conditions:
        condition_id = str(condition["id"])
        by_condition[condition_id] = {
            "condition": dict(condition),
            **evaluate_views(
                predictions[condition_id],
                truth,
                decoder=decoder,
                use_boundary_head=spec.use_boundary_head,
                mapping_record=mapping,
            ),
        }
        print(
            json.dumps(
                {
                    "event": "wave5_condition_complete",
                    "cell": spec.cell,
                    "seed": seed,
                    "condition": condition_id,
                    "fine_primary": by_condition[condition_id]["fine26"]["primary_composite"],
                    "upper_primary": by_condition[condition_id][upper_key]["primary_composite"],
                }
            ),
            flush=True,
        )
    source_clean_metrics = {
        name: crossarch.METRICS[name](summary) for name in METRIC_NAMES
    }
    clean_replay_delta = {
        name: by_condition["clean"]["fine26"][name] - source_value
        for name, source_value in source_clean_metrics.items()
    }
    clean_replay_max_abs_delta = max(abs(value) for value in clean_replay_delta.values())
    del predictions, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "complete",
        "purpose": f"validation_only_{upper_key}_and_visual_missingness_audit",
        "created_unix_s": time.time(),
        "elapsed_s": time.time() - started,
        "test_manifest_read": False,
        "model_selection_performed": False,
        "head": head,
        "representation": representation,
        "seed": seed,
        "visual_corruptions": visual_corruptions,
        "source": {
            **wave4,
            "run_dir": str(run_dir.resolve()),
            "summary_sha256": sha256_file(run_dir / "summary.json"),
            "best_checkpoint_sha256": sha256_file(checkpoint_path),
            "data_summary_sha256": sha256_file(data_dir / "summary.json"),
            "source_protocol_sha256": data_summary["source_protocol_sha256"],
            "source_audit_sha256": data_summary["source_audit_sha256"],
        },
        "ontology_group_view": mapping,
        "decoder": decoder,
        "conditions": by_condition,
        "clean_replay_crosscheck": {
            "source_metrics": source_clean_metrics,
            "delta": clean_replay_delta,
            "maximum_absolute_delta": clean_replay_max_abs_delta,
            "tolerance": 0.05,
        },
        "checks": {
            "wave4_complete": True,
            "test_sealed": True,
            "frozen_checkpoint": True,
            "frozen_decoder": True,
            "all_conditions_complete": set(by_condition)
            == {str(value["id"]) for value in conditions},
            "no_recalibration": True,
            "no_model_selection": True,
            "clean_replay_matches_formal_wave4": clean_replay_max_abs_delta <= 0.05,
        },
    }
    if not all(result["checks"].values()):
        raise RuntimeError(f"Wave-5 audit failed: {result['checks']}")
    _atomic_json(summary_path, result)
    complete = {
        "schema_version": AUDIT_COMPLETE_SCHEMA,
        "status": "complete",
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "test_manifest_read": False,
    }
    _atomic_json(complete_path, complete)
    return result


def audit_saved_predictions(
    *,
    data_dir: Path,
    ontology_path: Path,
    prediction_path: Path,
    output_dir: Path,
    label: str,
    seed: int,
) -> dict[str, Any]:
    started = time.time()
    mapping = load_group_mapping(ontology_path)
    upper_key = upper_view_key(mapping)
    summary_path = output_dir / "summary.json"
    complete_path = output_dir / "COMPLETE.json"
    if summary_path.is_file() and complete_path.is_file():
        value = _read_json(summary_path)
        complete = _read_json(complete_path)
        if (
            value.get("status") == "complete"
            and complete.get("summary_sha256") == sha256_file(summary_path)
            and _audit_matches_mapping(value, mapping)
        ):
            return value
        raise ValueError(
            "existing saved-prediction audit is incomplete or uses a different "
            f"ontology: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _prediction_metadata(prediction_path)
    predictions = load_predictions(prediction_path)
    data_summary, _, truth = _source_truth(data_dir)
    views = evaluate_views(
        predictions,
        truth,
        decoder=metadata["fixed_decoder"],
        use_boundary_head=True,
        mapping_record=mapping,
    )
    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "complete",
        "purpose": f"validation_only_{upper_key}_saved_prediction_audit",
        "created_unix_s": time.time(),
        "elapsed_s": time.time() - started,
        "test_manifest_read": False,
        "model_selection_performed": False,
        "head": "ASRF_INSPIRED",
        "representation": label,
        "seed": seed,
        "visual_corruptions": False,
        "source": {
            "prediction": str(prediction_path.resolve()),
            "prediction_sha256": sha256_file(prediction_path),
            "data_summary_sha256": sha256_file(data_dir / "summary.json"),
            "source_protocol_sha256": data_summary["source_protocol_sha256"],
            "source_audit_sha256": data_summary["source_audit_sha256"],
        },
        "ontology_group_view": mapping,
        "decoder": metadata["fixed_decoder"],
        "conditions": {"clean": {"condition": {"id": "clean", "kind": "clean"}, **views}},
        "checks": {
            "test_sealed": True,
            "persisted_prediction_valid": True,
            "frozen_decoder": True,
            "no_recalibration": True,
            "no_model_selection": True,
        },
    }
    _atomic_json(summary_path, result)
    _atomic_json(
        complete_path,
        {
            "schema_version": AUDIT_COMPLETE_SCHEMA,
            "status": "complete",
            "summary": str(summary_path.resolve()),
            "summary_sha256": sha256_file(summary_path),
            "test_manifest_read": False,
        },
    )
    return result


METRIC_NAMES = (
    "primary_composite",
    "collapsed_f1@10",
    "collapsed_f1@25",
    "collapsed_f1@50",
    "collapsed_edit",
    "class_change_bf1@250ms",
    "class_change_bf1@500ms",
    "class_change_bf1@1000ms",
    "frame_accuracy",
    "tick_macro_f1",
)


def aggregate_audits(*, audits_root: Path, output: Path) -> dict[str, Any]:
    cross_values: dict[str, dict[int, dict[str, Any]]] = {}
    upper_key: str | None = None
    upper_mapping_signature: tuple[Any, ...] | None = None
    for head in crossarch.HEADS:
        for representation in crossarch.REPRESENTATIONS:
            cell = f"{head}_{representation}"
            cross_values[cell] = {}
            for seed in crossarch.SEEDS:
                path = audits_root / "crossarch" / head / representation / f"seed_{seed}" / "summary.json"
                value = _read_json(path)
                if value.get("status") != "complete" or value.get("test_manifest_read") is not False:
                    raise ValueError(f"invalid Wave-5 cell audit: {path}")
                row_upper_key, row_mapping = _audit_upper_view(value)
                row_signature = (
                    row_mapping.get("ontology_sha256"),
                    tuple(row_mapping.get("class_to_group") or ()),
                )
                if upper_key is None:
                    upper_key = row_upper_key
                    upper_mapping_signature = row_signature
                elif row_upper_key != upper_key or row_signature != upper_mapping_signature:
                    raise ValueError(
                        "Wave-5 cells use different upper-level ontologies: "
                        f"expected {upper_key}, found {row_upper_key} in {path}"
                    )
                cross_values[cell][seed] = value
    if upper_key is None or upper_mapping_signature is None:
        raise ValueError("Wave-5 aggregate has no cross-architecture cells")
    views = ("fine26", upper_key)
    aggregate: dict[str, Any] = {}
    for cell, seeds in cross_values.items():
        aggregate[cell] = {}
        for view in views:
            aggregate[cell][view] = {
                metric: _mean_std(
                    [seeds[seed]["conditions"]["clean"][view][metric] for seed in crossarch.SEEDS]
                )
                for metric in METRIC_NAMES
            }
    paired: dict[str, Any] = {}
    for head in crossarch.HEADS:
        for first, second in (("V", "VT"), ("M2R2", "VT")):
            key = f"{head}_{second}_minus_{first}"
            paired[key] = {}
            for view in views:
                paired[key][view] = {}
                for metric in METRIC_NAMES:
                    differences = [
                        cross_values[f"{head}_{second}"][seed]["conditions"]["clean"][view][metric]
                        - cross_values[f"{head}_{first}"][seed]["conditions"]["clean"][view][metric]
                        for seed in crossarch.SEEDS
                    ]
                    paired[key][view][metric] = {
                        **_mean_std(differences),
                        "wins": sum(value > 0 for value in differences),
                    }

    mainline_values: dict[str, dict[int, dict[str, Any]]] = {"B0": {}, "D30B": {}}
    for label, seeds in (
        ("B0", crossarch.SEEDS),
        ("D30B", (*crossarch.SEEDS, 20260827, 20260828, 20260829, 20260830)),
    ):
        for seed in seeds:
            path = audits_root / "mainline" / label / f"seed_{seed}" / "summary.json"
            value = _read_json(path)
            if value.get("status") != "complete" or value.get("test_manifest_read") is not False:
                raise ValueError(f"invalid mainline upper-level audit: {path}")
            row_upper_key, row_mapping = _audit_upper_view(value)
            row_signature = (
                row_mapping.get("ontology_sha256"),
                tuple(row_mapping.get("class_to_group") or ()),
            )
            if row_upper_key != upper_key or row_signature != upper_mapping_signature:
                raise ValueError(
                    "mainline audit uses a different upper-level ontology: "
                    f"expected {upper_key}, found {row_upper_key} in {path}"
                )
            mainline_values[label][seed] = value
    mainline_aggregate: dict[str, Any] = {}
    for label, seeds in mainline_values.items():
        mainline_aggregate[label] = {}
        for view in views:
            mainline_aggregate[label][view] = {
                metric: _mean_std(
                    [value["conditions"]["clean"][view][metric] for value in seeds.values()]
                )
                for metric in METRIC_NAMES
            }
    mainline_paired: dict[str, Any] = {"D30B_minus_B0_original3": {}}
    for view in views:
        mainline_paired["D30B_minus_B0_original3"][view] = {}
        for metric in METRIC_NAMES:
            differences = [
                mainline_values["D30B"][seed]["conditions"]["clean"][view][metric]
                - mainline_values["B0"][seed]["conditions"]["clean"][view][metric]
                for seed in crossarch.SEEDS
            ]
            mainline_paired["D30B_minus_B0_original3"][view][metric] = {
                **_mean_std(differences),
                "wins": sum(value > 0 for value in differences),
            }

    condition_ids = [str(value["id"]) for value in corruption_conditions()]
    occlusion: dict[str, Any] = {}
    for condition_id in condition_ids:
        occlusion[condition_id] = {}
        for view in views:
            occlusion[condition_id][view] = {}
            for metric in METRIC_NAMES:
                rescue = [
                    cross_values["ASRF_VT"][seed]["conditions"][condition_id][view][metric]
                    - cross_values["ASRF_V"][seed]["conditions"][condition_id][view][metric]
                    for seed in crossarch.SEEDS
                ]
                clean = [
                    cross_values["ASRF_VT"][seed]["conditions"]["clean"][view][metric]
                    - cross_values["ASRF_V"][seed]["conditions"]["clean"][view][metric]
                    for seed in crossarch.SEEDS
                ]
                interaction = [value - base for value, base in zip(rescue, clean)]
                occlusion[condition_id][view][metric] = {
                    "tactile_rescue_vt_minus_v": {
                        **_mean_std(rescue),
                        "wins": sum(value > 0 for value in rescue),
                    },
                    "interaction_over_clean": {
                        **_mean_std(interaction),
                        "wins": sum(value > 0 for value in interaction),
                    },
                    "v": _mean_std(
                        [cross_values["ASRF_V"][seed]["conditions"][condition_id][view][metric] for seed in crossarch.SEEDS]
                    ),
                    "vt": _mean_std(
                        [cross_values["ASRF_VT"][seed]["conditions"][condition_id][view][metric] for seed in crossarch.SEEDS]
                    ),
                }
    first = cross_values["ASRF_V"][crossarch.SEEDS[0]]
    accounting = first["conditions"]["clean"]["ground_truth_segment_accounting"]
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "complete",
        "purpose": f"validation_only_{upper_key}_and_visual_missingness_aggregate",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "model_selection_performed": False,
        "jobs": 27,
        "seeds": list(crossarch.SEEDS),
        "upper_view": upper_key,
        "upper_group_mapping": first["ontology_group_view"],
        f"{upper_key}_mapping": first["ontology_group_view"],
        "ground_truth_segment_accounting": accounting,
        "mainline_aggregate": mainline_aggregate,
        "mainline_paired": mainline_paired,
        "aggregate": aggregate,
        "paired": paired,
        "visual_missingness": {
            "head": OCCLUSION_HEAD,
            "representations": list(OCCLUSION_REPRESENTATIONS),
            "conditions": occlusion,
            "primary_hypothesis": (
                "interaction_over_clean_positive_means_touch_is_more_valuable_when_vision_is_missing"
            ),
            "spatial_occlusion_claimed": False,
        },
        "checks": {
            "all_27_crossarch_cells": len(cross_values) == 9
            and all(len(value) == 3 for value in cross_values.values()),
            "mainline_10_saved_predictions": len(mainline_values["B0"]) == 3
            and len(mainline_values["D30B"]) == 7,
            "all_occlusion_conditions": all(
                condition_id in cross_values[cell][seed]["conditions"]
                for cell in ("ASRF_V", "ASRF_VT")
                for seed in crossarch.SEEDS
                for condition_id in condition_ids
            ),
            "same_ground_truth_mapping": all(
                value["conditions"]["clean"]["ground_truth_segment_accounting"] == accounting
                for seeds in cross_values.values()
                for value in seeds.values()
            ),
            "test_sealed": True,
            "no_model_selection": True,
        },
    }
    if not all(result["checks"].values()):
        raise RuntimeError(f"Wave-5 aggregate failed: {result['checks']}")
    _atomic_json(output, result)
    complete_path = output.parent / "WAVE5_UPPER_VISUAL_COMPLETE.json"
    _atomic_json(
        complete_path,
        {
            "schema_version": AGGREGATE_COMPLETE_SCHEMA,
            "status": "complete",
            "aggregate": str(output.resolve()),
            "aggregate_sha256": sha256_file(output),
            "jobs": 27,
            "test_manifest_read": False,
        },
    )
    return result


def aggregate_saved_hierarchy(
    *, audits_root: Path, output: Path, view: str | None = None
) -> dict[str, Any]:
    variant_seeds = {
        "B0": crossarch.SEEDS,
        "D30B": (*crossarch.SEEDS, 20260827, 20260828, 20260829, 20260830),
    }
    rows: dict[str, dict[int, dict[str, Any]]] = {}
    mapping_hashes: set[str] = set()
    resolved_view: str | None = None
    for variant, seeds in variant_seeds.items():
        rows[variant] = {}
        for seed in seeds:
            path = audits_root / variant / f"seed_{seed}" / "summary.json"
            value = _read_json(path)
            checks = value.get("checks") or {}
            mapping = value.get("ontology_group_view") or {}
            try:
                row_view, _ = _audit_upper_view(value)
            except ValueError as exc:
                raise ValueError(f"invalid saved hierarchy audit: {path}") from exc
            if resolved_view is None:
                resolved_view = row_view
                if view is not None and view != resolved_view:
                    raise ValueError(
                        f"saved hierarchy audit view mismatch: expected {view}, "
                        f"found {resolved_view} in {path}"
                    )
            elif row_view != resolved_view:
                raise ValueError(
                    "saved hierarchy audits use different upper-level views: "
                    f"expected {resolved_view}, found {row_view} in {path}"
                )
            if (
                value.get("status") != "complete"
                or value.get("test_manifest_read") is not False
                or row_view not in (value.get("conditions") or {}).get("clean", {})
                or not all(bool(item) for item in checks.values())
            ):
                raise ValueError(f"invalid saved hierarchy audit: {path}")
            mapping_hashes.add(str(mapping.get("ontology_sha256")))
            rows[variant][seed] = value
    if resolved_view is None:
        raise ValueError("saved hierarchy aggregate has no audits")
    if len(mapping_hashes) != 1:
        raise ValueError("saved hierarchy audits use different mappings")

    aggregate: dict[str, Any] = {}
    for variant, by_seed in rows.items():
        aggregate[variant] = {
            result_view: {
                metric: _mean_std(
                    [by_seed[seed]["conditions"]["clean"][result_view][metric] for seed in by_seed]
                )
                for metric in METRIC_NAMES
            }
            for result_view in ("fine26", resolved_view)
        }
    paired: dict[str, Any] = {}
    for result_view in ("fine26", resolved_view):
        paired[result_view] = {}
        for metric in METRIC_NAMES:
            differences = [
                rows["D30B"][seed]["conditions"]["clean"][result_view][metric]
                - rows["B0"][seed]["conditions"]["clean"][result_view][metric]
                for seed in crossarch.SEEDS
            ]
            paired[result_view][metric] = {
                **_mean_std(differences),
                "wins": sum(value > 0 for value in differences),
            }
    first = rows["B0"][crossarch.SEEDS[0]]
    result = {
        "schema_version": SAVED_HIERARCHY_SCHEMA,
        "status": "complete",
        "purpose": f"quick_validation_only_{resolved_view}_upper_level_audit",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "model_selection_performed": False,
        "view": resolved_view,
        "mapping": first["ontology_group_view"],
        "ground_truth_segment_accounting": first["conditions"]["clean"][
            "ground_truth_segment_accounting"
        ],
        "aggregate": aggregate,
        "paired_D30B_minus_B0_original3": paired,
        "checks": {
            "b0_three_seeds": len(rows["B0"]) == 3,
            "d30b_seven_seeds": len(rows["D30B"]) == 7,
            "one_preregistered_mapping": len(mapping_hashes) == 1,
            "test_sealed": True,
            "no_retraining": True,
            "no_recalibration": True,
            "no_model_selection": True,
        },
    }
    if not all(result["checks"].values()):
        raise RuntimeError(f"saved hierarchy aggregate failed: {result['checks']}")
    _atomic_json(output, result)
    _atomic_json(
        output.parent / "ROBOT_PHASE8_QUICK_COMPLETE.json",
        {
            "schema_version": SAVED_HIERARCHY_SCHEMA,
            "status": "complete",
            "aggregate": str(output.resolve()),
            "aggregate_sha256": sha256_file(output),
            "audits": 10,
            "test_manifest_read": False,
        },
    )
    return result


def train_hand_zero(
    *, data_dir: Path, output_dir: Path, seed: int, device_name: str
) -> dict[str, Any]:
    if seed not in crossarch.SEEDS:
        raise ValueError("hand ablation seed is not frozen")
    original_weight = crossarch.FROZEN_HAND_WEIGHT
    original_lineage = crossarch._release_lineage

    def lineage(spec: crossarch.CellSpec) -> dict[str, Any]:
        upstream = original_lineage(spec)
        return {
            **upstream,
            "schema_version": "formal500h_wave5_hand_loss_runtime_v1",
            "purpose": "paired_hand_loss_zero_ablation",
            "formal_wave4_crossarch_upstream": upstream,
            "wave5_hand_ablation": {
                "module": str(Path(__file__).resolve()),
                "module_sha256": sha256_file(Path(__file__)),
                "hand_weight": 0.0,
                "head": "ASRF",
                "representation": "VT",
                "test_manifest_read": False,
            },
        }

    crossarch.FROZEN_HAND_WEIGHT = 0.0
    crossarch._release_lineage = lineage
    try:
        result = crossarch.train_cell(
            data_dir=data_dir,
            output_dir=output_dir,
            head="ASRF",
            representation="VT",
            seed=seed,
            device=device_name,
            precision="bf16",
            batch_size=16,
            num_workers=20,
            epochs=3,
            learning_rate=3.0e-4,
            calibration_workers=4,
        )
    finally:
        crossarch.FROZEN_HAND_WEIGHT = original_weight
        crossarch._release_lineage = original_lineage
    resolved = _read_json(Path(str(result["resolved_config"])))
    marker = (
        (resolved.get("implementation_release") or {})
        .get("formal_wave4_crossarch", {})
        .get("wave5_hand_ablation", {})
    )
    if not math.isclose(float(resolved.get("hand_weight", -1)), 0.0, abs_tol=0.0) or marker.get("hand_weight") != 0.0:
        raise RuntimeError("hand-loss-zero lineage was not persisted")
    return result


def aggregate_hand_ablation(*, exp_root: Path, hand_root: Path, output: Path) -> dict[str, Any]:
    h1: dict[int, dict[str, Any]] = {}
    h0: dict[int, dict[str, Any]] = {}
    for seed in crossarch.SEEDS:
        source = exp_root / "wave4/cross_arch/runs/ASRF/VT" / f"seed_{seed}"
        h1[seed], _ = crossarch.validate_run(source, head="ASRF", representation="VT", seed=seed)
        target = hand_root / "runs/H0" / f"seed_{seed}"
        h0[seed], resolved = crossarch.validate_run(target, head="ASRF", representation="VT", seed=seed)
        marker = (
            (resolved.get("implementation_release") or {})
            .get("formal_wave4_crossarch", {})
            .get("wave5_hand_ablation", {})
        )
        if float(resolved.get("hand_weight", -1)) != 0.0 or marker.get("hand_weight") != 0.0:
            raise ValueError(f"not a Wave-5 hand-zero run: {target}")
    metrics: dict[str, Any] = {}
    for name, getter in crossarch.METRICS.items():
        values = [getter(h1[seed]) - getter(h0[seed]) for seed in crossarch.SEEDS]
        metrics[name] = {**_mean_std(values), "wins": sum(value > 0 for value in values)}
    result = {
        "schema_version": HAND_AGGREGATE_SCHEMA,
        "status": "complete",
        "purpose": "paired_asrf_vt_hand_loss_ablation_validation_only",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "head": "ASRF",
        "representation": "VT",
        "seeds": list(crossarch.SEEDS),
        "contrast": "hand_weight_0.1_minus_0.0",
        "metrics": metrics,
        "decision_scope": "auxiliary_hand_loss_only_no_test_selection",
    }
    _atomic_json(output, result)
    _atomic_json(
        output.parent / "WAVE5_HAND_ABLATION_COMPLETE.json",
        {
            "schema_version": HAND_COMPLETE_SCHEMA,
            "status": "complete",
            "aggregate": str(output.resolve()),
            "aggregate_sha256": sha256_file(output),
            "jobs": 3,
            "test_manifest_read": False,
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cell = commands.add_parser("audit-cell")
    cell.add_argument("--exp-root", type=Path, required=True)
    cell.add_argument("--data-dir", type=Path, required=True)
    cell.add_argument("--ontology", type=Path, required=True)
    cell.add_argument("--output-dir", type=Path, required=True)
    cell.add_argument("--head", required=True, choices=crossarch.HEADS)
    cell.add_argument("--representation", required=True, choices=crossarch.REPRESENTATIONS)
    cell.add_argument("--seed", type=int, required=True, choices=crossarch.SEEDS)
    cell.add_argument("--device", default="cuda:0")
    cell.add_argument("--batch-size", type=int, default=16)
    cell.add_argument("--num-workers", type=int, default=20)
    cell.add_argument("--visual-corruptions", action="store_true")
    saved = commands.add_parser("audit-saved")
    saved.add_argument("--data-dir", type=Path, required=True)
    saved.add_argument("--ontology", type=Path, required=True)
    saved.add_argument("--prediction", type=Path, required=True)
    saved.add_argument("--output-dir", type=Path, required=True)
    saved.add_argument("--label", required=True)
    saved.add_argument("--seed", type=int, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--audits-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    saved_aggregate = commands.add_parser("aggregate-saved")
    saved_aggregate.add_argument("--audits-root", type=Path, required=True)
    saved_aggregate.add_argument("--output", type=Path, required=True)
    saved_aggregate.add_argument("--view", default="upper8")
    hand = commands.add_parser("train-hand-zero")
    hand.add_argument("--data-dir", type=Path, required=True)
    hand.add_argument("--output-dir", type=Path, required=True)
    hand.add_argument("--seed", type=int, required=True, choices=crossarch.SEEDS)
    hand.add_argument("--device", default="cuda:0")
    hand_aggregate = commands.add_parser("aggregate-hand")
    hand_aggregate.add_argument("--exp-root", type=Path, required=True)
    hand_aggregate.add_argument("--hand-root", type=Path, required=True)
    hand_aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "audit-cell":
        result = audit_crossarch_cell(
            exp_root=arguments.exp_root,
            data_dir=arguments.data_dir,
            ontology_path=arguments.ontology,
            output_dir=arguments.output_dir,
            head=arguments.head,
            representation=arguments.representation,
            seed=arguments.seed,
            device_name=arguments.device,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            visual_corruptions=arguments.visual_corruptions,
        )
    elif arguments.command == "audit-saved":
        result = audit_saved_predictions(
            data_dir=arguments.data_dir,
            ontology_path=arguments.ontology,
            prediction_path=arguments.prediction,
            output_dir=arguments.output_dir,
            label=arguments.label,
            seed=arguments.seed,
        )
    elif arguments.command == "aggregate":
        result = aggregate_audits(audits_root=arguments.audits_root, output=arguments.output)
    elif arguments.command == "aggregate-saved":
        result = aggregate_saved_hierarchy(
            audits_root=arguments.audits_root,
            output=arguments.output,
            view=arguments.view,
        )
    elif arguments.command == "train-hand-zero":
        result = train_hand_zero(
            data_dir=arguments.data_dir,
            output_dir=arguments.output_dir,
            seed=arguments.seed,
            device_name=arguments.device,
        )
    else:
        result = aggregate_hand_ablation(
            exp_root=arguments.exp_root,
            hand_root=arguments.hand_root,
            output=arguments.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
