"""Wave-4 validation-only common-metric and output-rate audit.

This gate replays frozen B0 and D30B validation checkpoints.  It never reads
the formal test manifest, never recalibrates a decoder, and never performs
model selection.  The audit adds two reporting views:

* class-change Boundary-F1 at 250/500/1000 ms plus an explicitly adapted
  M2R2-style detection rate at +/-1 second;
* a causal 2/5/10/15/30 Hz semantic-output sensitivity sweep, always scored
  against the untouched 30 Hz validation ground truth.
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
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .formal500h_evaluation import (
    GroundTruthEpisode,
    decode_evaluable_prediction,
    evaluate_dataset,
    ground_truth_boundary_types,
    match_boundaries_optimal,
)
from .formal500h_infra_runtime import (
    VariantAwareFormal500hDataset,
    collate_formal500h_fast,
)
from .formal500h_protocol import NUM_CLASSES, sha256_file
from .formal500h_runtime import Formal500hModel
from .formal500h_wave1_data import WAVE1_DATA_SCHEMA, WINDOW_SCHEMA
from .inference import (
    BOUNDARY_CLASS_CHANGE,
    StitchedEpisode,
    WindowPrediction,
    segments_to_frame_labels,
)
from .inference import stitch_window_predictions
from . import formal500h_wave1_train as wave1
from . import formal500h_wave3b_vt as wave3b
from . import formal500h_wave3c_train as wave3c


AUDIT_SCHEMA = "formal500h_wave4_metric_rate_audit_v1"
ORACLE_SCHEMA = "formal500h_wave4_rate_oracle_v1"
AGGREGATE_SCHEMA = "formal500h_wave4_metric_rate_aggregate_v1"
COMPLETE_SCHEMA = "formal500h_wave4_metric_rate_complete_v1"
AUDIT_COMPLETE_SCHEMA = "formal500h_wave4_checkpoint_audit_complete_v1"
PREDICTION_SCHEMA = "formal500h_wave4_stitched_predictions_v1"
WAVE3C_COMPLETE_SCHEMA = "formal500h_wave3c_complete_v1"

MASTER_RATE_HZ = 30
RATES_HZ = (2, 5, 10, 15, 30)
STRIDE_BY_RATE = {rate: MASTER_RATE_HZ // rate for rate in RATES_HZ}
BOUNDARY_TOLERANCES_MS = (50, 100, 250, 500, 1000)
HEADLINE_TOLERANCES_MS = (250, 500, 1000)
ORIGINAL_SEEDS = (20260824, 20260825, 20260826)
REPLICATION_SEEDS = (20260827, 20260828, 20260829, 20260830)
D30B_SEEDS = ORIGINAL_SEEDS + REPLICATION_SEEDS
VARIANT_SEEDS = {
    "B0": ORIGINAL_SEEDS,
    "D30B": D30B_SEEDS,
}
MODALITIES = {
    "B0": ("vision",),
    "D30B": ("vision", "tactile"),
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _valid_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    indices = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
    if indices.size == 0:
        return
    start = int(indices[0])
    previous = start
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index != previous + 1:
            yield start, previous + 1
            start = index
        previous = index
    yield start, previous + 1


def causal_sample_and_hold(
    labels: np.ndarray | Sequence[int],
    valid_mask: np.ndarray | Sequence[bool],
    *,
    source_rate_hz: int = MASTER_RATE_HZ,
    target_rate_hz: int,
) -> np.ndarray:
    """Run-anchored causal sampling and zero-order hold.

    No label crosses an invalid gap.  The first tick of every valid run is the
    fixed phase-zero sample, so the operation never chooses a GT-aware phase.
    """

    labels_value = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if labels_value.ndim != 1 or valid.shape != labels_value.shape:
        raise ValueError("labels and valid_mask must be aligned one-dimensional arrays")
    if (
        not isinstance(source_rate_hz, (int, np.integer))
        or isinstance(source_rate_hz, (bool, np.bool_))
        or source_rate_hz <= 0
    ):
        raise ValueError("source rate must be a positive integer")
    if (
        not isinstance(target_rate_hz, (int, np.integer))
        or isinstance(target_rate_hz, (bool, np.bool_))
        or target_rate_hz <= 0
        or source_rate_hz % target_rate_hz
    ):
        raise ValueError("target rate must be a positive integer divisor of source rate")
    stride = source_rate_hz // target_rate_hz
    output = np.full(labels_value.shape, -100, dtype=np.int64)
    for start, end in _valid_runs(valid):
        sample_positions = np.arange(start, end, stride, dtype=np.int64)
        for sample_index, position in enumerate(sample_positions):
            next_position = (
                int(sample_positions[sample_index + 1])
                if sample_index + 1 < sample_positions.size
                else end
            )
            output[int(position) : next_position] = labels_value[int(position)]
    return output


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = 100.0 * tp / (tp + fp) if tp + fp else 0.0
    recall = 100.0 * tp / (tp + fn) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": 200.0 * tp / denominator if denominator else 0.0,
    }


def gap_safe_class_change_counts(
    predicted_labels: Mapping[str, np.ndarray],
    ground_truth: Mapping[str, GroundTruthEpisode],
    *,
    tolerance_ticks: int,
) -> dict[str, float | int]:
    """Micro counts with matching isolated by episode and contiguous valid run."""

    if set(predicted_labels) != set(ground_truth):
        raise ValueError("prediction and ground-truth episode sets differ")
    tp = fp = fn = 0
    for episode_id in sorted(ground_truth):
        truth = ground_truth[episode_id]
        predicted = np.asarray(predicted_labels[episode_id], dtype=np.int64)
        if predicted.shape != truth.atomic_target.shape:
            raise ValueError(f"label shape mismatch: {episode_id}")
        valid = truth.label_mask & (truth.atomic_target >= 0)
        if np.any(valid & (predicted < 0)):
            raise ValueError(f"prediction is missing on an evaluable tick: {episode_id}")
        for start, end in _valid_runs(valid):
            predicted_boundaries = [
                int(truth.global_frame_60[index])
                for index in range(start + 1, end)
                if truth.boundary_eval_mask[index]
                and predicted[index] != predicted[index - 1]
            ]
            target_boundaries = [
                int(truth.global_frame_60[index])
                for index in range(start + 1, end)
                if truth.boundary_hard[index]
                and truth.boundary_eval_mask[index]
                and int(truth.boundary_type[index]) == 1
            ]
            matched = match_boundaries_optimal(
                predicted_boundaries,
                target_boundaries,
                tolerance_ticks,
            )
            tp += matched.true_positive
            fp += matched.false_positive
            fn += matched.false_negative
    return _prf(tp, fp, fn)


def gap_safe_typed_boundary_counts(
    predicted_boundary_types: Mapping[str, Mapping[int, str]],
    ground_truth: Mapping[str, GroundTruthEpisode],
    *,
    evaluation_type: str,
    tolerance_ticks: int,
) -> dict[str, float | int]:
    """Match a decoded boundary stream without crossing an invalid GT run.

    ``evaluation_type='all'`` is the closest within-dataset adaptation of the
    M2R2 detection-rate boundary stream.  A named type (currently
    ``class_change``) filters both prediction and ground truth before the same
    one-to-one ordered matcher is applied.
    """

    if evaluation_type not in {"all", BOUNDARY_CLASS_CHANGE}:
        raise ValueError(f"unsupported Wave-4 boundary view: {evaluation_type}")
    if tolerance_ticks < 0:
        raise ValueError("tolerance_ticks must be non-negative")
    if set(predicted_boundary_types) != set(ground_truth):
        raise ValueError("prediction and ground-truth episode sets differ")
    tp = fp = fn = 0
    for episode_id in sorted(ground_truth):
        truth = ground_truth[episode_id]
        predicted = predicted_boundary_types[episode_id]
        target = ground_truth_boundary_types(truth, strict=True)
        tick_to_index = {
            int(tick): index
            for index, tick in enumerate(truth.global_frame_60.tolist())
        }
        unknown = set(predicted) - set(tick_to_index)
        if unknown:
            raise ValueError(
                f"predicted boundary lies outside GT clock: {episode_id}: "
                f"{sorted(unknown)[:3]}"
            )
        valid = truth.label_mask & (truth.atomic_target >= 0)
        for start, end in _valid_runs(valid):
            allowed_ticks = {
                int(truth.global_frame_60[index])
                for index in range(start + 1, end)
                if truth.boundary_eval_mask[index]
            }
            predicted_ticks = [
                int(tick)
                for tick, boundary_type in predicted.items()
                if int(tick) in allowed_ticks
                and (
                    evaluation_type == "all"
                    or boundary_type == evaluation_type
                )
            ]
            target_ticks = [
                int(tick)
                for tick, boundary_type in target.items()
                if int(tick) in allowed_ticks
                and (
                    evaluation_type == "all"
                    or boundary_type == evaluation_type
                )
            ]
            matched = match_boundaries_optimal(
                predicted_ticks,
                target_ticks,
                tolerance_ticks,
            )
            tp += matched.true_positive
            fp += matched.false_positive
            fn += matched.false_negative
    return _prf(tp, fp, fn)


def _synthetic_predictions(
    labels: Mapping[str, np.ndarray],
    truth: Mapping[str, GroundTruthEpisode],
) -> dict[str, StitchedEpisode]:
    output: dict[str, StitchedEpisode] = {}
    for episode_id in sorted(truth):
        target = truth[episode_id]
        value = np.asarray(labels[episode_id], dtype=np.int64)
        if value.shape != target.atomic_target.shape:
            raise ValueError(f"synthetic label shape mismatch: {episode_id}")
        coverage = target.label_mask & (value >= 0) & (value < NUM_CLASSES)
        logits = np.zeros((value.size, NUM_CLASSES), dtype=np.float32)
        rows = np.flatnonzero(coverage)
        logits[rows, value[rows]] = 1.0
        output[episode_id] = StitchedEpisode(
            episode_id=episode_id,
            global_frame_60=target.global_frame_60,
            atomic_logits=logits,
            boundary_probability=np.zeros(value.size, dtype=np.float32),
            coverage_mask=coverage,
            contribution_count=coverage.astype(np.int64),
            weight_sum=coverage.astype(np.float32),
            time_ns=target.time_ns,
        )
    return output


def _semantic_metrics(
    labels: Mapping[str, np.ndarray],
    truth: Mapping[str, GroundTruthEpisode],
) -> dict[str, Any]:
    evaluated = evaluate_dataset(
        _synthetic_predictions(labels, truth),
        truth,
        num_actions=NUM_CLASSES,
        segment_iou_thresholds=(0.10, 0.25, 0.50),
        boundary_tolerances_ms=BOUNDARY_TOLERANCES_MS,
        primary_boundary_tolerance_ms=250,
        output_rate_hz=float(MASTER_RATE_HZ),
        boundary_threshold=1.0,
        boundary_min_distance_frames=1,
        min_segment_frames=1,
        use_boundary_head=False,
        include_action_changes=True,
        strict=True,
        collect_artifacts=False,
    ).metrics
    compact = _compact_metrics(evaluated)
    # A semantic-label rate proxy cannot encode a boundary between adjacent
    # instances that share the same class.  Therefore only the class-change
    # boundary stream is reportable for this post-hoc rate sweep.
    compact.pop("all_boundary")
    gap_safe: dict[str, Any] = {}
    compatible = True
    for tolerance_ms in HEADLINE_TOLERANCES_MS:
        tolerance_ticks = int(round(tolerance_ms * MASTER_RATE_HZ / 1000.0))
        safe = gap_safe_class_change_counts(
            labels,
            truth,
            tolerance_ticks=tolerance_ticks,
        )
        frozen = compact["class_change_boundary"][f"{tolerance_ms}ms"]
        gap_safe[f"{tolerance_ms}ms"] = safe
        compatible = compatible and all(
            int(safe[key]) == int(frozen[key]) for key in ("tp", "fp", "fn")
        )
    compact["gap_safe_class_change_boundary"] = gap_safe
    compact["gap_safe_matches_frozen_episode_scope_counts"] = compatible
    return compact


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    class_change_boundary = metrics["boundary"]["class_change"]
    all_boundary = metrics["boundary"]["all"]
    collapsed = metrics["collapsed"]
    if "predicted_segments" in collapsed:
        semantic_segment_count = int(collapsed["predicted_segments"])
    else:
        semantic_segment_count = int(
            collapsed["counts@10"]["tp"] + collapsed["counts@10"]["fp"]
        )
    return {
        "frame_accuracy": float(metrics["frame_accuracy"]),
        "tick_macro_f1": float(metrics["tick_macro_f1"]),
        "collapsed": {
            "edit": float(collapsed["edit"]),
            "f1@10": float(collapsed["f1@10"]),
            "f1@25": float(collapsed["f1@25"]),
            "f1@50": float(collapsed["f1@50"]),
        },
        "class_change_boundary": {
            f"{value}ms": {
                key: class_change_boundary[f"{value}ms"][key]
                for key in ("tp", "fp", "fn", "precision", "recall", "f1")
            }
            for value in HEADLINE_TOLERANCES_MS
        },
        "all_boundary": {
            f"{value}ms": {
                key: all_boundary[f"{value}ms"][key]
                for key in ("tp", "fp", "fn", "precision", "recall", "f1")
            }
            for value in HEADLINE_TOLERANCES_MS
        },
        "semantic_segment_count": semantic_segment_count,
    }


def _contract() -> dict[str, Any]:
    tolerance_ticks = {
        f"{value}ms": int(round(value * MASTER_RATE_HZ / 1000.0))
        for value in HEADLINE_TOLERANCES_MS
    }
    return {
        "master_rate_hz": MASTER_RATE_HZ,
        "headline_boundary_type": "class_change",
        "requested_tolerances_ms": list(HEADLINE_TOLERANCES_MS),
        "tolerance_ticks": tolerance_ticks,
        "effective_tolerance_ms": {
            name: 1000.0 * ticks / MASTER_RATE_HZ
            for name, ticks in tolerance_ticks.items()
        },
        "inclusive_tolerance_endpoints": True,
        "matcher": "ordered_dp_max_cardinality_then_min_abs_error_v1",
        "match_scope": "episode_and_contiguous_evaluable_run",
        "aggregation": "micro_within_seed_mean_std_across_seeds",
        "adapted_dr": {
            "name": "m2r2_style_adapted_dr_all_boundaries@1000ms",
            "boundary_type": "all_annotated_action_boundaries",
            "tolerance_ms": 1000,
            "tolerance_ticks_at_30hz": 30,
            "official_m2r2_reproduction": False,
            "note": (
                "metric adapted to this dataset; it is not a cross-dataset "
                "official reproduction"
            ),
        },
        "rate_operator": "run_anchored_causal_sample_and_zero_order_hold_v1",
        "rate_sweep_interpretation": "semantic_output_resolution_proxy_not_retraining",
        "rates_hz": list(RATES_HZ),
        "stride_ticks": {str(key): value for key, value in STRIDE_BY_RATE.items()},
        "rate_phase": "first_tick_of_each_contiguous_valid_run",
        "rate_ground_truth": "untouched_30hz",
        "decoder_recalibrated": False,
        "model_selection_performed": False,
    }


def _data_and_truth(
    data_dir: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, GroundTruthEpisode]]:
    data_summary, artifacts = wave1._validated_data(data_dir)
    rows = wave1._read_jsonl(artifacts["val_episodes"])
    truth = wave1._ground_truth(rows)
    return data_summary, artifacts, truth


def build_rate_oracle(*, data_dir: Path, output: Path) -> dict[str, Any]:
    started = time.time()
    data_summary, artifacts, truth = _data_and_truth(data_dir)
    native = {
        episode_id: target.atomic_target.copy()
        for episode_id, target in truth.items()
    }
    by_rate: dict[str, Any] = {}
    for rate in RATES_HZ:
        labels = {
            episode_id: causal_sample_and_hold(
                native[episode_id],
                target.label_mask,
                target_rate_hz=rate,
            )
            for episode_id, target in truth.items()
        }
        metrics = _semantic_metrics(labels, truth)
        by_rate[str(rate)] = {
            "rate_hz": rate,
            "stride_ticks": STRIDE_BY_RATE[rate],
            "metrics": metrics,
        }
        print(
            json.dumps(
                {
                    "event": "wave4_oracle_rate_complete",
                    "rate_hz": rate,
                    "collapsed_f1@50": metrics["collapsed"]["f1@50"],
                    "class_change_boundary_f1@1000ms": metrics[
                        "class_change_boundary"
                    ]["1000ms"]["f1"],
                }
            ),
            flush=True,
        )
    native_metrics = by_rate[str(MASTER_RATE_HZ)]["metrics"]
    perfect = all(
        math.isclose(value, 100.0, abs_tol=1.0e-12)
        for value in (
            native_metrics["frame_accuracy"],
            native_metrics["tick_macro_f1"],
            native_metrics["collapsed"]["edit"],
            native_metrics["collapsed"]["f1@10"],
            native_metrics["collapsed"]["f1@25"],
            native_metrics["collapsed"]["f1@50"],
            native_metrics["class_change_boundary"]["250ms"]["f1"],
            native_metrics["class_change_boundary"]["500ms"]["f1"],
            native_metrics["class_change_boundary"]["1000ms"]["f1"],
        )
    )
    result = {
        "schema_version": ORACLE_SCHEMA,
        "status": "complete",
        "purpose": "validation_only_ground_truth_temporal_quantization_reference",
        "test_manifest_read": False,
        "created_unix_s": time.time(),
        "elapsed_s": time.time() - started,
        "contract": _contract(),
        "source": {
            "data_summary": str((data_dir / "summary.json").resolve()),
            "data_summary_sha256": sha256_file(data_dir / "summary.json"),
            "val_episodes": str(artifacts["val_episodes"].resolve()),
            "val_episodes_sha256": sha256_file(artifacts["val_episodes"]),
            "source_protocol_sha256": data_summary["source_protocol_sha256"],
            "source_audit_sha256": data_summary["source_audit_sha256"],
        },
        "by_rate_hz": by_rate,
        "checks": {
            "data_complete": data_summary.get("status") == "complete",
            "test_sealed": data_summary.get("test_manifest_read") is False,
            "all_rates_complete": set(by_rate) == {str(value) for value in RATES_HZ},
            "native_30hz_oracle_perfect": perfect,
            "untouched_30hz_ground_truth_used": True,
            "no_model_selection": True,
        },
    }
    if not all(result["checks"].values()):
        raise RuntimeError(f"Wave-4 rate oracle failed: {result['checks']}")
    _atomic_json(output, result)
    return result


@dataclass(frozen=True)
class SourceRun:
    variant: str
    seed: int
    root: Path
    summary: dict[str, Any]
    resolved: dict[str, Any]
    source_kind: str


def _validated_source(exp_root: Path, variant: str, seed: int) -> SourceRun:
    if variant not in VARIANT_SEEDS or seed not in VARIANT_SEEDS[variant]:
        raise ValueError("Wave-4 audit requires a frozen B0 or D30B seed")
    if variant == "B0":
        root = exp_root / "wave1" / "runs" / "B0" / f"seed_{seed}"
        summary, resolved = wave3b._validated_summary(
            root,
            variant="B0",
            seed=seed,
            enabled_modalities=MODALITIES[variant],
            expected_workers=16,
        )
        source_kind = "wave1_b0"
    elif seed in ORIGINAL_SEEDS:
        root = exp_root / "wave3" / "wave3b" / "runs" / "VT" / f"seed_{seed}"
        summary, resolved = wave3b._validated_summary(
            root,
            variant=wave3b.VARIANT,
            seed=seed,
            enabled_modalities=MODALITIES[variant],
            expected_workers=20,
        )
        source_kind = "wave3b_vt_released_as_d30b"
    else:
        root = exp_root / "wave3" / "wave3c" / "runs" / "D30B" / f"seed_{seed}"
        summary, resolved = wave3c._validated_wave3c_summary(
            root,
            spec=wave3c.CELL_BY_ID["D30B"],
            seed=seed,
        )
        source_kind = "wave3c_d30b_replication"
    return SourceRun(variant, seed, root, summary, resolved, source_kind)


def _dataset(
    manifest: Path,
    statistics_path: Path,
    *,
    enabled_modalities: Sequence[str],
) -> VariantAwareFormal500hDataset:
    rows = wave1._read_jsonl(manifest)
    if any(row.get("schema_version") != WINDOW_SCHEMA for row in rows):
        raise ValueError("Wave-1 validation window manifest schema changed")
    return VariantAwareFormal500hDataset(
        manifest,
        statistics_path,
        expected_statistics_schema=WAVE1_DATA_SCHEMA,
        enabled_modalities=enabled_modalities,
        hdf5_cache_size=8,
    )


@torch.no_grad()
def _predict_validation(
    model: nn.Module,
    dataset: VariantAwareFormal500hDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    precision: str,
    variant: str,
    seed: int,
) -> dict[str, StitchedEpisode]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_formal500h_fast,
    )
    amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    windows: list[WindowPrediction] = []
    model.eval()
    for batch_index, batch in enumerate(loader, start=1):
        batch = wave1._to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            model_output = model(batch)
        atomic = model_output["atomic_logits"].float().detach().cpu().numpy()
        boundary = (
            torch.sigmoid(model_output["boundary_logits"].float())
            .detach()
            .cpu()
            .numpy()
        )
        mask = batch["sequence_mask"].detach().cpu().numpy().astype(bool)
        global_frame = batch["global_frame_60"].detach().cpu().numpy()
        time_ns = batch["time_ns"].detach().cpu().numpy()
        for row_index, episode_id in enumerate(batch["episode_id"]):
            used = mask[row_index]
            windows.append(
                WindowPrediction(
                    episode_id=str(episode_id),
                    global_frame_60=global_frame[row_index, used],
                    atomic_logits=atomic[row_index, used],
                    boundary_probability=boundary[row_index, used],
                    time_ns=time_ns[row_index, used],
                    source_id=f"{episode_id}:{int(global_frame[row_index, used][0])}",
                )
            )
        if batch_index % 200 == 0:
            print(
                json.dumps(
                    {
                        "event": "wave4_validation_inference_progress",
                        "variant": variant,
                        "seed": seed,
                        "batches": batch_index,
                        "windows": len(windows),
                    }
                ),
                flush=True,
            )
    return stitch_window_predictions(
        windows,
        weighting="uniform",
        require_full_coverage=True,
    )


def _prediction_digest(predictions: Mapping[str, StitchedEpisode]) -> str:
    digest = hashlib.sha256()
    for episode_id in sorted(predictions):
        value = predictions[episode_id]
        digest.update(episode_id.encode("utf-8"))
        for array in (
            value.global_frame_60.astype("<i8", copy=False),
            value.atomic_logits.astype("<f4", copy=False),
            value.boundary_probability.astype("<f4", copy=False),
            value.coverage_mask.astype(np.uint8, copy=False),
            value.contribution_count.astype("<i8", copy=False),
            value.weight_sum.astype("<f4", copy=False),
        ):
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(np.ascontiguousarray(array).tobytes())
        if value.time_ns is None:
            digest.update(b"no-time-ns")
        else:
            timestamps = value.time_ns.astype("<i8", copy=False)
            digest.update(np.asarray(timestamps.shape, dtype="<i8").tobytes())
            digest.update(np.ascontiguousarray(timestamps).tobytes())
    return digest.hexdigest()


def persist_predictions(
    path: Path,
    predictions: Mapping[str, StitchedEpisode],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the exact shared episode predictions used by every audit view."""

    episode_ids = sorted(predictions)
    if not episode_ids:
        raise ValueError("cannot persist an empty prediction collection")
    total_ticks = sum(predictions[value].num_ticks for value in episode_ids)
    class_count = predictions[episode_ids[0]].atomic_logits.shape[1]
    if class_count != NUM_CLASSES:
        raise ValueError("prediction class count differs from the frozen ontology")
    if any(predictions[value].time_ns is None for value in episode_ids):
        raise ValueError("Wave-4 prediction artifacts require episode timestamps")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    offsets = np.zeros(len(episode_ids) + 1, dtype=np.int64)
    for index, episode_id in enumerate(episode_ids, start=1):
        offsets[index] = offsets[index - 1] + predictions[episode_id].num_ticks
    chunk_ticks = max(1, min(8192, total_ticks))
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = PREDICTION_SCHEMA
        handle.attrs["complete"] = False
        handle.attrs["prediction_digest_sha256"] = _prediction_digest(predictions)
        handle.attrs["metadata_json"] = json.dumps(
            dict(metadata), ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        string_type = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            "episode_id", data=np.asarray(episode_ids, dtype=object), dtype=string_type
        )
        handle.create_dataset("offset", data=offsets)
        ticks = handle.create_dataset(
            "global_frame_30",
            shape=(total_ticks,),
            dtype="<i8",
            chunks=(chunk_ticks,),
            compression="lzf",
            shuffle=True,
        )
        timestamps = handle.create_dataset(
            "time_ns",
            shape=(total_ticks,),
            dtype="<i8",
            chunks=(chunk_ticks,),
            compression="lzf",
            shuffle=True,
        )
        logits = handle.create_dataset(
            "atomic_logits",
            shape=(total_ticks, class_count),
            dtype="<f4",
            chunks=(chunk_ticks, class_count),
            compression="lzf",
            shuffle=True,
        )
        boundary = handle.create_dataset(
            "boundary_probability",
            shape=(total_ticks,),
            dtype="<f4",
            chunks=(chunk_ticks,),
            compression="lzf",
            shuffle=True,
        )
        coverage = handle.create_dataset(
            "coverage_mask",
            shape=(total_ticks,),
            dtype="u1",
            chunks=(chunk_ticks,),
            compression="lzf",
            shuffle=True,
        )
        contribution = handle.create_dataset(
            "contribution_count",
            shape=(total_ticks,),
            dtype="<i8",
            chunks=(chunk_ticks,),
            compression="lzf",
            shuffle=True,
        )
        weights = handle.create_dataset(
            "weight_sum",
            shape=(total_ticks,),
            dtype="<f4",
            chunks=(chunk_ticks,),
            compression="lzf",
            shuffle=True,
        )
        for index, episode_id in enumerate(episode_ids):
            start, end = int(offsets[index]), int(offsets[index + 1])
            value = predictions[episode_id]
            ticks[start:end] = value.global_frame_60
            timestamps[start:end] = value.time_ns
            logits[start:end] = value.atomic_logits
            boundary[start:end] = value.boundary_probability
            coverage[start:end] = value.coverage_mask.astype(np.uint8)
            contribution[start:end] = value.contribution_count
            weights[start:end] = value.weight_sum
        handle.attrs["complete"] = True
        handle.flush()
    os.replace(temporary, path)
    verified_predictions = load_predictions(path)
    verified_digest = _prediction_digest(verified_predictions)
    expected_digest = _prediction_digest(predictions)
    if verified_digest != expected_digest:
        raise RuntimeError("Wave-4 prediction artifact failed write/read verification")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "prediction_digest_sha256": verified_digest,
        "episodes": len(episode_ids),
        "ticks": total_ticks,
        "classes": class_count,
        "bytes": path.stat().st_size,
    }


def load_predictions(path: Path) -> dict[str, StitchedEpisode]:
    """Load a persisted Wave-4 artifact for independent metric recomputation."""

    output: dict[str, StitchedEpisode] = {}
    with h5py.File(path, "r") as handle:
        schema = handle.attrs.get("schema_version", "")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        if schema != PREDICTION_SCHEMA or not bool(
            handle.attrs.get("complete", False)
        ):
            raise ValueError(f"invalid Wave-4 prediction artifact: {path}")
        required = {
            "episode_id",
            "offset",
            "global_frame_30",
            "time_ns",
            "atomic_logits",
            "boundary_probability",
            "coverage_mask",
            "contribution_count",
            "weight_sum",
        }
        if not required.issubset(handle):
            raise ValueError(
                f"Wave-4 prediction datasets missing: {sorted(required-set(handle))}"
            )
        episode_ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["episode_id"][:]
        ]
        offsets = np.asarray(handle["offset"][:], dtype=np.int64)
        if (
            not episode_ids
            or len(set(episode_ids)) != len(episode_ids)
            or offsets.shape != (len(episode_ids) + 1,)
            or offsets[0] != 0
            or np.any(np.diff(offsets) <= 0)
        ):
            raise ValueError("invalid Wave-4 prediction offsets")
        total_ticks = int(offsets[-1])
        expected_shapes = {
            "global_frame_30": (total_ticks,),
            "time_ns": (total_ticks,),
            "atomic_logits": (total_ticks, NUM_CLASSES),
            "boundary_probability": (total_ticks,),
            "coverage_mask": (total_ticks,),
            "contribution_count": (total_ticks,),
            "weight_sum": (total_ticks,),
        }
        for name, expected_shape in expected_shapes.items():
            if handle[name].shape != expected_shape:
                raise ValueError(
                    f"invalid Wave-4 prediction dataset shape: "
                    f"{name}={handle[name].shape}, expected={expected_shape}"
                )
        for index, episode_id in enumerate(episode_ids):
            selection = slice(int(offsets[index]), int(offsets[index + 1]))
            coverage = np.asarray(handle["coverage_mask"][selection], dtype=np.bool_)
            output[episode_id] = StitchedEpisode(
                episode_id=episode_id,
                global_frame_60=np.asarray(
                    handle["global_frame_30"][selection], dtype=np.int64
                ),
                atomic_logits=np.asarray(
                    handle["atomic_logits"][selection], dtype=np.float32
                ),
                boundary_probability=np.asarray(
                    handle["boundary_probability"][selection], dtype=np.float32
                ),
                coverage_mask=coverage,
                contribution_count=np.asarray(
                    handle["contribution_count"][selection], dtype=np.int64
                ),
                weight_sum=np.asarray(handle["weight_sum"][selection], dtype=np.float32),
                time_ns=np.asarray(handle["time_ns"][selection], dtype=np.int64),
            )
        expected_value = handle.attrs["prediction_digest_sha256"]
        expected = (
            expected_value.decode("utf-8")
            if isinstance(expected_value, bytes)
            else str(expected_value)
        )
    if _prediction_digest(output) != expected:
        raise ValueError("Wave-4 persisted prediction digest mismatch")
    return output


def _evaluate_native(
    predictions: Mapping[str, StitchedEpisode],
    truth: Mapping[str, GroundTruthEpisode],
    decoder: Mapping[str, Any],
) -> Any:
    return evaluate_dataset(
        predictions,
        truth,
        num_actions=NUM_CLASSES,
        segment_iou_thresholds=(0.10, 0.25, 0.50),
        boundary_tolerances_ms=BOUNDARY_TOLERANCES_MS,
        primary_boundary_tolerance_ms=250,
        output_rate_hz=float(MASTER_RATE_HZ),
        boundary_threshold=float(decoder["boundary_threshold"]),
        boundary_min_distance_frames=int(decoder["boundary_min_distance_frames"]),
        min_segment_frames=int(decoder["min_segment_frames"]),
        use_boundary_head=True,
        include_action_changes=True,
        strict=True,
        collect_artifacts=False,
    )


def _decoded_native_views(
    predictions: Mapping[str, StitchedEpisode],
    truth: Mapping[str, GroundTruthEpisode],
    decoder: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[int, str]]]:
    labels_by_episode: dict[str, np.ndarray] = {}
    boundaries_by_episode: dict[str, dict[int, str]] = {}
    for episode_id in sorted(truth):
        prediction = predictions[episode_id]
        target = truth[episode_id]
        if not np.array_equal(prediction.global_frame_60, target.global_frame_60):
            raise ValueError(f"prediction/GT clocks differ: {episode_id}")
        covered = prediction.coverage_mask.copy()
        evaluation_mask, segments, boundary_types = decode_evaluable_prediction(
            prediction.atomic_logits,
            prediction.boundary_probability,
            target,
            covered,
            boundary_threshold=float(decoder["boundary_threshold"]),
            boundary_min_distance_frames=int(
                decoder["boundary_min_distance_frames"]
            ),
            min_segment_frames=int(decoder["min_segment_frames"]),
            use_boundary_head=True,
            include_action_changes=True,
        )
        labels = segments_to_frame_labels(segments, target.num_ticks)
        if np.any(evaluation_mask & (labels < 0)):
            raise RuntimeError(f"decoded labels do not cover evaluation ticks: {episode_id}")
        labels_by_episode[episode_id] = labels
        boundaries_by_episode[episode_id] = dict(boundary_types)
    return labels_by_episode, boundaries_by_episode


def _max_numeric_subset_delta(expected: Any, actual: Any, path: str = "") -> float:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError(f"reproduction type mismatch at {path}")
        missing = set(expected) - set(actual)
        if missing:
            raise ValueError(
                f"reproduction keys missing at {path}: {sorted(missing)[:5]}"
            )
        return max(
            (
                _max_numeric_subset_delta(value, actual[key], f"{path}/{key}")
                for key, value in expected.items()
            ),
            default=0.0,
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"reproduction list mismatch at {path}")
        return max(
            (
                _max_numeric_subset_delta(left, right, f"{path}/{index}")
                for index, (left, right) in enumerate(zip(expected, actual))
            ),
            default=0.0,
        )
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise ValueError(f"reproduction value mismatch at {path}")
        return 0.0
    if isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)):
            raise ValueError(f"reproduction numeric type mismatch at {path}")
        expected_number = float(expected)
        actual_number = float(actual)
        if not math.isfinite(expected_number) or not math.isfinite(actual_number):
            raise ValueError(f"non-finite reproduction metric at {path}")
        return abs(expected_number - actual_number)
    if actual != expected:
        raise ValueError(f"reproduction value mismatch at {path}")
    return 0.0


def _adapted_dr(
    predicted_boundary_types: Mapping[str, Mapping[int, str]],
    truth: Mapping[str, GroundTruthEpisode],
    compact: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    safe = gap_safe_typed_boundary_counts(
        predicted_boundary_types,
        truth,
        evaluation_type="all",
        tolerance_ticks=30,
    )
    frozen = compact["all_boundary"]["1000ms"]
    equal = all(int(safe[key]) == int(frozen[key]) for key in ("tp", "fp", "fn"))
    return {
        "name": "m2r2_style_adapted_dr_all_boundaries@1000ms",
        "official_m2r2_reproduction": False,
        "boundary_type": "all_annotated_action_boundaries",
        "tolerance_ms": 1000,
        "tolerance_ticks_at_30hz": 30,
        "match_scope": "episode_and_contiguous_evaluable_run",
        "frozen_episode_scope_counts": {
            key: frozen[key]
            for key in ("tp", "fp", "fn", "precision", "recall", "f1")
        },
        "matches_frozen_episode_scope_counts": equal,
        **safe,
    }, equal


def audit_checkpoint(
    *,
    exp_root: Path,
    data_dir: Path,
    oracle_path: Path,
    output_dir: Path,
    variant: str,
    seed: int,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    started = time.time()
    if batch_size != 64 or num_workers != 20:
        raise ValueError("released Wave-4 replay runtime is batch=64 workers=20")
    source = _validated_source(exp_root, variant, seed)
    oracle = _read_json(oracle_path)
    if (
        oracle.get("schema_version") != ORACLE_SCHEMA
        or oracle.get("status") != "complete"
        or oracle.get("test_manifest_read") is not False
        or not all((oracle.get("checks") or {}).values())
    ):
        raise ValueError("invalid Wave-4 rate oracle")
    data_summary, artifacts, truth = _data_and_truth(data_dir)
    if oracle["source"]["data_summary_sha256"] != sha256_file(data_dir / "summary.json"):
        raise ValueError("rate oracle belongs to different frozen data")

    checkpoint_path = Path(str(source.summary["best_checkpoint"]))
    checkpoint = wave1._checkpoint_load(checkpoint_path, torch.device("cpu"))
    if checkpoint.get("resolved_config") != source.resolved:
        raise ValueError("checkpoint resolved configuration changed")
    device = torch.device(device_name)
    model = Formal500hModel(source.resolved["model"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    dataset = _dataset(
        artifacts["val_windows"],
        artifacts["train_statistics"],
        enabled_modalities=MODALITIES[variant],
    )
    predictions = _predict_validation(
        model,
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        precision=str(source.resolved["precision"]),
        variant=variant,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    decoder = dict(source.summary["calibrated_validation"]["selected"])
    native_result = _evaluate_native(predictions, truth, decoder)
    native_full = native_result.metrics
    reproduction_delta = _max_numeric_subset_delta(
        source.summary["calibrated_validation"]["metrics"],
        native_full,
    )
    effective_batch_size = batch_size
    batch16_fallback = False
    if reproduction_delta > 1.0e-9 and batch_size != 16:
        print(
            json.dumps(
                {
                    "event": "wave4_exact_reproduction_fallback",
                    "variant": variant,
                    "seed": seed,
                    "requested_batch_size": batch_size,
                    "fallback_batch_size": 16,
                    "first_max_abs": reproduction_delta,
                }
            ),
            flush=True,
        )
        del predictions, native_result
        if device.type == "cuda":
            torch.cuda.empty_cache()
        predictions = _predict_validation(
            model,
            dataset,
            batch_size=16,
            num_workers=num_workers,
            device=device,
            precision=str(source.resolved["precision"]),
            variant=variant,
            seed=seed,
        )
        native_result = _evaluate_native(predictions, truth, decoder)
        native_full = native_result.metrics
        reproduction_delta = _max_numeric_subset_delta(
            source.summary["calibrated_validation"]["metrics"],
            native_full,
        )
        effective_batch_size = 16
        batch16_fallback = True
    prediction_sha256 = _prediction_digest(predictions)
    prediction_artifact = persist_predictions(
        output_dir / "stitched_predictions.h5",
        predictions,
        metadata={
            "variant": variant,
            "seed": seed,
            "source_kind": source.source_kind,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "validation_manifest_sha256": sha256_file(artifacts["val_windows"]),
            "fixed_decoder": decoder,
            "requested_batch_size": batch_size,
            "effective_batch_size": effective_batch_size,
            "test_manifest_read": False,
        },
    )
    native = _compact_metrics(native_full)
    decoded, native_boundary_types = _decoded_native_views(
        predictions,
        truth,
        decoder,
    )
    adapted_dr, _ = _adapted_dr(
        native_boundary_types,
        truth,
        native,
    )
    episode_metrics_path = output_dir / "native_30hz_episode_metrics.jsonl"
    _atomic_jsonl(
        episode_metrics_path,
        [
            {
                "episode_id": row["episode_id"],
                "metrics": _compact_metrics(row),
            }
            for row in native_result.episode_metrics
        ],
    )

    rate_sweep: dict[str, Any] = {}
    native_segment_count = native["semantic_segment_count"]
    native_semantic_reference = {
        key: value for key, value in native.items() if key != "all_boundary"
    }
    native_rate_equal = False
    for rate in RATES_HZ:
        labels = {
            episode_id: causal_sample_and_hold(
                decoded[episode_id],
                target.label_mask,
                target_rate_hz=rate,
            )
            for episode_id, target in truth.items()
        }
        metrics = _semantic_metrics(labels, truth)
        rate_sweep[str(rate)] = {
            "rate_hz": rate,
            "stride_ticks": STRIDE_BY_RATE[rate],
            "metrics": metrics,
            "semantic_segments_removed_vs_30hz": (
                native_segment_count - metrics["semantic_segment_count"]
            ),
            "m2r2_style_dr_reportable": False,
            "m2r2_style_dr_not_reportable_reason": (
                "semantic output-rate proxy cannot preserve same-label "
                "instance boundaries"
            ),
        }
        if rate == MASTER_RATE_HZ:
            native_rate_equal = (
                _max_numeric_subset_delta(native_semantic_reference, metrics)
                <= 1.0e-12
            )
        print(
            json.dumps(
                {
                    "event": "wave4_rate_complete",
                    "variant": variant,
                    "seed": seed,
                    "rate_hz": rate,
                    "collapsed_f1@50": metrics["collapsed"]["f1@50"],
                    "class_change_boundary_f1@1000ms": metrics[
                        "class_change_boundary"
                    ]["1000ms"]["f1"],
                }
            ),
            flush=True,
        )

    checks = {
        "source_lineage_valid": True,
        "checkpoint_hash_valid": sha256_file(checkpoint_path)
        == source.summary["best_checkpoint_sha256"],
        "data_lineage_valid": data_summary.get("status") == "complete",
        "oracle_lineage_valid": oracle["source"]["data_summary_sha256"]
        == sha256_file(data_dir / "summary.json"),
        "predictions_persisted": Path(prediction_artifact["path"]).is_file()
        and prediction_artifact["prediction_digest_sha256"] == prediction_sha256,
        "episode_metrics_persisted": episode_metrics_path.is_file(),
        "decoder_frozen": decoder
        == source.summary["calibrated_validation"]["selected"],
        "clean_native_reproduction": reproduction_delta <= 1.0e-9,
        "native_30hz_rate_operator_identity": native_rate_equal,
        "adapted_dr_all_boundary_contract": (
            adapted_dr["boundary_type"] == "all_annotated_action_boundaries"
            and math.isfinite(float(adapted_dr["f1"]))
        ),
        "rate_sweep_does_not_mislabel_class_change_as_m2r2_dr": all(
            row["m2r2_style_dr_reportable"] is False
            for row in rate_sweep.values()
        ),
        "all_rates_complete": set(rate_sweep) == {str(value) for value in RATES_HZ},
        "test_sealed": data_summary.get("test_manifest_read") is False
        and source.summary.get("test_manifest_read") is False,
        "no_decoder_recalibration": True,
        "no_model_selection": True,
    }
    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "complete",
        "purpose": "validation_only_metric_and_rate_sensitivity_no_model_selection",
        "test_manifest_read": False,
        "created_unix_s": time.time(),
        "elapsed_s": time.time() - started,
        "variant": variant,
        "seed": seed,
        "source_kind": source.source_kind,
        "source_lineage": {
            "run_root": str(source.root.resolve()),
            "summary": str((source.root / "summary.json").resolve()),
            "summary_sha256": sha256_file(source.root / "summary.json"),
            "resolved_config": str(Path(source.summary["resolved_config"]).resolve()),
            "resolved_config_sha256": source.summary["resolved_config_sha256"],
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "prediction_digest_sha256": prediction_sha256,
            "prediction_artifact": prediction_artifact,
            "native_episode_metrics": {
                "path": str(episode_metrics_path.resolve()),
                "sha256": sha256_file(episode_metrics_path),
                "rows": len(native_result.episode_metrics),
            },
            "data_summary_sha256": sha256_file(data_dir / "summary.json"),
            "validation_manifest_sha256": sha256_file(artifacts["val_windows"]),
            "rate_oracle": str(oracle_path.resolve()),
            "rate_oracle_sha256": sha256_file(oracle_path),
            "fixed_decoder": decoder,
        },
        "runtime": {
            "requested_batch_size_per_gpu": batch_size,
            "effective_batch_size_per_gpu": effective_batch_size,
            "batch16_exact_reproduction_fallback": batch16_fallback,
            "workers_per_job": num_workers,
            "precision": source.resolved["precision"],
            "hdf5_cache_size": 8,
            "prefetch_factor": 2,
        },
        "contract": _contract(),
        "native_30hz": {
            "metrics": native,
            "adapted_dr": adapted_dr,
            "source_metric_reproduction_max_abs": reproduction_delta,
        },
        "prediction_rate_sweep": rate_sweep,
        "oracle_reference": {
            "path": str(oracle_path.resolve()),
            "sha256": sha256_file(oracle_path),
        },
        "checks": checks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Wave-4 checkpoint audit failed: {checks}")
    summary_path = output_dir / "summary.json"
    _atomic_json(summary_path, result)
    complete = {
        "schema_version": AUDIT_COMPLETE_SCHEMA,
        "status": "complete",
        "variant": variant,
        "seed": seed,
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "prediction_artifact": prediction_artifact["path"],
        "prediction_artifact_sha256": prediction_artifact["sha256"],
        "native_episode_metrics": str(episode_metrics_path.resolve()),
        "native_episode_metrics_sha256": sha256_file(episode_metrics_path),
        "test_manifest_read": False,
        "no_model_selection": True,
    }
    _atomic_json(output_dir / "COMPLETE.json", complete)
    return result


def validate_audit(
    output_dir: Path,
    *,
    variant: str,
    seed: int,
    verify_prediction_content: bool = True,
) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    complete_path = output_dir / "COMPLETE.json"
    summary = _read_json(summary_path)
    complete = _read_json(complete_path)
    prediction_path = Path(str(complete.get("prediction_artifact", "")))
    episode_metrics_path = Path(str(complete.get("native_episode_metrics", "")))
    prediction_valid = False
    if prediction_path.is_file():
        with h5py.File(prediction_path, "r") as handle:
            schema = handle.attrs.get("schema_version", "")
            if isinstance(schema, bytes):
                schema = schema.decode("utf-8")
            prediction_valid = (
                schema == PREDICTION_SCHEMA
                and bool(handle.attrs.get("complete", False))
                and str(handle.attrs.get("prediction_digest_sha256", ""))
                == summary.get("source_lineage", {}).get(
                    "prediction_digest_sha256"
                )
            )
        if prediction_valid and verify_prediction_content:
            loaded = load_predictions(prediction_path)
            prediction_valid = (
                _prediction_digest(loaded)
                == summary.get("source_lineage", {}).get(
                    "prediction_digest_sha256"
                )
            )
    prediction_file_hash_valid = prediction_path.is_file() and (
        not verify_prediction_content
        or complete.get("prediction_artifact_sha256")
        == sha256_file(prediction_path)
    )
    source_lineage = summary.get("source_lineage") or {}
    source_summary_path = Path(str(source_lineage.get("summary", "")))
    source_resolved_path = Path(str(source_lineage.get("resolved_config", "")))
    checkpoint_path = Path(str(source_lineage.get("checkpoint", "")))
    oracle_path = Path(str(source_lineage.get("rate_oracle", "")))
    checks = {
        "schema": summary.get("schema_version") == AUDIT_SCHEMA,
        "complete_schema": complete.get("schema_version")
        == AUDIT_COMPLETE_SCHEMA,
        "complete": summary.get("status") == complete.get("status") == "complete",
        "identity": summary.get("variant") == complete.get("variant") == variant
        and int(summary.get("seed", -1)) == int(complete.get("seed", -2)) == seed,
        "hash": complete.get("summary_sha256") == sha256_file(summary_path),
        "checks": bool(summary.get("checks"))
        and all(summary["checks"].values()),
        "summary_path": Path(str(complete.get("summary", ""))).resolve()
        == summary_path.resolve(),
        "prediction_artifact": prediction_valid and prediction_file_hash_valid,
        "episode_metrics": episode_metrics_path.is_file()
        and complete.get("native_episode_metrics_sha256")
        == sha256_file(episode_metrics_path),
        "test_sealed": summary.get("test_manifest_read") is False
        and complete.get("test_manifest_read") is False,
        "no_selection": complete.get("no_model_selection") is True,
        "contract": summary.get("contract") == _contract(),
        "source_summary": source_summary_path.is_file()
        and source_lineage.get("summary_sha256")
        == sha256_file(source_summary_path),
        "source_resolved_config": source_resolved_path.is_file()
        and source_lineage.get("resolved_config_sha256")
        == sha256_file(source_resolved_path),
        "source_checkpoint": checkpoint_path.is_file()
        and source_lineage.get("checkpoint_sha256")
        == sha256_file(checkpoint_path),
        "oracle": oracle_path.is_file()
        and source_lineage.get("rate_oracle_sha256")
        == sha256_file(oracle_path)
        and (summary.get("oracle_reference") or {}).get("sha256")
        == sha256_file(oracle_path),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid Wave-4 checkpoint audit {output_dir}: {checks}")
    return summary


def _metric_paths() -> dict[str, tuple[str, ...]]:
    return {
        "frame_accuracy": ("native_30hz", "metrics", "frame_accuracy"),
        "tick_macro_f1": ("native_30hz", "metrics", "tick_macro_f1"),
        "collapsed_edit": ("native_30hz", "metrics", "collapsed", "edit"),
        "collapsed_f1@10": ("native_30hz", "metrics", "collapsed", "f1@10"),
        "collapsed_f1@25": ("native_30hz", "metrics", "collapsed", "f1@25"),
        "collapsed_f1@50": ("native_30hz", "metrics", "collapsed", "f1@50"),
        "class_change_bf1@250ms": (
            "native_30hz",
            "metrics",
            "class_change_boundary",
            "250ms",
            "f1",
        ),
        "class_change_bf1@500ms": (
            "native_30hz",
            "metrics",
            "class_change_boundary",
            "500ms",
            "f1",
        ),
        "class_change_bf1@1000ms": (
            "native_30hz",
            "metrics",
            "class_change_boundary",
            "1000ms",
            "f1",
        ),
        "m2r2_style_adapted_dr_all_boundaries@1000ms": (
            "native_30hz",
            "adapted_dr",
            "f1",
        ),
    }


def _get(value: Mapping[str, Any], path: Sequence[str]) -> float:
    current: Any = value
    for key in path:
        current = current[key]
    return float(current)


def _mean_std(values: Sequence[float]) -> dict[str, Any]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "values": [float(value) for value in values],
    }


def _path_points_to(value: Any, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return Path(value).resolve() == expected.resolve()


def _validated_parent(
    name: str,
    aggregate_path: Path,
    complete_path: Path,
    *,
    wave1_sha256: str | None = None,
    wave3b_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregate = _read_json(aggregate_path)
    complete = _read_json(complete_path)
    controls = aggregate.get("controls") or {}
    expected_aggregate_schema: str
    expected_complete_schema: str
    complete_path_key: str
    specific: dict[str, bool]
    if name == "wave1":
        expected_aggregate_schema = wave1.AGGREGATE_SCHEMA
        # Wave-1 COMPLETE embeds the aggregate object and therefore retains
        # the aggregate schema instead of using a separate completion schema.
        expected_complete_schema = wave1.AGGREGATE_SCHEMA
        complete_path_key = "aggregate_path"
        specific = {
            "frozen_seeds": aggregate.get("seeds") == list(ORIGINAL_SEEDS),
            "b0_present": "B0" in (aggregate.get("variants") or []),
            "paired_initialization": controls.get(
                "paired_initialization_within_seed"
            )
            is True,
            "full_episode_validation": controls.get("full_episode_validation")
            is True,
            "control_test_sealed": controls.get("test_sealed") is True,
        }
    elif name == "wave3b_vt":
        expected_aggregate_schema = wave3b.AGGREGATE_SCHEMA
        expected_complete_schema = wave3b.TRAINING_SCHEMA
        complete_path_key = "aggregate"
        specific = {
            "frozen_seeds": aggregate.get("seeds") == list(ORIGINAL_SEEDS),
            "variants": aggregate.get("variants") == ["B0", "VT"],
            "controls": bool(controls)
            and all(value is True for value in controls.values()),
            "selection_deferred": aggregate.get("test_selection_deferred") is True
            and complete.get("test_selection_deferred") is True,
            "wave1_hash_chain": bool(wave1_sha256)
            and aggregate.get("wave1_aggregate_sha256") == wave1_sha256,
        }
    elif name == "wave3c":
        expected_aggregate_schema = wave3c.AGGREGATE_SCHEMA
        expected_complete_schema = WAVE3C_COMPLETE_SCHEMA
        complete_path_key = "aggregate"
        cell_ids = {
            str(row.get("cell"))
            for row in (aggregate.get("cells") or [])
            if isinstance(row, Mapping)
        }
        parent_rows = aggregate.get("parents") or {}
        specific = {
            "all_seeds": aggregate.get("seeds") == list(D30B_SEEDS),
            "original_seeds": aggregate.get("original_seeds")
            == list(ORIGINAL_SEEDS),
            "replication_seeds": aggregate.get("replication_seeds")
            == list(REPLICATION_SEEDS),
            "cells": cell_ids == set(wave3c.CELL_BY_ID),
            "controls": bool(controls)
            and all(value is True for value in controls.values()),
            "selection_deferred": aggregate.get("test_selection_deferred") is True
            and complete.get("test_selection_deferred") is True,
            "wave1_hash_chain": bool(wave1_sha256)
            and (parent_rows.get("wave1_aggregate") or {}).get("sha256")
            == wave1_sha256,
            "wave3b_hash_chain": bool(wave3b_sha256)
            and (parent_rows.get("wave3b_aggregate") or {}).get("sha256")
            == wave3b_sha256,
        }
    else:
        raise ValueError(f"unknown Wave-4 parent: {name}")
    checks = {
        "aggregate_schema": aggregate.get("schema_version")
        == expected_aggregate_schema,
        "complete_schema": complete.get("schema_version")
        == expected_complete_schema,
        "complete": aggregate.get("status") == "complete"
        and complete.get("status") == "complete",
        "hash": complete.get("aggregate_sha256") == sha256_file(aggregate_path),
        "complete_points_to_aggregate": _path_points_to(
            complete.get(complete_path_key),
            aggregate_path,
        ),
        "test_sealed": aggregate.get("test_manifest_read") is False
        and complete.get("test_manifest_read") is False,
        **specific,
    }
    if not all(checks.values()):
        raise ValueError(
            f"invalid frozen Wave-4 parent {aggregate_path}: {checks}"
        )
    return aggregate, complete


def validate_parents(
    *,
    wave1_aggregate: Path,
    wave1_complete: Path,
    wave3b_aggregate: Path,
    wave3b_complete: Path,
    wave3c_aggregate: Path,
    wave3c_complete: Path,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Validate every frozen parent and the Wave-1 -> 3B -> 3C hash chain."""

    wave1_value = _validated_parent(
        "wave1",
        wave1_aggregate,
        wave1_complete,
    )
    wave1_hash = sha256_file(wave1_aggregate)
    wave3b_value = _validated_parent(
        "wave3b_vt",
        wave3b_aggregate,
        wave3b_complete,
        wave1_sha256=wave1_hash,
    )
    wave3b_hash = sha256_file(wave3b_aggregate)
    wave3c_value = _validated_parent(
        "wave3c",
        wave3c_aggregate,
        wave3c_complete,
        wave1_sha256=wave1_hash,
        wave3b_sha256=wave3b_hash,
    )
    return {
        "wave1": wave1_value,
        "wave3b_vt": wave3b_value,
        "wave3c": wave3c_value,
    }


def _audit_matches_oracle(
    row: Mapping[str, Any],
    *,
    oracle_sha256: str,
    data_summary_sha256: str,
) -> bool:
    return (
        row.get("contract") == _contract()
        and row.get("test_manifest_read") is False
        and (row.get("source_lineage") or {}).get("rate_oracle_sha256")
        == oracle_sha256
        and (row.get("oracle_reference") or {}).get("sha256")
        == oracle_sha256
        and (row.get("source_lineage") or {}).get("data_summary_sha256")
        == data_summary_sha256
    )


def aggregate_audits(
    *,
    runs_root: Path,
    oracle_path: Path,
    wave1_aggregate: Path,
    wave1_complete: Path,
    wave3b_aggregate: Path,
    wave3b_complete: Path,
    wave3c_aggregate: Path,
    wave3c_complete: Path,
    output: Path,
    complete_output: Path,
) -> dict[str, Any]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for variant, seeds in VARIANT_SEEDS.items():
        for seed in seeds:
            row = validate_audit(
                runs_root / variant / f"seed_{seed}",
                variant=variant,
                seed=seed,
            )
            rows[(variant, seed)] = row
    oracle = _read_json(oracle_path)
    if (
        oracle.get("schema_version") != ORACLE_SCHEMA
        or oracle.get("status") != "complete"
        or oracle.get("test_manifest_read") is not False
        or not all((oracle.get("checks") or {}).values())
    ):
        raise ValueError("invalid Wave-4 rate oracle during aggregation")
    oracle_hash = sha256_file(oracle_path)
    oracle_data_hash = (oracle.get("source") or {}).get("data_summary_sha256")
    row_lineage_checks = {
        f"{variant}_{seed}": _audit_matches_oracle(
            row,
            oracle_sha256=oracle_hash,
            data_summary_sha256=str(oracle_data_hash),
        )
        for (variant, seed), row in rows.items()
    }
    if not all(row_lineage_checks.values()):
        raise ValueError(
            f"Wave-4 audit/oracle lineage mismatch: {row_lineage_checks}"
        )

    parent_inputs = {
        "wave1": (wave1_aggregate, wave1_complete),
        "wave3b_vt": (wave3b_aggregate, wave3b_complete),
        "wave3c": (wave3c_aggregate, wave3c_complete),
    }
    parent_values = validate_parents(
        wave1_aggregate=wave1_aggregate,
        wave1_complete=wave1_complete,
        wave3b_aggregate=wave3b_aggregate,
        wave3b_complete=wave3b_complete,
        wave3c_aggregate=wave3c_aggregate,
        wave3c_complete=wave3c_complete,
    )

    paths = _metric_paths()
    aggregate: dict[str, Any] = {}
    reporting_blocks = {
        "B0_original_three": ("B0", ORIGINAL_SEEDS),
        "D30B_original_three": ("D30B", ORIGINAL_SEEDS),
        "D30B_replication_four": ("D30B", REPLICATION_SEEDS),
        "D30B_all_seven": ("D30B", D30B_SEEDS),
    }
    for block, (variant, seeds) in reporting_blocks.items():
        aggregate[block] = {
            name: _mean_std([_get(rows[(variant, seed)], path) for seed in seeds])
            for name, path in paths.items()
        }
    paired: dict[str, Any] = {}
    for name, path in paths.items():
        values = [
            _get(rows[("D30B", seed)], path) - _get(rows[("B0", seed)], path)
            for seed in ORIGINAL_SEEDS
        ]
        paired[name] = {**_mean_std(values), "wins": sum(value > 0 for value in values)}

    rate_aggregate: dict[str, Any] = {}
    for block, (variant, seeds) in reporting_blocks.items():
        rate_aggregate[block] = {}
        for rate in RATES_HZ:
            prefix = ("prediction_rate_sweep", str(rate), "metrics")
            rate_paths = {
                "frame_accuracy": (*prefix, "frame_accuracy"),
                "tick_macro_f1": (*prefix, "tick_macro_f1"),
                "collapsed_edit": (*prefix, "collapsed", "edit"),
                "collapsed_f1@50": (*prefix, "collapsed", "f1@50"),
                "class_change_bf1@250ms": (
                    *prefix,
                    "class_change_boundary",
                    "250ms",
                    "f1",
                ),
                "class_change_bf1@1000ms": (
                    *prefix,
                    "class_change_boundary",
                    "1000ms",
                    "f1",
                ),
            }
            rate_aggregate[block][str(rate)] = {
                name: _mean_std(
                    [_get(rows[(variant, seed)], path) for seed in seeds]
                )
                for name, path in rate_paths.items()
            }

    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "complete",
        "purpose": "validation_only_common_metric_and_rate_audit_no_model_selection",
        "test_manifest_read": False,
        "created_unix_s": time.time(),
        "jobs": {"B0": 3, "D30B": 7, "total": 10},
        "seeds": {
            "B0": list(ORIGINAL_SEEDS),
            "D30B": list(D30B_SEEDS),
            "paired": list(ORIGINAL_SEEDS),
        },
        "contract": _contract(),
        "parents": {
            name: {
                "aggregate": str(aggregate_path.resolve()),
                "aggregate_sha256": sha256_file(aggregate_path),
                "complete": str(complete_path.resolve()),
                "complete_sha256": sha256_file(complete_path),
                "status": parent_values[name][0]["status"],
                "test_manifest_read": parent_values[name][0][
                    "test_manifest_read"
                ],
            }
            for name, (aggregate_path, complete_path) in parent_inputs.items()
        },
        "audits": {
            variant: {
                str(seed): {
                    "summary": str(
                        (runs_root / variant / f"seed_{seed}" / "summary.json").resolve()
                    ),
                    "summary_sha256": sha256_file(
                        runs_root / variant / f"seed_{seed}" / "summary.json"
                    ),
                    "complete": str(
                        (runs_root / variant / f"seed_{seed}" / "COMPLETE.json").resolve()
                    ),
                    "complete_sha256": sha256_file(
                        runs_root / variant / f"seed_{seed}" / "COMPLETE.json"
                    ),
                    "prediction_artifact_sha256": (
                        rows[(variant, seed)]["source_lineage"][
                            "prediction_artifact"
                        ]["sha256"]
                    ),
                }
                for seed in seeds
            }
            for variant, seeds in VARIANT_SEEDS.items()
        },
        "oracle": {
            "path": str(oracle_path.resolve()),
            "sha256": oracle_hash,
            "by_rate_hz": oracle["by_rate_hz"],
        },
        "aggregate": aggregate,
        "paired_D30B_minus_B0_original_three_seeds": paired,
        "prediction_rate_sweep": rate_aggregate,
        "decision": "reporting_audit_complete_no_model_selection",
        "next_gate": "wave4_mstcn_asrf_diffact_and_m2r2_style_adapted_heads",
        "model_selection_performed": False,
        "decoder_recalibrated": False,
        "actual_low_rate_models_trained": False,
        "checks": {
            "all_audits_complete": len(rows) == 10,
            "all_audit_checks_pass": all(
                all(row["checks"].values()) for row in rows.values()
            ),
            "oracle_complete": oracle.get("status") == "complete",
            "oracle_30hz_perfect": oracle["checks"]["native_30hz_oracle_perfect"],
            "parents_valid": len(parent_values) == 3,
            "audit_oracle_contract_lineage": all(row_lineage_checks.values()),
            "test_sealed": all(row.get("test_manifest_read") is False for row in rows.values())
            and oracle.get("test_manifest_read") is False,
            "no_model_selection": True,
        },
    }
    if not all(result["checks"].values()):
        raise RuntimeError(f"Wave-4 aggregate failed: {result['checks']}")
    _atomic_json(output, result)
    complete = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "complete",
        "aggregate": str(output.resolve()),
        "aggregate_sha256": sha256_file(output),
        "oracle": str(oracle_path.resolve()),
        "oracle_sha256": sha256_file(oracle_path),
        "test_manifest_read": False,
        "model_selection_performed": False,
        "next_gate": result["next_gate"],
    }
    _atomic_json(complete_output, complete)
    return result


def validate_final(
    *,
    aggregate_path: Path,
    complete_path: Path,
    oracle_path: Path,
) -> dict[str, Any]:
    """Fast fail-closed validation for status reads and completed-run reuse."""

    aggregate = _read_json(aggregate_path)
    complete = _read_json(complete_path)
    oracle = _read_json(oracle_path)
    parents = aggregate.get("parents") or {}
    expected_parent_names = {"wave1", "wave3b_vt", "wave3c"}
    parent_paths_valid = set(parents) == expected_parent_names
    if parent_paths_valid:
        try:
            validate_parents(
                wave1_aggregate=Path(parents["wave1"]["aggregate"]),
                wave1_complete=Path(parents["wave1"]["complete"]),
                wave3b_aggregate=Path(parents["wave3b_vt"]["aggregate"]),
                wave3b_complete=Path(parents["wave3b_vt"]["complete"]),
                wave3c_aggregate=Path(parents["wave3c"]["aggregate"]),
                wave3c_complete=Path(parents["wave3c"]["complete"]),
            )
        except (KeyError, OSError, ValueError):
            parent_paths_valid = False
    audit_rows = [
        row
        for by_seed in (aggregate.get("audits") or {}).values()
        for row in by_seed.values()
        if isinstance(row, Mapping)
    ]
    audit_hashes_valid = len(audit_rows) == 10
    if audit_hashes_valid:
        for row in audit_rows:
            summary_path = Path(str(row.get("summary", "")))
            audit_complete_path = Path(str(row.get("complete", "")))
            if (
                not summary_path.is_file()
                or not audit_complete_path.is_file()
                or row.get("summary_sha256") != sha256_file(summary_path)
                or row.get("complete_sha256") != sha256_file(audit_complete_path)
            ):
                audit_hashes_valid = False
                break
    checks = {
        "aggregate_schema": aggregate.get("schema_version") == AGGREGATE_SCHEMA,
        "complete_schema": complete.get("schema_version") == COMPLETE_SCHEMA,
        "oracle_schema": oracle.get("schema_version") == ORACLE_SCHEMA,
        "status": aggregate.get("status") == complete.get("status")
        == oracle.get("status")
        == "complete",
        "aggregate_hash": complete.get("aggregate_sha256")
        == sha256_file(aggregate_path),
        "aggregate_path": _path_points_to(
            complete.get("aggregate"),
            aggregate_path,
        ),
        "oracle_hash": complete.get("oracle_sha256") == sha256_file(oracle_path)
        and (aggregate.get("oracle") or {}).get("sha256")
        == sha256_file(oracle_path),
        "oracle_path": _path_points_to(complete.get("oracle"), oracle_path),
        "checks": bool(aggregate.get("checks"))
        and all(aggregate["checks"].values())
        and bool(oracle.get("checks"))
        and all(oracle["checks"].values()),
        "contract": aggregate.get("contract") == _contract(),
        "parents": parent_paths_valid,
        "audit_hashes": audit_hashes_valid,
        "test_sealed": aggregate.get("test_manifest_read") is False
        and complete.get("test_manifest_read") is False
        and oracle.get("test_manifest_read") is False,
        "no_selection": aggregate.get("model_selection_performed") is False
        and complete.get("model_selection_performed") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid completed Wave-4 metric audit: {checks}")
    return {"status": "complete", "checks": checks}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    oracle = commands.add_parser("oracle")
    oracle.add_argument("--data-dir", type=Path, required=True)
    oracle.add_argument("--output", type=Path, required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("--exp-root", type=Path, required=True)
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--oracle", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--variant", choices=tuple(VARIANT_SEEDS), required=True)
    audit.add_argument("--seed", type=int, required=True)
    audit.add_argument("--device", default="cuda:0")
    audit.add_argument("--batch-size", type=int, default=64)
    audit.add_argument("--num-workers", type=int, default=20)

    validate = commands.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--variant", choices=tuple(VARIANT_SEEDS), required=True)
    validate.add_argument("--seed", type=int, required=True)
    validate.add_argument("--metadata-only", action="store_true")

    parents = commands.add_parser("validate-parents")
    parents.add_argument("--wave1-aggregate", type=Path, required=True)
    parents.add_argument("--wave1-complete", type=Path, required=True)
    parents.add_argument("--wave3b-aggregate", type=Path, required=True)
    parents.add_argument("--wave3b-complete", type=Path, required=True)
    parents.add_argument("--wave3c-aggregate", type=Path, required=True)
    parents.add_argument("--wave3c-complete", type=Path, required=True)

    final = commands.add_parser("validate-final")
    final.add_argument("--aggregate", type=Path, required=True)
    final.add_argument("--complete", type=Path, required=True)
    final.add_argument("--oracle", type=Path, required=True)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--runs-root", type=Path, required=True)
    aggregate.add_argument("--oracle", type=Path, required=True)
    aggregate.add_argument("--wave1-aggregate", type=Path, required=True)
    aggregate.add_argument("--wave1-complete", type=Path, required=True)
    aggregate.add_argument("--wave3b-aggregate", type=Path, required=True)
    aggregate.add_argument("--wave3b-complete", type=Path, required=True)
    aggregate.add_argument("--wave3c-aggregate", type=Path, required=True)
    aggregate.add_argument("--wave3c-complete", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--complete-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "oracle":
        build_rate_oracle(data_dir=arguments.data_dir, output=arguments.output)
    elif arguments.command == "audit":
        audit_checkpoint(
            exp_root=arguments.exp_root,
            data_dir=arguments.data_dir,
            oracle_path=arguments.oracle,
            output_dir=arguments.output_dir,
            variant=arguments.variant,
            seed=arguments.seed,
            device_name=arguments.device,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
        )
    elif arguments.command == "validate":
        validate_audit(
            arguments.output_dir,
            variant=arguments.variant,
            seed=arguments.seed,
            verify_prediction_content=not arguments.metadata_only,
        )
    elif arguments.command == "validate-parents":
        validate_parents(
            wave1_aggregate=arguments.wave1_aggregate,
            wave1_complete=arguments.wave1_complete,
            wave3b_aggregate=arguments.wave3b_aggregate,
            wave3b_complete=arguments.wave3b_complete,
            wave3c_aggregate=arguments.wave3c_aggregate,
            wave3c_complete=arguments.wave3c_complete,
        )
        print("FORMAL500H WAVE-4 PARENT CHAIN: PASS", flush=True)
    elif arguments.command == "validate-final":
        validate_final(
            aggregate_path=arguments.aggregate,
            complete_path=arguments.complete,
            oracle_path=arguments.oracle,
        )
        print("FORMAL500H WAVE-4 FINAL SEAL: PASS", flush=True)
    elif arguments.command == "aggregate":
        aggregate_audits(
            runs_root=arguments.runs_root,
            oracle_path=arguments.oracle,
            wave1_aggregate=arguments.wave1_aggregate,
            wave1_complete=arguments.wave1_complete,
            wave3b_aggregate=arguments.wave3b_aggregate,
            wave3b_complete=arguments.wave3b_complete,
            wave3c_aggregate=arguments.wave3c_aggregate,
            wave3c_complete=arguments.wave3c_complete,
            output=arguments.output,
            complete_output=arguments.complete_output,
        )
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
