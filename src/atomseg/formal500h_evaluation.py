"""Formal500h full-episode evaluation core.

This module contains only the stitch/decode/matching metrics used by the
frozen 26-class Formal500h protocol and has no external dataset reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from .inference import (
    BOUNDARY_CLASS_CHANGE,
    BOUNDARY_SAME_LABEL,
    Segment,
    StitchedEpisode,
    boundary_aware_decode,
    collapse_adjacent_same_label,
    decoded_boundary_types,
    segments_to_frame_labels,
)


BOUNDARY_NONE = 0
BOUNDARY_CLASS_CHANGE_ID = 1
BOUNDARY_SAME_LABEL_ID = 2
BOUNDARY_ID_TO_NAME = {
    BOUNDARY_CLASS_CHANGE_ID: BOUNDARY_CLASS_CHANGE,
    BOUNDARY_SAME_LABEL_ID: BOUNDARY_SAME_LABEL,
}
EVALUATION_TYPES = ("all", BOUNDARY_CLASS_CHANGE, BOUNDARY_SAME_LABEL)


def _array(value: np.ndarray | Sequence[Any], dtype: np.dtype) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _valid_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    indices = np.flatnonzero(mask)
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


@dataclass
class GroundTruthEpisode:
    episode_id: str
    global_frame_60: np.ndarray
    atomic_target: np.ndarray
    atomic_instance_id: np.ndarray
    boundary_hard: np.ndarray
    boundary_type: np.ndarray
    label_mask: np.ndarray
    boundary_eval_mask: np.ndarray
    time_ns: np.ndarray | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.global_frame_60 = _array(self.global_frame_60, np.int64)
        self.atomic_target = _array(self.atomic_target, np.int64)
        self.atomic_instance_id = _array(self.atomic_instance_id, np.int64)
        self.boundary_hard = _array(self.boundary_hard, np.bool_)
        self.boundary_type = _array(self.boundary_type, np.int8)
        self.label_mask = _array(self.label_mask, np.bool_)
        self.boundary_eval_mask = _array(self.boundary_eval_mask, np.bool_)
        if self.global_frame_60.ndim != 1 or self.global_frame_60.size == 0:
            raise ValueError("global_frame_60 must be a non-empty one-dimensional array")
        steps = self.global_frame_60.size
        for name in (
            "atomic_target",
            "atomic_instance_id",
            "boundary_hard",
            "boundary_type",
            "label_mask",
            "boundary_eval_mask",
        ):
            if getattr(self, name).shape != (steps,):
                raise ValueError(f"{name} must be [T]")
        if np.any(np.diff(self.global_frame_60) <= 0):
            raise ValueError("global_frame_60 must be strictly increasing")
        if self.time_ns is not None:
            self.time_ns = _array(self.time_ns, np.int64)
            if self.time_ns.shape != (steps,) or np.any(np.diff(self.time_ns) <= 0):
                raise ValueError("time_ns must be strictly increasing [T]")

    @property
    def num_ticks(self) -> int:
        return int(self.global_frame_60.size)


@dataclass(frozen=True)
class BoundaryMatchResult:
    predicted: Tuple[int, ...]
    target: Tuple[int, ...]
    pairs: Tuple[Tuple[int, int], ...]
    tolerance: int

    @property
    def true_positive(self) -> int:
        return len(self.pairs)

    @property
    def false_positive(self) -> int:
        return len(self.predicted) - len(self.pairs)

    @property
    def false_negative(self) -> int:
        return len(self.target) - len(self.pairs)

    @property
    def signed_errors(self) -> Tuple[int, ...]:
        return tuple(prediction - truth for prediction, truth in self.pairs)


@dataclass(frozen=True)
class SegmentMatchResult:
    pairs: Tuple[Tuple[int, int, float], ...]
    predicted_count: int
    target_count: int
    threshold: float

    @property
    def true_positive(self) -> int:
        return len(self.pairs)

    @property
    def false_positive(self) -> int:
        return self.predicted_count - len(self.pairs)

    @property
    def false_negative(self) -> int:
        return self.target_count - len(self.pairs)


@dataclass
class EpisodeEvaluation:
    episode_id: str
    metrics: Dict[str, Any]
    confusion: np.ndarray
    segment_counts: Dict[Tuple[str, float], Tuple[int, int, int]]
    boundary_counts: Dict[Tuple[str, int], Tuple[int, int, int]]
    gt_type_any_prediction_counts: Dict[Tuple[str, int], Tuple[int, int]]
    raw_edit: float
    collapsed_edit: float
    raw_count_error: float
    evaluated_ticks: int
    boundary_eval_ticks: int
    main_boundary_errors: List[int]
    segment_rows: List[Dict[str, Any]]
    boundary_rows: List[Dict[str, Any]]
    decoded_rows: List[Dict[str, Any]]


@dataclass
class EvaluationResult:
    metrics: Dict[str, Any]
    episode_metrics: List[Dict[str, Any]]
    segment_matches: List[Dict[str, Any]]
    boundary_matches: List[Dict[str, Any]]
    decoded_segments: List[Dict[str, Any]]


def _derive_boundary_name(previous_label: int, current_label: int) -> str:
    return (
        BOUNDARY_CLASS_CHANGE
        if previous_label != current_label
        else BOUNDARY_SAME_LABEL
    )


def ground_truth_segments(
    episode: GroundTruthEpisode, *, strict: bool = True
) -> List[Segment]:
    """Build raw GT instances from instance IDs and human boundary markers."""

    valid = (
        episode.label_mask
        & (episode.atomic_target >= 0)
        & (episode.atomic_instance_id >= 0)
    )
    if strict and np.any(episode.label_mask & ~valid):
        raise ValueError(
            f"{episode.episode_id}: labeled ticks require atomic_target and atomic_instance_id"
        )

    segments: List[Segment] = []
    run_id = 0
    for run_start, run_end in _valid_runs(valid):
        starts = [run_start]
        for index in range(run_start + 1, run_end):
            instance_changed = (
                episode.atomic_instance_id[index]
                != episode.atomic_instance_id[index - 1]
            )
            hard = bool(episode.boundary_hard[index])
            label_changed = episode.atomic_target[index] != episode.atomic_target[index - 1]
            if strict and hard != instance_changed:
                raise ValueError(
                    f"{episode.episode_id}: boundary_hard/instance-id mismatch at "
                    f"tick {int(episode.global_frame_60[index])}"
                )
            if strict and label_changed and not instance_changed:
                raise ValueError(
                    f"{episode.episode_id}: class change without a new instance at "
                    f"tick {int(episode.global_frame_60[index])}"
                )
            if hard or instance_changed:
                starts.append(index)
        ends = [*starts[1:], run_end]

        for start, end in zip(starts, ends):
            labels = np.unique(episode.atomic_target[start:end])
            instance_ids = np.unique(episode.atomic_instance_id[start:end])
            if strict and (labels.size != 1 or instance_ids.size != 1):
                raise ValueError(
                    f"{episode.episode_id}: non-constant GT instance in [{start},{end})"
                )
            label = int(labels[0])
            instance_id = int(instance_ids[0])
            source = None if start == run_start else "human_instance_boundary"
            segments.append(
                Segment(
                    start_index=start,
                    end_index=end,
                    start_tick=int(episode.global_frame_60[start]),
                    end_tick=int(episode.global_frame_60[end - 1]) + 1,
                    label=label,
                    confidence=1.0,
                    run_id=run_id,
                    instance_id=instance_id,
                    boundary_score=1.0 if source else None,
                    boundary_source=source,
                )
            )
        run_id += 1
    return segments


def ground_truth_boundary_types(
    episode: GroundTruthEpisode, *, strict: bool = True
) -> Dict[int, str]:
    """Return evaluable human boundaries without recreating them from labels."""

    valid = episode.label_mask & (episode.atomic_target >= 0)
    output: Dict[int, str] = {}
    for run_start, run_end in _valid_runs(valid):
        for index in range(run_start + 1, run_end):
            if not episode.boundary_hard[index] or not episode.boundary_eval_mask[index]:
                continue
            expected = _derive_boundary_name(
                int(episode.atomic_target[index - 1]),
                int(episode.atomic_target[index]),
            )
            type_id = int(episode.boundary_type[index])
            declared = BOUNDARY_ID_TO_NAME.get(type_id)
            if strict and declared != expected:
                raise ValueError(
                    f"{episode.episode_id}: boundary_type={type_id} disagrees with labels "
                    f"at tick {int(episode.global_frame_60[index])}"
                )
            output[int(episode.global_frame_60[index])] = declared or expected
    return output


def _is_better(
    candidate_count: int,
    candidate_cost: float,
    best_count: int,
    best_cost: float,
    *,
    candidate_priority: int,
    best_priority: int,
) -> bool:
    if candidate_count != best_count:
        return candidate_count > best_count
    # The protocol says *minimum* total error, not approximately minimum.
    # ``np.isclose`` would treat a one-tick difference as a tie once aggregate
    # costs become large enough and could select the higher-error traceback.
    if candidate_cost != best_cost:
        return candidate_cost < best_cost
    return candidate_priority > best_priority


def match_boundaries_optimal(
    predicted: Sequence[int], target: Sequence[int], tolerance: int
) -> BoundaryMatchResult:
    """Maximum-cardinality ordered matching, then minimum absolute error.

    A nearest-pair greedy matcher can sacrifice recall.  Dynamic programming on
    the two ordered boundary streams enforces the frozen protocol objective:
    maximize the number of one-to-one matches first and minimize total temporal
    error only among maximum-cardinality solutions.
    """

    if tolerance < 0:
        raise ValueError("Boundary tolerance cannot be negative")
    predictions = tuple(sorted(int(value) for value in predicted))
    truths = tuple(sorted(int(value) for value in target))
    if len(set(predictions)) != len(predictions) or len(set(truths)) != len(truths):
        raise ValueError("Boundary positions must be unique within an episode")
    n_predictions, n_truths = len(predictions), len(truths)
    counts = np.zeros((n_predictions + 1, n_truths + 1), dtype=np.int32)
    costs = np.zeros((n_predictions + 1, n_truths + 1), dtype=np.float64)
    decisions = np.zeros((n_predictions + 1, n_truths + 1), dtype=np.int8)
    decisions[1:, 0] = 1  # skip prediction
    decisions[0, 1:] = 2  # skip truth

    for row in range(1, n_predictions + 1):
        for column in range(1, n_truths + 1):
            best_count = int(counts[row - 1, column])
            best_cost = float(costs[row - 1, column])
            best_decision = 1
            best_priority = 1

            candidate_count = int(counts[row, column - 1])
            candidate_cost = float(costs[row, column - 1])
            if _is_better(
                candidate_count,
                candidate_cost,
                best_count,
                best_cost,
                candidate_priority=2,
                best_priority=best_priority,
            ):
                best_count, best_cost = candidate_count, candidate_cost
                best_decision, best_priority = 2, 2

            error = abs(predictions[row - 1] - truths[column - 1])
            if error <= tolerance:
                candidate_count = int(counts[row - 1, column - 1]) + 1
                candidate_cost = float(costs[row - 1, column - 1]) + error
                if _is_better(
                    candidate_count,
                    candidate_cost,
                    best_count,
                    best_cost,
                    candidate_priority=3,
                    best_priority=best_priority,
                ):
                    best_count, best_cost = candidate_count, candidate_cost
                    best_decision = 3

            counts[row, column] = best_count
            costs[row, column] = best_cost
            decisions[row, column] = best_decision

    pairs: List[Tuple[int, int]] = []
    row, column = n_predictions, n_truths
    while row > 0 or column > 0:
        decision = int(decisions[row, column])
        if row > 0 and column > 0 and decision == 3:
            pairs.append((predictions[row - 1], truths[column - 1]))
            row -= 1
            column -= 1
        elif row > 0 and (column == 0 or decision == 1):
            row -= 1
        elif column > 0:
            column -= 1
        else:
            raise AssertionError("Boundary-match traceback did not progress")
    pairs.reverse()
    return BoundaryMatchResult(predictions, truths, tuple(pairs), tolerance)


def _segment_iou(first: Segment, second: Segment) -> float:
    intersection = max(
        0,
        min(first.end_tick, second.end_tick) - max(first.start_tick, second.start_tick),
    )
    union = max(first.end_tick, second.end_tick) - min(
        first.start_tick, second.start_tick
    )
    return float(intersection / max(union, 1))


def match_segments_optimal(
    predicted: Sequence[Segment], target: Sequence[Segment], threshold: float
) -> SegmentMatchResult:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Segment IoU threshold must be in [0,1]")
    n_predictions, n_truths = len(predicted), len(target)
    counts = np.zeros((n_predictions + 1, n_truths + 1), dtype=np.int32)
    costs = np.zeros((n_predictions + 1, n_truths + 1), dtype=np.float64)
    decisions = np.zeros((n_predictions + 1, n_truths + 1), dtype=np.int8)
    decisions[1:, 0] = 1
    decisions[0, 1:] = 2

    for row in range(1, n_predictions + 1):
        for column in range(1, n_truths + 1):
            best_count = int(counts[row - 1, column])
            best_cost = float(costs[row - 1, column])
            best_decision, best_priority = 1, 1

            candidate_count = int(counts[row, column - 1])
            candidate_cost = float(costs[row, column - 1])
            if _is_better(
                candidate_count,
                candidate_cost,
                best_count,
                best_cost,
                candidate_priority=2,
                best_priority=best_priority,
            ):
                best_count, best_cost = candidate_count, candidate_cost
                best_decision, best_priority = 2, 2

            prediction = predicted[row - 1]
            truth = target[column - 1]
            iou = _segment_iou(prediction, truth)
            allowed = (
                prediction.label == truth.label
                and prediction.run_id == truth.run_id
                and iou >= threshold
            )
            if allowed:
                candidate_count = int(counts[row - 1, column - 1]) + 1
                candidate_cost = float(costs[row - 1, column - 1]) + (1.0 - iou)
                if _is_better(
                    candidate_count,
                    candidate_cost,
                    best_count,
                    best_cost,
                    candidate_priority=3,
                    best_priority=best_priority,
                ):
                    best_count, best_cost = candidate_count, candidate_cost
                    best_decision = 3
            counts[row, column] = best_count
            costs[row, column] = best_cost
            decisions[row, column] = best_decision

    pairs: List[Tuple[int, int, float]] = []
    row, column = n_predictions, n_truths
    while row > 0 or column > 0:
        decision = int(decisions[row, column])
        if row > 0 and column > 0 and decision == 3:
            pairs.append(
                (
                    row - 1,
                    column - 1,
                    _segment_iou(predicted[row - 1], target[column - 1]),
                )
            )
            row -= 1
            column -= 1
        elif row > 0 and (column == 0 or decision == 1):
            row -= 1
        elif column > 0:
            column -= 1
        else:
            raise AssertionError("Segment-match traceback did not progress")
    pairs.reverse()
    return SegmentMatchResult(tuple(pairs), n_predictions, n_truths, threshold)


def _levenshtein(first: Sequence[int], second: Sequence[int]) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for row, first_value in enumerate(first, start=1):
        current = [row]
        for column, second_value in enumerate(second, start=1):
            substitution = previous[column - 1] + int(first_value != second_value)
            current.append(
                min(previous[column] + 1, current[column - 1] + 1, substitution)
            )
        previous = current
    return previous[-1]


def _edit_similarity(predicted: Sequence[Segment], target: Sequence[Segment]) -> float:
    scores: List[float] = []
    run_ids = sorted(
        {segment.run_id for segment in predicted}
        | {segment.run_id for segment in target}
    )
    for run_id in run_ids:
        pred_tokens = [segment.label for segment in predicted if segment.run_id == run_id]
        gt_tokens = [segment.label for segment in target if segment.run_id == run_id]
        denominator = max(len(pred_tokens), len(gt_tokens), 1)
        scores.append(100.0 * (1.0 - _levenshtein(pred_tokens, gt_tokens) / denominator))
    return float(np.mean(scores)) if scores else 0.0


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 100.0 * 2 * true_positive / denominator if denominator else 0.0


def _prf(true_positive: int, false_positive: int, false_negative: int) -> Dict[str, float | int]:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": 100.0 * true_positive / precision_denominator
        if precision_denominator
        else 0.0,
        "recall": 100.0 * true_positive / recall_denominator
        if recall_denominator
        else 0.0,
        "f1": _f1(true_positive, false_positive, false_negative),
    }


def _validate_metric_protocol(
    segment_iou_thresholds: Sequence[float],
    boundary_tolerances_ms: Sequence[int],
    primary_boundary_tolerance_ms: int,
    output_rate_hz: float,
) -> Tuple[Tuple[float, ...], Tuple[int, ...]]:
    thresholds = tuple(float(value) for value in segment_iou_thresholds)
    tolerances = tuple(int(value) for value in boundary_tolerances_ms)
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("segment_iou_thresholds must be non-empty values in [0,1]")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("segment_iou_thresholds must be unique")
    if not any(np.isclose(value, 0.50) for value in thresholds):
        raise ValueError("The frozen primary score requires segment IoU threshold 0.50")
    if not tolerances or any(value < 0 for value in tolerances):
        raise ValueError("boundary_tolerances_ms must be non-empty non-negative values")
    if len(set(tolerances)) != len(tolerances):
        raise ValueError("boundary_tolerances_ms must be unique")
    if int(primary_boundary_tolerance_ms) not in tolerances:
        raise ValueError("Primary boundary tolerance must be in boundary_tolerances_ms")
    if not np.isfinite(output_rate_hz) or output_rate_hz <= 0:
        raise ValueError("output_rate_hz must be finite and positive")
    return thresholds, tolerances


def _align_prediction(
    prediction: StitchedEpisode,
    target: GroundTruthEpisode,
    *,
    strict: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if prediction.episode_id != target.episode_id:
        raise ValueError("Prediction and target episode IDs differ")
    class_count = prediction.atomic_logits.shape[1]
    logits = np.zeros((target.num_ticks, class_count), dtype=np.float32)
    boundary = np.zeros(target.num_ticks, dtype=np.float32)
    covered = np.zeros(target.num_ticks, dtype=np.bool_)
    target_positions = {
        int(tick): index for index, tick in enumerate(target.global_frame_60.tolist())
    }
    for source_index, tick in enumerate(prediction.global_frame_60.tolist()):
        destination = target_positions.get(int(tick))
        if destination is None:
            if strict and prediction.coverage_mask[source_index]:
                raise ValueError(
                    f"{target.episode_id}: prediction tick {tick} is outside GT clock"
                )
            continue
        if not prediction.coverage_mask[source_index]:
            continue
        logits[destination] = prediction.atomic_logits[source_index]
        boundary[destination] = prediction.boundary_probability[source_index]
        covered[destination] = True
    missing = target.label_mask & ~covered
    if strict and missing.any():
        ticks = target.global_frame_60[missing]
        raise ValueError(
            f"{target.episode_id}: {ticks.size} labeled ticks lack stitched predictions; "
            f"first={int(ticks[0])}"
        )
    return logits, boundary, covered


def decode_evaluable_prediction(
    logits: np.ndarray,
    boundary_probability: np.ndarray,
    target: GroundTruthEpisode,
    covered: np.ndarray,
    *,
    boundary_threshold: float,
    boundary_min_distance_frames: int,
    min_segment_frames: int = 1,
    use_boundary_head: bool,
    include_action_changes: bool,
) -> Tuple[np.ndarray, List[Segment], Dict[int, str]]:
    """Apply the evaluator decoder and its two masks in one canonical place.

    C0 validation calls this function too.  That prevents calibration from
    scoring its detector's *pre-decoder* peaks (or peaks in unlabeled gaps)
    while the held-out evaluator scores post-threshold/post-NMS instances.
    """

    covered_value = _array(covered, np.bool_)
    if covered_value.shape != (target.num_ticks,):
        raise ValueError("covered must align with the ground-truth episode")
    evaluation_mask = (
        target.label_mask & covered_value & (target.atomic_target >= 0)
    )
    predicted_raw = boundary_aware_decode(
        logits,
        boundary_probability,
        threshold=boundary_threshold,
        min_distance=boundary_min_distance_frames,
        min_segment_frames=min_segment_frames,
        valid_mask=evaluation_mask,
        global_frame_60=target.global_frame_60,
        use_boundary_head=use_boundary_head,
        include_action_changes=include_action_changes,
    )
    tick_to_index = {
        int(tick): index for index, tick in enumerate(target.global_frame_60.tolist())
    }
    predicted_types = {
        tick: boundary_type
        for tick, boundary_type in decoded_boundary_types(predicted_raw).items()
        if tick in tick_to_index
        and target.boundary_eval_mask[tick_to_index[tick]]
        and evaluation_mask[tick_to_index[tick]]
    }
    return evaluation_mask, predicted_raw, predicted_types


def _segment_artifact_rows(
    episode_id: str,
    view: str,
    threshold: float,
    predicted: Sequence[Segment],
    target: Sequence[Segment],
    result: SegmentMatchResult,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    matched_pred = {pair[0] for pair in result.pairs}
    matched_gt = {pair[1] for pair in result.pairs}
    for pred_index, gt_index, iou in result.pairs:
        rows.append(
            {
                "episode_id": episode_id,
                "view": view,
                "iou_threshold": threshold,
                "status": "matched",
                "iou": iou,
                "predicted": predicted[pred_index].to_dict(),
                "target": target[gt_index].to_dict(),
            }
        )
    for index, segment in enumerate(predicted):
        if index not in matched_pred:
            rows.append(
                {
                    "episode_id": episode_id,
                    "view": view,
                    "iou_threshold": threshold,
                    "status": "false_positive",
                    "iou": None,
                    "predicted": segment.to_dict(),
                    "target": None,
                }
            )
    for index, segment in enumerate(target):
        if index not in matched_gt:
            rows.append(
                {
                    "episode_id": episode_id,
                    "view": view,
                    "iou_threshold": threshold,
                    "status": "missed",
                    "iou": None,
                    "predicted": None,
                    "target": segment.to_dict(),
                }
            )
    return rows


def _boundary_artifact_rows(
    episode_id: str,
    tolerance_ms: int,
    evaluation_type: str,
    result: BoundaryMatchResult,
    predicted_types: Mapping[int, str],
    target_types: Mapping[int, str],
    output_rate_hz: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    matched_pred = {pair[0] for pair in result.pairs}
    matched_gt = {pair[1] for pair in result.pairs}
    for prediction, truth in result.pairs:
        error_ticks = prediction - truth
        rows.append(
            {
                "episode_id": episode_id,
                "tolerance_ms": tolerance_ms,
                "evaluation_type": evaluation_type,
                "status": "matched",
                "predicted_tick": prediction,
                "target_tick": truth,
                "predicted_type": predicted_types.get(prediction),
                "target_type": target_types.get(truth),
                "signed_error_ticks": error_ticks,
                "signed_error_ms": 1000.0 * error_ticks / output_rate_hz,
                "absolute_error_ms": 1000.0 * abs(error_ticks) / output_rate_hz,
            }
        )
    for prediction in result.predicted:
        if prediction not in matched_pred:
            rows.append(
                {
                    "episode_id": episode_id,
                    "tolerance_ms": tolerance_ms,
                    "evaluation_type": evaluation_type,
                    "status": "false_positive",
                    "predicted_tick": prediction,
                    "target_tick": None,
                    "predicted_type": predicted_types.get(prediction),
                    "target_type": None,
                    "signed_error_ticks": None,
                    "signed_error_ms": None,
                    "absolute_error_ms": None,
                }
            )
    for truth in result.target:
        if truth not in matched_gt:
            rows.append(
                {
                    "episode_id": episode_id,
                    "tolerance_ms": tolerance_ms,
                    "evaluation_type": evaluation_type,
                    "status": "missed",
                    "predicted_tick": None,
                    "target_tick": truth,
                    "predicted_type": None,
                    "target_type": target_types.get(truth),
                    "signed_error_ticks": None,
                    "signed_error_ms": None,
                    "absolute_error_ms": None,
                }
            )
    return rows


def evaluate_episode(
    prediction: StitchedEpisode,
    target: GroundTruthEpisode,
    *,
    num_actions: int,
    segment_iou_thresholds: Sequence[float] = (0.10, 0.25, 0.50),
    boundary_tolerances_ms: Sequence[int] = (50, 100, 250, 500),
    primary_boundary_tolerance_ms: int = 250,
    output_rate_hz: float = 60.0,
    boundary_threshold: float = 0.5,
    boundary_min_distance_frames: int = 6,
    min_segment_frames: int = 1,
    use_boundary_head: bool = True,
    include_action_changes: bool = True,
    strict: bool = True,
    collect_artifacts: bool = True,
) -> EpisodeEvaluation:
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")
    segment_iou_thresholds, boundary_tolerances_ms = _validate_metric_protocol(
        segment_iou_thresholds,
        boundary_tolerances_ms,
        primary_boundary_tolerance_ms,
        output_rate_hz,
    )
    logits, boundary_probability, covered = _align_prediction(
        prediction, target, strict=strict
    )
    evaluation_mask, predicted_raw, predicted_boundary_types = (
        decode_evaluable_prediction(
            logits,
            boundary_probability,
            target,
            covered,
            boundary_threshold=boundary_threshold,
            boundary_min_distance_frames=boundary_min_distance_frames,
            min_segment_frames=min_segment_frames,
            use_boundary_head=use_boundary_head,
            include_action_changes=include_action_changes,
        )
    )
    predicted_collapsed = collapse_adjacent_same_label(predicted_raw)
    target_raw = ground_truth_segments(target, strict=strict)
    target_collapsed = collapse_adjacent_same_label(target_raw)

    decoded_labels = segments_to_frame_labels(predicted_raw, target.num_ticks)
    confusion = np.zeros((num_actions, num_actions), dtype=np.int64)
    for truth, guess in zip(
        target.atomic_target[evaluation_mask], decoded_labels[evaluation_mask]
    ):
        if not 0 <= int(truth) < num_actions:
            if strict:
                raise ValueError(f"Target class {truth} lies outside num_actions")
            continue
        if not 0 <= int(guess) < num_actions:
            if strict:
                raise ValueError(f"Predicted class {guess} lies outside num_actions")
            continue
        confusion[int(truth), int(guess)] += 1

    segment_counts: Dict[Tuple[str, float], Tuple[int, int, int]] = {}
    segment_rows: List[Dict[str, Any]] = []
    per_view_metrics: Dict[str, Any] = {}
    for view, pred_segments, gt_segments in (
        ("raw", predicted_raw, target_raw),
        ("collapsed", predicted_collapsed, target_collapsed),
    ):
        view_metrics: Dict[str, Any] = {
            "predicted_segments": len(pred_segments),
            "target_segments": len(gt_segments),
        }
        for threshold in segment_iou_thresholds:
            result = match_segments_optimal(pred_segments, gt_segments, float(threshold))
            counts = (
                result.true_positive,
                result.false_positive,
                result.false_negative,
            )
            segment_counts[(view, float(threshold))] = counts
            view_metrics[f"f1@{int(round(100 * threshold))}"] = _f1(*counts)
            if collect_artifacts:
                segment_rows.extend(
                    _segment_artifact_rows(
                        target.episode_id,
                        view,
                        float(threshold),
                        pred_segments,
                        gt_segments,
                        result,
                    )
                )
        view_metrics["edit"] = _edit_similarity(pred_segments, gt_segments)
        per_view_metrics[view] = view_metrics

    target_boundary_types = ground_truth_boundary_types(target, strict=strict)

    boundary_counts: Dict[Tuple[str, int], Tuple[int, int, int]] = {}
    gt_type_any_counts: Dict[Tuple[str, int], Tuple[int, int]] = {}
    boundary_rows: List[Dict[str, Any]] = []
    main_errors: List[int] = []
    boundary_metrics: Dict[str, Any] = {}
    for tolerance_ms in boundary_tolerances_ms:
        tolerance_frames = int(round(float(tolerance_ms) * output_rate_hz / 1000.0))
        for evaluation_type in EVALUATION_TYPES:
            predictions_for_type = [
                tick
                for tick, boundary_type in predicted_boundary_types.items()
                if evaluation_type == "all" or boundary_type == evaluation_type
            ]
            targets_for_type = [
                tick
                for tick, boundary_type in target_boundary_types.items()
                if evaluation_type == "all" or boundary_type == evaluation_type
            ]
            match = match_boundaries_optimal(
                predictions_for_type, targets_for_type, tolerance_frames
            )
            counts = (match.true_positive, match.false_positive, match.false_negative)
            boundary_counts[(evaluation_type, int(tolerance_ms))] = counts
            boundary_metrics.setdefault(evaluation_type, {})[
                f"{int(tolerance_ms)}ms"
            ] = _prf(*counts)
            if collect_artifacts:
                boundary_rows.extend(
                    _boundary_artifact_rows(
                        target.episode_id,
                        int(tolerance_ms),
                        evaluation_type,
                        match,
                        predicted_boundary_types,
                        target_boundary_types,
                        output_rate_hz,
                    )
                )
            if evaluation_type == "all" and tolerance_ms == primary_boundary_tolerance_ms:
                main_errors.extend(match.signed_errors)

        for target_type in (BOUNDARY_CLASS_CHANGE, BOUNDARY_SAME_LABEL):
            truths = [
                tick
                for tick, boundary_type in target_boundary_types.items()
                if boundary_type == target_type
            ]
            match = match_boundaries_optimal(
                list(predicted_boundary_types), truths, tolerance_frames
            )
            gt_type_any_counts[(target_type, int(tolerance_ms))] = (
                match.true_positive,
                len(truths),
            )

    total = int(confusion.sum())
    frame_accuracy = 100.0 * float(np.trace(confusion)) / total if total else 0.0
    class_f1: List[float] = []
    for class_index in range(num_actions):
        tp = int(confusion[class_index, class_index])
        fp = int(confusion[:, class_index].sum()) - tp
        fn = int(confusion[class_index, :].sum()) - tp
        if tp + fn:
            class_f1.append(_f1(tp, fp, fn))
    raw_edit = _edit_similarity(predicted_raw, target_raw)
    collapsed_edit = _edit_similarity(predicted_collapsed, target_collapsed)
    raw_count_error = float(abs(len(predicted_raw) - len(target_raw)))
    raw_target_count = len(target_raw)
    raw_predicted_count = len(predicted_raw)
    raw_count_denominator = max(raw_target_count, 1)
    metrics = {
        "episode_id": target.episode_id,
        "evaluated_ticks": int(evaluation_mask.sum()),
        "frame_accuracy": frame_accuracy,
        "tick_macro_f1": float(np.mean(class_f1)) if class_f1 else 0.0,
        "raw": {
            **per_view_metrics["raw"],
            "edit": raw_edit,
            "over_segmentation_ratio": float(
                max(raw_predicted_count - raw_target_count, 0)
                / raw_count_denominator
            ),
            "under_segmentation_ratio": float(
                max(raw_target_count - raw_predicted_count, 0)
                / raw_count_denominator
            ),
        },
        "collapsed": {**per_view_metrics["collapsed"], "edit": collapsed_edit},
        "raw_segment_count_absolute_error": raw_count_error,
        "boundary": boundary_metrics,
        "decoder": {
            "use_boundary_head": bool(use_boundary_head),
            "include_action_change_boundaries": bool(include_action_changes),
            "boundary_threshold": float(boundary_threshold),
            "boundary_min_distance_frames": int(boundary_min_distance_frames),
            "min_segment_frames": int(min_segment_frames),
        },
    }
    decoded_rows: List[Dict[str, Any]] = []
    if collect_artifacts:
        for origin, view, segments in (
            ("prediction", "raw", predicted_raw),
            ("prediction", "collapsed", predicted_collapsed),
            ("target", "raw", target_raw),
            ("target", "collapsed", target_collapsed),
        ):
            for segment in segments:
                decoded_rows.append(
                    {
                        "episode_id": target.episode_id,
                        "origin": origin,
                        "view": view,
                        **segment.to_dict(),
                    }
                )

    return EpisodeEvaluation(
        episode_id=target.episode_id,
        metrics=metrics,
        confusion=confusion,
        segment_counts=segment_counts,
        boundary_counts=boundary_counts,
        gt_type_any_prediction_counts=gt_type_any_counts,
        raw_edit=raw_edit,
        collapsed_edit=collapsed_edit,
        raw_count_error=raw_count_error,
        evaluated_ticks=int(evaluation_mask.sum()),
        boundary_eval_ticks=int((target.boundary_eval_mask & evaluation_mask).sum()),
        main_boundary_errors=main_errors,
        segment_rows=segment_rows,
        boundary_rows=boundary_rows,
        decoded_rows=decoded_rows,
    )


def _merge_counts(
    destination: MutableMapping[Any, List[int]],
    source: Mapping[Any, Tuple[int, int, int]],
) -> None:
    for key, values in source.items():
        totals = destination.setdefault(key, [0, 0, 0])
        for index, value in enumerate(values):
            totals[index] += int(value)


def evaluate_dataset(
    predictions: Mapping[str, StitchedEpisode],
    ground_truth: Mapping[str, GroundTruthEpisode],
    *,
    num_actions: int,
    segment_iou_thresholds: Sequence[float] = (0.10, 0.25, 0.50),
    boundary_tolerances_ms: Sequence[int] = (50, 100, 250, 500),
    primary_boundary_tolerance_ms: int = 250,
    output_rate_hz: float = 60.0,
    boundary_threshold: float = 0.5,
    boundary_min_distance_frames: int = 6,
    min_segment_frames: int = 1,
    use_boundary_head: bool = True,
    include_action_changes: bool = True,
    strict: bool = True,
    collect_artifacts: bool = True,
) -> EvaluationResult:
    segment_iou_thresholds, boundary_tolerances_ms = _validate_metric_protocol(
        segment_iou_thresholds,
        boundary_tolerances_ms,
        primary_boundary_tolerance_ms,
        output_rate_hz,
    )
    prediction_ids = set(predictions)
    target_ids = set(ground_truth)
    if prediction_ids != target_ids:
        raise ValueError(
            "Prediction/GT episode sets differ: "
            f"prediction_only={sorted(prediction_ids-target_ids)[:5]}, "
            f"target_only={sorted(target_ids-prediction_ids)[:5]}"
        )

    evaluations: List[EpisodeEvaluation] = []
    for episode_id in sorted(target_ids):
        evaluations.append(
            evaluate_episode(
                predictions[episode_id],
                ground_truth[episode_id],
                num_actions=num_actions,
                segment_iou_thresholds=segment_iou_thresholds,
                boundary_tolerances_ms=boundary_tolerances_ms,
                primary_boundary_tolerance_ms=primary_boundary_tolerance_ms,
                output_rate_hz=output_rate_hz,
                boundary_threshold=boundary_threshold,
                boundary_min_distance_frames=boundary_min_distance_frames,
                min_segment_frames=min_segment_frames,
                use_boundary_head=use_boundary_head,
                include_action_changes=include_action_changes,
                strict=strict,
                collect_artifacts=collect_artifacts,
            )
        )

    confusion = np.zeros((num_actions, num_actions), dtype=np.int64)
    segment_totals: Dict[Tuple[str, float], List[int]] = {}
    boundary_totals: Dict[Tuple[str, int], List[int]] = {}
    gt_type_any_totals: Dict[Tuple[str, int], List[int]] = {}
    main_errors: List[int] = []
    for evaluation in evaluations:
        confusion += evaluation.confusion
        _merge_counts(segment_totals, evaluation.segment_counts)
        _merge_counts(boundary_totals, evaluation.boundary_counts)
        for key, (matched, total) in evaluation.gt_type_any_prediction_counts.items():
            values = gt_type_any_totals.setdefault(key, [0, 0])
            values[0] += matched
            values[1] += total
        main_errors.extend(evaluation.main_boundary_errors)

    total_ticks = int(confusion.sum())
    frame_accuracy = (
        100.0 * float(np.trace(confusion)) / total_ticks if total_ticks else 0.0
    )
    per_class: List[Dict[str, Any]] = []
    macro_values: List[float] = []
    for class_index in range(num_actions):
        tp = int(confusion[class_index, class_index])
        fp = int(confusion[:, class_index].sum()) - tp
        fn = int(confusion[class_index, :].sum()) - tp
        values = _prf(tp, fp, fn)
        support = int(confusion[class_index, :].sum())
        per_class.append({"class_index": class_index, "support": support, **values})
        if support:
            macro_values.append(float(values["f1"]))

    view_metrics: Dict[str, Any] = {}
    for view in ("raw", "collapsed"):
        values: Dict[str, Any] = {}
        for threshold in segment_iou_thresholds:
            counts = segment_totals[(view, float(threshold))]
            values[f"f1@{int(round(100 * threshold))}"] = _f1(*counts)
            values[f"counts@{int(round(100 * threshold))}"] = {
                "tp": counts[0],
                "fp": counts[1],
                "fn": counts[2],
            }
        values["edit"] = float(
            np.mean(
                [
                    evaluation.raw_edit if view == "raw" else evaluation.collapsed_edit
                    for evaluation in evaluations
                ]
            )
        ) if evaluations else 0.0
        view_metrics[view] = values

    boundary_metrics: Dict[str, Any] = {}
    for evaluation_type in EVALUATION_TYPES:
        type_metrics: Dict[str, Any] = {}
        for tolerance_ms in boundary_tolerances_ms:
            type_metrics[f"{int(tolerance_ms)}ms"] = _prf(
                *boundary_totals[(evaluation_type, int(tolerance_ms))]
            )
        boundary_metrics[evaluation_type] = type_metrics

    gt_type_recall: Dict[str, Any] = {}
    for target_type in (BOUNDARY_CLASS_CHANGE, BOUNDARY_SAME_LABEL):
        values: Dict[str, Any] = {}
        for tolerance_ms in boundary_tolerances_ms:
            matched, total = gt_type_any_totals[(target_type, int(tolerance_ms))]
            values[f"{int(tolerance_ms)}ms"] = {
                "matched": matched,
                "target": total,
                "recall": 100.0 * matched / total if total else 0.0,
            }
        gt_type_recall[target_type] = values

    main_counts = boundary_totals[("all", int(primary_boundary_tolerance_ms))]
    errors = np.asarray(main_errors, dtype=np.float64)
    error_summary = {
        "matched": int(errors.size),
        "median_absolute_error_ms": float(np.median(np.abs(errors)) * 1000.0 / output_rate_hz)
        if errors.size
        else None,
        "p90_absolute_error_ms": float(np.percentile(np.abs(errors), 90) * 1000.0 / output_rate_hz)
        if errors.size
        else None,
        "signed_median_error_ms": float(np.median(errors) * 1000.0 / output_rate_hz)
        if errors.size
        else None,
        "miss_rate": main_counts[2] / (main_counts[0] + main_counts[2])
        if main_counts[0] + main_counts[2]
        else 0.0,
    }
    boundary_eval_ticks = sum(item.boundary_eval_ticks for item in evaluations)
    duration_minutes = boundary_eval_ticks / output_rate_hz / 60.0
    error_summary["false_boundaries_per_minute"] = (
        main_counts[1] / duration_minutes if duration_minutes else 0.0
    )

    raw_f1_50 = float(view_metrics["raw"].get("f1@50", 0.0))
    collapsed_f1_50 = float(view_metrics["collapsed"].get("f1@50", 0.0))
    primary_boundary_f1 = float(
        boundary_metrics["all"][f"{int(primary_boundary_tolerance_ms)}ms"]["f1"]
    )
    class_change_boundary_f1 = float(
        boundary_metrics["class_change"][f"{int(primary_boundary_tolerance_ms)}ms"]["f1"]
    )
    total_raw_target_segments = sum(
        int(evaluation.metrics["raw"]["target_segments"])
        for evaluation in evaluations
    )
    raw_over_count = sum(
        max(
            int(evaluation.metrics["raw"]["predicted_segments"])
            - int(evaluation.metrics["raw"]["target_segments"]),
            0,
        )
        for evaluation in evaluations
    )
    raw_under_count = sum(
        max(
            int(evaluation.metrics["raw"]["target_segments"])
            - int(evaluation.metrics["raw"]["predicted_segments"]),
            0,
        )
        for evaluation in evaluations
    )
    metrics = {
        "episodes": len(evaluations),
        "evaluated_ticks": sum(item.evaluated_ticks for item in evaluations),
        "output_rate_hz": output_rate_hz,
        "frame_accuracy": frame_accuracy,
        "tick_macro_f1": float(np.mean(macro_values)) if macro_values else 0.0,
        "decoder": {
            "use_boundary_head": bool(use_boundary_head),
            "include_action_change_boundaries": bool(include_action_changes),
            "boundary_threshold": float(boundary_threshold),
            "boundary_min_distance_frames": int(boundary_min_distance_frames),
            "min_segment_frames": int(min_segment_frames),
        },
        "per_class": per_class,
        "raw": {
            **view_metrics["raw"],
            "segment_count_mae": float(
                np.mean([item.raw_count_error for item in evaluations])
            )
            if evaluations
            else 0.0,
            "over_segmentation_ratio": float(
                raw_over_count / total_raw_target_segments
            )
            if total_raw_target_segments
            else 0.0,
            "under_segmentation_ratio": float(
                raw_under_count / total_raw_target_segments
            )
            if total_raw_target_segments
            else 0.0,
            "segment_count_ratio_counts": {
                "over_excess_segments": int(raw_over_count),
                "under_missing_segments": int(raw_under_count),
                "target_segments": int(total_raw_target_segments),
            },
        },
        "collapsed": view_metrics["collapsed"],
        "boundary": {
            **boundary_metrics,
            "gt_type_recall_with_any_prediction_type": gt_type_recall,
            "primary_tolerance_ms": int(primary_boundary_tolerance_ms),
            "primary_error_summary": error_summary,
        },
        # 2026-08-05 view switch: the composite now tracks the semantic
        # (same-label collapsed) view and class-change boundaries only.
        # Same-label instance boundaries stay reported but stop steering
        # checkpoint selection and decode calibration: a 500 ms tolerance
        # recovers only x1.31 of their F1 against x1.28 for class changes,
        # so their deficit is annotation existence disagreement rather than
        # model error, and raw F1@50 swings 11.9-13.3 across the decode grid
        # against 0.35-0.74 for class-change boundaries.
        "primary": {
            "collapsed_semantic_f1@50": collapsed_f1_50,
            f"class_change_boundary_f1@{int(primary_boundary_tolerance_ms)}ms": (
                class_change_boundary_f1
            ),
            "checkpoint_composite": (
                0.5 * collapsed_f1_50 + 0.5 * class_change_boundary_f1
            ),
            # Retained for reporting, lineage and tie breaking only.
            "raw_instance_f1@50": raw_f1_50,
            f"raw_boundary_f1@{int(primary_boundary_tolerance_ms)}ms": primary_boundary_f1,
        },
    }

    return EvaluationResult(
        metrics=metrics,
        episode_metrics=[item.metrics for item in evaluations],
        segment_matches=[row for item in evaluations for row in item.segment_rows],
        boundary_matches=[row for item in evaluations for row in item.boundary_rows],
        decoded_segments=[row for item in evaluations for row in item.decoded_rows],
    )
