"""Full-episode stitching and boundary-aware decoding for ContactTAS.

The training scaffold operates on overlapping read/core windows.  This module
keeps inference independent from the storage backend: callers provide one
``WindowPrediction`` per window, keyed by the episode-level 60 Hz tick index,
and receive one deterministic ``StitchedEpisode``.  Decoding intentionally
keeps a boundary-head split even when the action labels on both sides are the
same; collapsing those instances is a separate, explicit evaluation view.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


BOUNDARY_CLASS_CHANGE = "class_change"
BOUNDARY_SAME_LABEL = "same_label_repeat"


def _as_array(value: np.ndarray | Sequence[object], dtype: np.dtype) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _valid_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    """Yield half-open contiguous runs in a one-dimensional boolean mask."""

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
class WindowPrediction:
    """Predictions for one read window, expressed on episode-global ticks.

    ``core_mask`` controls which ticks contribute to stitching.  Context ticks
    can therefore be read by the model without being counted twice.  Boundary
    values must already be probabilities in ``[0, 1]``.
    """

    episode_id: str
    global_frame_60: np.ndarray
    atomic_logits: np.ndarray
    boundary_probability: np.ndarray
    core_mask: np.ndarray | None = None
    time_ns: np.ndarray | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        self.global_frame_60 = _as_array(self.global_frame_60, np.int64)
        self.atomic_logits = _as_array(self.atomic_logits, np.float64)
        self.boundary_probability = _as_array(
            self.boundary_probability, np.float64
        )
        if self.global_frame_60.ndim != 1:
            raise ValueError("global_frame_60 must be one-dimensional")
        steps = self.global_frame_60.size
        if self.atomic_logits.ndim != 2 or self.atomic_logits.shape[0] != steps:
            raise ValueError(
                "atomic_logits must be [T,C] and align with global_frame_60"
            )
        if self.boundary_probability.shape != (steps,):
            raise ValueError("boundary_probability must be [T]")
        if steps == 0:
            raise ValueError("A prediction window cannot be empty")
        if np.any(np.diff(self.global_frame_60) <= 0):
            raise ValueError("global_frame_60 must be strictly increasing")
        if not np.isfinite(self.atomic_logits).all():
            raise ValueError("atomic_logits contains non-finite values")
        if not np.isfinite(self.boundary_probability).all():
            raise ValueError("boundary_probability contains non-finite values")
        if np.any((self.boundary_probability < 0) | (self.boundary_probability > 1)):
            raise ValueError("boundary_probability must be in [0,1]")

        if self.core_mask is None:
            self.core_mask = np.ones(steps, dtype=np.bool_)
        else:
            self.core_mask = _as_array(self.core_mask, np.bool_)
            if self.core_mask.shape != (steps,):
                raise ValueError("core_mask must be [T]")
        if not self.core_mask.any():
            raise ValueError("core_mask excludes every tick in the window")

        if self.time_ns is not None:
            self.time_ns = _as_array(self.time_ns, np.int64)
            if self.time_ns.shape != (steps,):
                raise ValueError("time_ns must be [T]")
            if np.any(np.diff(self.time_ns) <= 0):
                raise ValueError("time_ns must be strictly increasing")


@dataclass(frozen=True)
class StitchedEpisode:
    episode_id: str
    global_frame_60: np.ndarray
    atomic_logits: np.ndarray
    boundary_probability: np.ndarray
    coverage_mask: np.ndarray
    contribution_count: np.ndarray
    weight_sum: np.ndarray
    time_ns: np.ndarray | None = None

    def __post_init__(self) -> None:
        ticks = _as_array(self.global_frame_60, np.int64)
        logits = _as_array(self.atomic_logits, np.float32)
        boundary = _as_array(self.boundary_probability, np.float32)
        coverage = _as_array(self.coverage_mask, np.bool_)
        counts = _as_array(self.contribution_count, np.int64)
        weights = _as_array(self.weight_sum, np.float32)
        steps = ticks.size
        if not self.episode_id or ticks.ndim != 1 or steps == 0:
            raise ValueError("StitchedEpisode requires an episode_id and non-empty [T] ticks")
        if np.any(np.diff(ticks) <= 0):
            raise ValueError("StitchedEpisode ticks must be strictly increasing")
        if logits.ndim != 2 or logits.shape[0] != steps or logits.shape[1] == 0:
            raise ValueError("StitchedEpisode atomic_logits must be [T,C]")
        for name, value in (
            ("boundary_probability", boundary),
            ("coverage_mask", coverage),
            ("contribution_count", counts),
            ("weight_sum", weights),
        ):
            if value.shape != (steps,):
                raise ValueError(f"StitchedEpisode {name} must be [T]")
        if not np.isfinite(logits).all() or not np.isfinite(boundary).all():
            raise ValueError("StitchedEpisode predictions must be finite")
        if np.any((boundary < 0) | (boundary > 1)):
            raise ValueError("StitchedEpisode boundary probabilities must be in [0,1]")
        if np.any(counts < 0) or np.any(weights < 0):
            raise ValueError("StitchedEpisode counts/weights cannot be negative")
        if np.any(coverage != (weights > 0)):
            raise ValueError("StitchedEpisode coverage_mask must equal weight_sum > 0")
        if np.any((counts == 0) != ~coverage):
            raise ValueError("StitchedEpisode contribution_count disagrees with coverage")

        object.__setattr__(self, "global_frame_60", ticks)
        object.__setattr__(self, "atomic_logits", logits)
        object.__setattr__(self, "boundary_probability", boundary)
        object.__setattr__(self, "coverage_mask", coverage)
        object.__setattr__(self, "contribution_count", counts)
        object.__setattr__(self, "weight_sum", weights)
        if self.time_ns is not None:
            timestamps = _as_array(self.time_ns, np.int64)
            if timestamps.shape != (steps,) or np.any(np.diff(timestamps) <= 0):
                raise ValueError("StitchedEpisode time_ns must be strictly increasing [T]")
            object.__setattr__(self, "time_ns", timestamps)

    @property
    def num_ticks(self) -> int:
        return int(self.global_frame_60.size)


@dataclass(frozen=True)
class Segment:
    """One half-open action instance.

    ``start_index``/``end_index`` index the dense episode arrays, while
    ``start_tick``/``end_tick`` retain the original episode-level clock.
    Adjacent segments are allowed to carry the same label.
    """

    start_index: int
    end_index: int
    start_tick: int
    end_tick: int
    label: int
    confidence: float
    run_id: int
    instance_id: int | None = None
    boundary_score: float | None = None
    boundary_source: str | None = None

    def __post_init__(self) -> None:
        if self.start_index < 0 or self.end_index <= self.start_index:
            raise ValueError("Segment indices must define a non-empty half-open interval")
        if self.end_tick <= self.start_tick:
            raise ValueError("Segment ticks must define a non-empty half-open interval")
        if self.run_id < 0:
            raise ValueError("Segment run_id cannot be negative")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Segment confidence must be finite in [0,1]")
        if self.boundary_score is not None and (
            not np.isfinite(self.boundary_score)
            or not 0.0 <= self.boundary_score <= 1.0
        ):
            raise ValueError("Segment boundary_score must be finite in [0,1]")

    @property
    def length(self) -> int:
        return self.end_index - self.start_index

    def to_dict(self) -> Dict[str, object]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "label": self.label,
            "confidence": self.confidence,
            "run_id": self.run_id,
            "instance_id": self.instance_id,
            "boundary_score": self.boundary_score,
            "boundary_source": self.boundary_source,
        }


def _window_weights(mask: np.ndarray, mode: str) -> np.ndarray:
    weights = np.zeros(mask.shape, dtype=np.float64)
    if mode == "uniform":
        weights[mask] = 1.0
        return weights
    if mode != "triangular":
        raise ValueError(f"Unsupported stitch weighting: {mode}")

    for start, end in _valid_runs(mask):
        length = end - start
        if length == 1:
            weights[start] = 1.0
            continue
        position = np.arange(length, dtype=np.float64)
        # Positive endpoints avoid a gap when adjacent cores only touch.
        distance = np.minimum(position + 1.0, length - position)
        weights[start:end] = distance / distance.max()
    return weights


def stitch_episode_windows(
    windows: Sequence[WindowPrediction],
    *,
    weighting: str = "uniform",
    require_full_coverage: bool = True,
) -> StitchedEpisode:
    """Average overlapping core predictions on the episode 60 Hz clock."""

    if not windows:
        raise ValueError("Cannot stitch an empty window list")
    episode_id = windows[0].episode_id
    class_count = windows[0].atomic_logits.shape[1]
    if any(window.episode_id != episode_id for window in windows):
        raise ValueError("All windows passed to stitch_episode_windows must share episode_id")
    if any(window.atomic_logits.shape[1] != class_count for window in windows):
        raise ValueError("All windows must use the same action-class order")

    contributing_ticks = np.concatenate(
        [window.global_frame_60[window.core_mask] for window in windows]
    )
    first_tick = int(contributing_ticks.min())
    last_tick = int(contributing_ticks.max())
    dense_ticks = np.arange(first_tick, last_tick + 1, dtype=np.int64)
    steps = dense_ticks.size

    action_sum = np.zeros((steps, class_count), dtype=np.float64)
    boundary_sum = np.zeros(steps, dtype=np.float64)
    weight_sum = np.zeros(steps, dtype=np.float64)
    contribution_count = np.zeros(steps, dtype=np.int64)
    dense_time = np.full(steps, np.iinfo(np.int64).min, dtype=np.int64)
    has_any_time = any(window.time_ns is not None for window in windows)

    for window in windows:
        local_weights = _window_weights(window.core_mask, weighting)
        used = local_weights > 0
        positions = window.global_frame_60[used] - first_tick
        if np.any((positions < 0) | (positions >= steps)):
            raise AssertionError("Internal stitch index error")
        weights = local_weights[used]
        np.add.at(action_sum, positions, window.atomic_logits[used] * weights[:, None])
        np.add.at(
            boundary_sum,
            positions,
            window.boundary_probability[used] * weights,
        )
        np.add.at(weight_sum, positions, weights)
        np.add.at(contribution_count, positions, 1)

        if window.time_ns is not None:
            for position, timestamp in zip(positions.tolist(), window.time_ns[used].tolist()):
                existing = int(dense_time[position])
                if existing != np.iinfo(np.int64).min and existing != int(timestamp):
                    raise ValueError(
                        f"Conflicting time_ns for {episode_id} tick "
                        f"{int(dense_ticks[position])}: {existing} vs {int(timestamp)}"
                    )
                dense_time[position] = int(timestamp)

    coverage = weight_sum > 0
    if require_full_coverage and not coverage.all():
        missing = dense_ticks[~coverage]
        preview = ",".join(str(int(value)) for value in missing[:8])
        raise ValueError(
            f"Stitched episode {episode_id} has {missing.size} uncovered ticks "
            f"between {first_tick} and {last_tick}; first missing: {preview}"
        )

    action = np.zeros_like(action_sum, dtype=np.float32)
    boundary = np.zeros_like(boundary_sum, dtype=np.float32)
    action[coverage] = (action_sum[coverage] / weight_sum[coverage, None]).astype(
        np.float32
    )
    boundary[coverage] = (boundary_sum[coverage] / weight_sum[coverage]).astype(
        np.float32
    )

    time_ns: np.ndarray | None = None
    if has_any_time:
        missing_time = coverage & (dense_time == np.iinfo(np.int64).min)
        if missing_time.any():
            raise ValueError(
                f"Some covered ticks in {episode_id} have no time_ns while other windows do"
            )
        # A dense timestamp array cannot truthfully represent uncovered holes.
        # Formal evaluation rejects such holes; partial debug stitching keeps the
        # coverage mask and omits episode-level timestamps instead of inventing them.
        if coverage.all():
            time_ns = dense_time

    return StitchedEpisode(
        episode_id=episode_id,
        global_frame_60=dense_ticks,
        atomic_logits=action,
        boundary_probability=boundary,
        coverage_mask=coverage,
        contribution_count=contribution_count,
        weight_sum=weight_sum.astype(np.float32),
        time_ns=time_ns,
    )


def stitch_window_predictions(
    windows: Sequence[WindowPrediction],
    *,
    weighting: str = "uniform",
    require_full_coverage: bool = True,
) -> Dict[str, StitchedEpisode]:
    grouped: Dict[str, List[WindowPrediction]] = {}
    for window in windows:
        grouped.setdefault(window.episode_id, []).append(window)
    return {
        episode_id: stitch_episode_windows(
            episode_windows,
            weighting=weighting,
            require_full_coverage=require_full_coverage,
        )
        for episode_id, episode_windows in sorted(grouped.items())
    }


def detect_boundary_peaks(
    probability: np.ndarray | Sequence[float],
    *,
    threshold: float,
    min_distance: int,
    valid_mask: np.ndarray | Sequence[bool] | None = None,
) -> List[int]:
    """Return deterministic local maxima after score-ordered NMS."""

    scores = _as_array(probability, np.float64)
    if scores.ndim != 1:
        raise ValueError("Boundary probability must be one-dimensional")
    if min_distance < 1:
        raise ValueError("min_distance must be at least one tick")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if valid_mask is None:
        valid = np.ones(scores.shape, dtype=np.bool_)
    else:
        valid = _as_array(valid_mask, np.bool_)
        if valid.shape != scores.shape:
            raise ValueError("valid_mask must match boundary probability")

    selected: List[int] = []
    for start, end in _valid_runs(valid):
        candidates: List[int] = []
        # A boundary marks the start of a new instance, so run starts are not
        # internal boundary candidates.
        for index in range(start + 1, end):
            left = scores[index - 1]
            right = scores[index + 1] if index + 1 < end else -np.inf
            if scores[index] >= threshold and scores[index] >= left and scores[index] >= right:
                candidates.append(index)
        # Invalid gaps define independent evaluation runs. NMS must not let a
        # peak on one side of a masked gap suppress a peak on the other side.
        candidates.sort(key=lambda index: (-float(scores[index]), index))
        run_selected: List[int] = []
        for candidate in candidates:
            if all(abs(candidate - kept) >= min_distance for kept in run_selected):
                run_selected.append(candidate)
        selected.extend(run_selected)
    return sorted(selected)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.maximum(exponential.sum(axis=-1, keepdims=True), 1.0e-12)


def boundary_aware_decode(
    atomic_logits: np.ndarray | Sequence[Sequence[float]],
    boundary_probability: np.ndarray | Sequence[float],
    *,
    threshold: float = 0.5,
    min_distance: int = 6,
    min_segment_frames: int = 1,
    valid_mask: np.ndarray | Sequence[bool] | None = None,
    global_frame_60: np.ndarray | Sequence[int] | None = None,
    use_boundary_head: bool = True,
    include_action_changes: bool = True,
) -> List[Segment]:
    """Decode action instances without merging adjacent equal-label segments.

    Boundary peaks and, by default, per-tick action-class changes both create
    candidate segment starts. Boundary peaks have deterministic priority, then
    nearby action-change candidates are suppressed using the same frozen
    distance; candidates that would create a segment shorter than
    ``min_segment_frames`` are removed. A segment label is selected from its mean action
    logits.  A boundary peak inside one semantic action therefore yields two
    distinct, adjacent same-label instances in the raw view.  Setting
    ``use_boundary_head=False`` is the controlled ``-boundary head`` ablation:
    boundary probabilities are treated as zero and same-label boundaries cannot
    be recovered from action changes alone.
    """

    logits = _as_array(atomic_logits, np.float64)
    boundary = _as_array(boundary_probability, np.float64)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("atomic_logits must be non-empty [T,C]")
    steps = logits.shape[0]
    if boundary.shape != (steps,):
        raise ValueError("boundary_probability must be [T]")
    if min_distance < 1:
        raise ValueError("min_distance must be positive")
    if min_segment_frames < 1:
        raise ValueError("min_segment_frames must be positive")
    if valid_mask is None:
        valid = np.ones(steps, dtype=np.bool_)
    else:
        valid = _as_array(valid_mask, np.bool_)
        if valid.shape != (steps,):
            raise ValueError("valid_mask must be [T]")
    if global_frame_60 is None:
        ticks = np.arange(steps, dtype=np.int64)
    else:
        ticks = _as_array(global_frame_60, np.int64)
        if ticks.shape != (steps,) or np.any(np.diff(ticks) <= 0):
            raise ValueError("global_frame_60 must be strictly increasing [T]")

    if use_boundary_head:
        peaks = set(
            detect_boundary_peaks(
                boundary,
                threshold=threshold,
                min_distance=min_distance,
                valid_mask=valid,
            )
        )
    else:
        boundary = np.zeros_like(boundary)
        peaks = set()
    tick_labels = logits.argmax(axis=1)
    segments: List[Segment] = []
    run_id = 0
    probabilities = _softmax(logits)

    for run_start, run_end in _valid_runs(valid):
        changes = {
            index
            for index in range(run_start + 1, run_end)
            if tick_labels[index] != tick_labels[index - 1]
        }
        required_spacing = max(min_distance, min_segment_frames)
        peak_candidates = [
            index
            for index in peaks
            if run_start < index < run_end
            and index - run_start >= min_segment_frames
            and run_end - index >= min_segment_frames
        ]
        # Peaks are the only evidence for adjacent equal-label instances, so
        # they win deterministic cross-source conflicts. Re-apply spacing when
        # min_segment_frames is stricter than peak NMS.
        peak_candidates.sort(key=lambda index: (-float(boundary[index]), index))
        selected_peaks: List[int] = []
        for candidate in peak_candidates:
            if all(
                abs(candidate - kept) >= required_spacing
                for kept in selected_peaks
            ):
                selected_peaks.append(candidate)
        internal = set(selected_peaks)
        if include_action_changes:
            for candidate in sorted(changes):
                if (
                    candidate - run_start < min_segment_frames
                    or run_end - candidate < min_segment_frames
                ):
                    continue
                if all(
                    abs(candidate - kept) >= required_spacing
                    for kept in internal
                ):
                    internal.add(candidate)
        starts = [run_start, *sorted(internal)]
        ends = [*sorted(internal), run_end]

        for start, end in zip(starts, ends):
            mean_logits = logits[start:end].mean(axis=0)
            label = int(mean_logits.argmax())
            confidence = float(probabilities[start:end, label].mean())
            source: str | None = None
            score: float | None = None
            if start != run_start:
                in_peak = start in peaks
                in_change = start in changes
                source = (
                    "boundary_peak+action_change"
                    if in_peak and in_change
                    else "boundary_peak"
                    if in_peak
                    else "action_change"
                )
                score = float(boundary[start])
            segments.append(
                Segment(
                    start_index=start,
                    end_index=end,
                    start_tick=int(ticks[start]),
                    end_tick=int(ticks[end - 1]) + 1,
                    label=label,
                    confidence=confidence,
                    run_id=run_id,
                    boundary_score=score,
                    boundary_source=source,
                )
            )
        run_id += 1
    return segments


def collapse_adjacent_same_label(segments: Sequence[Segment]) -> List[Segment]:
    """Create the literature view while leaving the raw list untouched."""

    collapsed: List[Segment] = []
    for segment in segments:
        if (
            collapsed
            and collapsed[-1].run_id == segment.run_id
            and collapsed[-1].end_index == segment.start_index
            and collapsed[-1].label == segment.label
        ):
            previous = collapsed[-1]
            total = previous.length + segment.length
            confidence = (
                previous.confidence * previous.length
                + segment.confidence * segment.length
            ) / max(total, 1)
            collapsed[-1] = replace(
                previous,
                end_index=segment.end_index,
                end_tick=segment.end_tick,
                confidence=float(confidence),
            )
        else:
            collapsed.append(segment)
    return collapsed


def segments_to_frame_labels(
    segments: Sequence[Segment], steps: int, *, ignore_index: int = -100
) -> np.ndarray:
    labels = np.full(steps, ignore_index, dtype=np.int64)
    for segment in segments:
        if not 0 <= segment.start_index < segment.end_index <= steps:
            raise ValueError("Segment lies outside requested episode length")
        labels[segment.start_index : segment.end_index] = segment.label
    return labels


def decoded_boundary_types(segments: Sequence[Segment]) -> Dict[int, str]:
    """Map internal predicted boundary ticks to protocol boundary types."""

    output: Dict[int, str] = {}
    for previous, current in zip(segments, segments[1:]):
        if previous.run_id != current.run_id or previous.end_index != current.start_index:
            continue
        output[current.start_tick] = (
            BOUNDARY_CLASS_CHANGE
            if previous.label != current.label
            else BOUNDARY_SAME_LABEL
        )
    return output


__all__ = [
    "BOUNDARY_CLASS_CHANGE",
    "BOUNDARY_SAME_LABEL",
    "Segment",
    "StitchedEpisode",
    "WindowPrediction",
    "boundary_aware_decode",
    "collapse_adjacent_same_label",
    "decoded_boundary_types",
    "detect_boundary_peaks",
    "segments_to_frame_labels",
    "stitch_episode_windows",
    "stitch_window_predictions",
]
