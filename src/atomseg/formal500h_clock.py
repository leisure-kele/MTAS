"""Canonical source-to-output clock conversions for Formal500h.

The released HDF5 stores targets at 60 Hz while the learning/evaluation
contract is 30 Hz.  Keeping this conversion in one small, dependency-light
module prevents the training loader and evaluator from silently using
different definitions when a transition falls on the second sample of a
60-Hz pair.

Action/instance labels use the first sample in each source-rate bin, matching
the historical ``[::2]`` convention.  Event markers (boundaries) are mapped
to the output bin containing the event using a ceiling rule, then OR-ed.  A
boundary at raw index 1 therefore belongs to output tick 1 rather than being
lost in tick 0.  Callers should record this policy in their run manifest.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


SOURCE_RATE_HZ = 60
OUTPUT_RATE_HZ = 30
SOURCE_TO_OUTPUT_RATIO = SOURCE_RATE_HZ // OUTPUT_RATE_HZ
CLOCK_POLICY = "first_sample_labels_ceiling_event_bins_v1"


def _one_dim(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


def _validate_ratio(source_rate_hz: int, output_rate_hz: int) -> int:
    source = int(source_rate_hz)
    output = int(output_rate_hz)
    if source < 1 or output < 1 or source % output:
        raise ValueError("source/output rates must have a positive integer ratio")
    return source // output


def source_indices(
    source_length: int,
    *,
    source_rate_hz: int = SOURCE_RATE_HZ,
    output_rate_hz: int = OUTPUT_RATE_HZ,
) -> np.ndarray:
    """Return source indices selected for output ticks (first sample policy)."""

    length = int(source_length)
    if length < 1:
        raise ValueError("source_length must be positive")
    ratio = _validate_ratio(source_rate_hz, output_rate_hz)
    return np.arange(0, length, ratio, dtype=np.int64)


def downsample_labels(
    labels: Any,
    *,
    source_rate_hz: int = SOURCE_RATE_HZ,
    output_rate_hz: int = OUTPUT_RATE_HZ,
) -> np.ndarray:
    """Downsample frame/instance labels by selecting the first source sample."""

    values = _one_dim("labels", labels)
    return values[source_indices(
        values.size,
        source_rate_hz=source_rate_hz,
        output_rate_hz=output_rate_hz,
    )]


def downsample_event_mask(
    events: Any,
    *,
    source_rate_hz: int = SOURCE_RATE_HZ,
    output_rate_hz: int = OUTPUT_RATE_HZ,
) -> np.ndarray:
    """Map source event markers to output bins with ceiling/OR semantics."""

    values = _one_dim("events", events).astype(bool, copy=False)
    ratio = _validate_ratio(source_rate_hz, output_rate_hz)
    output_length = (values.size + ratio - 1) // ratio
    result = np.zeros(output_length, dtype=bool)
    raw_events = np.flatnonzero(values)
    if raw_events.size:
        # ``ceil(index / ratio)`` maps index 0 to 0 and an event at the
        # second source sample to the following output tick.
        # ``ceil(index / ratio)`` can map a final odd source index to
        # ``output_length`` when the source length is an exact multiple of
        # the ratio (for example, index=5, ratio=2, length=6).  The final
        # partially/fully covered output bin is still the last legal tick;
        # clip explicitly so the helper is total for both odd and even
        # source lengths.
        output_indices = np.minimum(
            (raw_events + ratio - 1) // ratio,
            output_length - 1,
        )
        result[np.unique(output_indices)] = True
    return result


def pair_or_event_mask(events: Any, *, ratio: int = 2) -> np.ndarray:
    """Reproduce the historical floor/pair-OR event conversion.

    This helper exists only for read-only audits and regression tests.  New
    training/evaluation code must use :func:`downsample_event_mask`, whose
    ceiling-bin policy is explicit in :data:`CLOCK_POLICY`.
    """

    values = _one_dim("events", events).astype(bool, copy=False)
    ratio = int(ratio)
    if ratio < 1:
        raise ValueError("ratio must be positive")
    output = np.zeros((values.size + ratio - 1) // ratio, dtype=bool)
    for index in range(output.size):
        start = index * ratio
        output[index] = bool(values[start : start + ratio].any())
    return output


def derive_change_mask(values: Any) -> np.ndarray:
    """Return a start-of-segment marker from a sampled label/instance array."""

    sampled = _one_dim("values", values)
    result = np.zeros(sampled.size, dtype=bool)
    if sampled.size > 1:
        result[1:] = sampled[1:] != sampled[:-1]
    return result


def downsample_targets_60_to_30(
    *,
    action_class_60: Any,
    action_instance_60: Any | None = None,
    boundary_instance_60: Any | None = None,
    boundary_class_change_60: Any | None = None,
) -> dict[str, np.ndarray]:
    """Convert Formal500h target streams under one explicit clock policy.

    The returned dictionary always contains ``action_class``.  When instance
    labels are supplied it also contains ``action_instance`` and a derived
    ``boundary_instance``.  Explicit boundary streams are OR-ed into the
    corresponding derived stream after ceiling-bin mapping; this preserves
    human boundary markers while keeping odd-frame transitions deterministic.
    """

    action = _one_dim("action_class_60", action_class_60)
    sampled_action = downsample_labels(action)
    result: dict[str, np.ndarray] = {"action_class": sampled_action}

    sampled_instance: np.ndarray | None = None
    if action_instance_60 is not None:
        instance = _one_dim("action_instance_60", action_instance_60)
        if instance.size != action.size:
            raise ValueError("action_instance_60 length must match action_class_60")
        sampled_instance = downsample_labels(instance)
        result["action_instance"] = sampled_instance
        result["boundary_instance"] = derive_change_mask(sampled_instance)

    if boundary_instance_60 is not None:
        boundary = _one_dim("boundary_instance_60", boundary_instance_60)
        if boundary.size != action.size:
            raise ValueError("boundary_instance_60 length must match action_class_60")
        mapped = downsample_event_mask(boundary)
        if "boundary_instance" in result:
            result["boundary_instance"] |= mapped
        else:
            result["boundary_instance"] = mapped

    if boundary_class_change_60 is not None:
        class_boundary = _one_dim(
            "boundary_class_change_60", boundary_class_change_60
        )
        if class_boundary.size != action.size:
            raise ValueError(
                "boundary_class_change_60 length must match action_class_60"
            )
        mapped = downsample_event_mask(class_boundary)
        if "boundary_class_change" in result:
            result["boundary_class_change"] |= mapped
        else:
            result["boundary_class_change"] = mapped

    if "boundary_instance" in result and "boundary_class_change" not in result:
        result["boundary_class_change"] = result["boundary_instance"] & derive_change_mask(
            result["action_class"]
        )
    return result


def clock_metadata() -> Mapping[str, Any]:
    """Return serialisable metadata that must be stored in every new run."""

    return {
        "source_rate_hz": SOURCE_RATE_HZ,
        "output_rate_hz": OUTPUT_RATE_HZ,
        "ratio": SOURCE_TO_OUTPUT_RATIO,
        "policy": CLOCK_POLICY,
    }
