"""Post-test extension; count mathematics copied verbatim from corrected production statistics.
Original confirmatory bootstrap and original executor are not imported or modified.
"""
from __future__ import annotations
import hashlib, json, math
from collections.abc import Mapping, Sequence
from typing import Any
from pathlib import Path
import numpy as np
SPEC=json.loads((Path(__file__).parent/'PLAN_SPEC.json').read_text())
SCHEMA_VERSION='aspnet_supplement_group_counts_v1'
EXPECTED_VARIANTS=tuple(sorted({m['variant'] for m in SPEC['models']}))
EXPECTED_SEEDS=(20260824,20260825,20260826)
VISUAL_VARIANTS=set(SPEC['visual_variants'])
VISUAL_CONDITIONS=tuple(c['id'] for c in SPEC['conditions'])
SEGMENT_VIEWS=('raw','collapsed')
SEGMENT_THRESHOLDS=(10,25,50)
BOUNDARY_TYPES=('all','class_change','same_label')
BOUNDARY_TOLERANCES_MS=(250,500,1000)
RATE_SENSITIVITY_HZ=(30,15,10,5,2)
def expected_condition_cells(): return tuple(tuple(x) for x in SPEC['cells'])

class StatisticsContractError(ValueError):
    """Raised when a terminal cell or sufficient statistic is malformed."""

def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def with_payload_sha256(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    output.pop("payload_sha256", None)
    output["payload_sha256"] = canonical_sha256(output)
    return output

def _count_record(tp: int, fp: int, fn: int) -> dict[str, int]:
    values = {"tp": int(tp), "fp": int(fp), "fn": int(fn)}
    if any(value < 0 for value in values.values()):
        raise StatisticsContractError("metric counts cannot be negative")
    return values

def _f1(counts: Mapping[str, Any]) -> float:
    tp, fp, fn = (int(counts[name]) for name in ("tp", "fp", "fn"))
    denominator = 2 * tp + fp + fn
    return 200.0 * tp / denominator if denominator else 0.0

def _prf(counts: Mapping[str, Any]) -> dict[str, float | int]:
    tp, fp, fn = (int(counts[name]) for name in ("tp", "fp", "fn"))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": 100.0 * tp / (tp + fp) if tp + fp else 0.0,
        "recall": 100.0 * tp / (tp + fn) if tp + fn else 0.0,
        "f1": _f1(counts),
    }

def build_group_statistics(
    evaluations: Sequence[Any],
    *,
    group_id: str,
    variant: str,
    seed: int,
    condition: str,
    num_actions: int,
    semantic_output_rate_hz: int = 30,
) -> dict[str, Any]:
    """Reduce episode evaluations from one frozen group to sufficient counts.

    ``evaluations`` must contain the evaluator's ``EpisodeEvaluation`` values.
    No per-episode metric is averaged except edit similarity, whose dataset
    definition is explicitly an unweighted episode mean.  F1 values are never
    averaged; their TP/FP/FN counts are retained and recomputed after resampling.
    """

    if not evaluations:
        raise StatisticsContractError("a formal split group cannot be empty")
    if not isinstance(group_id, str) or not group_id:
        raise StatisticsContractError("group_id must be a non-empty string")
    if variant not in EXPECTED_VARIANTS or seed not in EXPECTED_SEEDS:
        raise StatisticsContractError("unknown frozen variant/seed")
    if (variant, seed, condition) not in expected_condition_cells():
        raise StatisticsContractError("outside explicitly authorized extension")
    if num_actions not in {8, 26}:
        raise StatisticsContractError("Formal500h H0 supports only 8 or 26 actions")
    if semantic_output_rate_hz not in RATE_SENSITIVITY_HZ:
        raise StatisticsContractError("unknown frozen semantic output rate")

    confusion = np.zeros((num_actions, num_actions), dtype=np.int64)
    segment = {
        f"{view}@{threshold}": {"tp": 0, "fp": 0, "fn": 0}
        for view in SEGMENT_VIEWS
        for threshold in SEGMENT_THRESHOLDS
    }
    boundary = {
        f"{kind}@{tolerance}ms": {"tp": 0, "fp": 0, "fn": 0}
        for kind in BOUNDARY_TYPES
        for tolerance in BOUNDARY_TOLERANCES_MS
    }
    raw_edit_sum = 0.0
    collapsed_edit_sum = 0.0
    evaluated_ticks = 0
    boundary_eval_ticks = 0
    episode_ids: list[str] = []

    for evaluation in evaluations:
        episode_id = getattr(evaluation, "episode_id", None)
        if not isinstance(episode_id, str) or not episode_id:
            raise StatisticsContractError("evaluation lacks a valid episode_id")
        episode_ids.append(episode_id)
        matrix = np.asarray(getattr(evaluation, "confusion", None), dtype=np.int64)
        if matrix.shape != (num_actions, num_actions) or np.any(matrix < 0):
            raise StatisticsContractError("episode confusion matrix is invalid")
        confusion += matrix
        for view in SEGMENT_VIEWS:
            for threshold in SEGMENT_THRESHOLDS:
                key = (view, threshold / 100.0)
                try:
                    counts = evaluation.segment_counts[key]
                except (AttributeError, KeyError) as error:
                    raise StatisticsContractError(f"missing segment counts {key}") from error
                target = segment[f"{view}@{threshold}"]
                for name, value in zip(("tp", "fp", "fn"), counts):
                    if int(value) < 0:
                        raise StatisticsContractError("negative segment count")
                    target[name] += int(value)
        for kind in BOUNDARY_TYPES:
            for tolerance in BOUNDARY_TOLERANCES_MS:
                # Preserve the frozen output schema's "same_label" name;
                # the real evaluator calls this type "same_label_repeat".
                input_kind = "same_label_repeat" if kind == "same_label" else kind
                key = (input_kind, tolerance)
                try:
                    supplied = evaluation.boundary_counts
                    legacy_key = (kind, tolerance)
                    if key != legacy_key and legacy_key in supplied:
                        # Accept the original data-source-agnostic fixture alias,
                        # but never silently choose between conflicting counts.
                        if key in supplied and tuple(supplied[key]) != tuple(supplied[legacy_key]):
                            raise StatisticsContractError(
                                f"conflicting boundary aliases {key} and {legacy_key}"
                            )
                        if key not in supplied:
                            key = legacy_key
                    counts = supplied[key]
                except (AttributeError, KeyError) as error:
                    raise StatisticsContractError(f"missing boundary counts {key}") from error
                target = boundary[f"{kind}@{tolerance}ms"]
                for name, value in zip(("tp", "fp", "fn"), counts):
                    if int(value) < 0:
                        raise StatisticsContractError("negative boundary count")
                    target[name] += int(value)
        raw_edit_sum += float(evaluation.raw_edit)
        collapsed_edit_sum += float(evaluation.collapsed_edit)
        evaluated_ticks += int(evaluation.evaluated_ticks)
        boundary_eval_ticks += int(evaluation.boundary_eval_ticks)

    if len(set(episode_ids)) != len(episode_ids):
        raise StatisticsContractError("duplicate episode within one group statistic")
    if not all(math.isfinite(v) for v in (raw_edit_sum, collapsed_edit_sum)):
        raise StatisticsContractError("edit statistics are non-finite")

    return with_payload_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "variant": variant,
            "seed": seed,
            "condition": condition,
            "group_id": group_id,
            "num_actions": num_actions,
            "semantic_output_rate_hz": semantic_output_rate_hz,
            "evaluation_timeline_rate_hz": 30,
            "episode_count": len(evaluations),
            "episode_ids_sha256": canonical_sha256(sorted(episode_ids)),
            "evaluated_ticks": evaluated_ticks,
            "boundary_eval_ticks": boundary_eval_ticks,
            "confusion": confusion.tolist(),
            "segment_counts": segment,
            "boundary_counts": boundary,
            "edit_sums": {
                "raw": raw_edit_sum,
                "collapsed": collapsed_edit_sum,
            },
        }
    )

def validate_group_statistics(
    value: Mapping[str, Any],
    *,
    expected_cell: tuple[str, int, str] | None = None,
    expected_rate_hz: int | None = None,
) -> None:
    required = {
        "schema_version", "status", "variant", "seed", "condition", "group_id",
        "num_actions", "episode_count", "episode_ids_sha256", "evaluated_ticks",
        "boundary_eval_ticks", "confusion", "segment_counts", "boundary_counts",
        "edit_sums", "semantic_output_rate_hz", "evaluation_timeline_rate_hz",
        "payload_sha256",
    }
    if set(value) != required:
        raise StatisticsContractError("group statistic key set changed")
    payload = dict(value)
    recorded = payload.pop("payload_sha256", None)
    if recorded != canonical_sha256(payload):
        raise StatisticsContractError("group statistic payload SHA-256 mismatch")
    cell = (value["variant"], value["seed"], value["condition"])
    if value["schema_version"] != SCHEMA_VERSION or value["status"] != "complete":
        raise StatisticsContractError("group statistic schema/status changed")
    if cell not in expected_condition_cells():
        raise StatisticsContractError("group statistic is outside extension cells")
    if expected_cell is not None and cell != expected_cell:
        raise StatisticsContractError("group statistic cell identity mismatch")
    num_actions = value["num_actions"]
    expected_actions = 8 if value["variant"].startswith("DIRECT8_") else 26
    if num_actions != expected_actions:
        raise StatisticsContractError("group statistic label dimension changed")
    rate_hz = value["semantic_output_rate_hz"]
    if rate_hz not in RATE_SENSITIVITY_HZ:
        raise StatisticsContractError("group statistic output rate changed")
    if expected_rate_hz is not None and rate_hz != expected_rate_hz:
        raise StatisticsContractError("group statistic output-rate identity mismatch")
    if value["evaluation_timeline_rate_hz"] != 30:
        raise StatisticsContractError("rate sensitivity must use untouched 30-Hz truth")
    matrix = np.asarray(value["confusion"])
    if matrix.shape != (num_actions, num_actions) or not np.issubdtype(matrix.dtype, np.integer):
        raise StatisticsContractError("group statistic confusion shape/type changed")
    if np.any(matrix < 0):
        raise StatisticsContractError("negative confusion count")
    expected_segment = {
        f"{view}@{threshold}" for view in SEGMENT_VIEWS for threshold in SEGMENT_THRESHOLDS
    }
    expected_boundary = {
        f"{kind}@{tol}ms" for kind in BOUNDARY_TYPES for tol in BOUNDARY_TOLERANCES_MS
    }
    if set(value["segment_counts"]) != expected_segment:
        raise StatisticsContractError("segment sufficient-statistic set changed")
    if set(value["boundary_counts"]) != expected_boundary:
        raise StatisticsContractError("boundary sufficient-statistic set changed")
    for record in (*value["segment_counts"].values(), *value["boundary_counts"].values()):
        if set(record) != {"tp", "fp", "fn"} or any(
            isinstance(record[name], bool) or not isinstance(record[name], int) or record[name] < 0
            for name in ("tp", "fp", "fn")
        ):
            raise StatisticsContractError("invalid TP/FP/FN sufficient statistic")
    if (
        isinstance(value["episode_count"], bool)
        or not isinstance(value["episode_count"], int)
        or value["episode_count"] <= 0
        or isinstance(value["evaluated_ticks"], bool)
        or not isinstance(value["evaluated_ticks"], int)
        or value["evaluated_ticks"] <= 0
        or isinstance(value["boundary_eval_ticks"], bool)
        or not isinstance(value["boundary_eval_ticks"], int)
        or value["boundary_eval_ticks"] <= 0
    ):
        raise StatisticsContractError("invalid group support")
    if int(matrix.sum()) != value["evaluated_ticks"]:
        raise StatisticsContractError("confusion support does not match evaluated ticks")
    if value["boundary_eval_ticks"] > value["evaluated_ticks"]:
        raise StatisticsContractError("boundary support exceeds evaluated ticks")
    if (
        not isinstance(value["episode_ids_sha256"], str)
        or len(value["episode_ids_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["episode_ids_sha256"])
    ):
        raise StatisticsContractError("invalid episode-id SHA-256")
    if set(value["edit_sums"]) != {"raw", "collapsed"} or any(
        isinstance(value["edit_sums"][name], bool)
        or not isinstance(value["edit_sums"][name], (int, float))
        or not math.isfinite(float(value["edit_sums"][name]))
        for name in ("raw", "collapsed")
    ):
        raise StatisticsContractError("invalid edit sufficient statistic")

def _merge_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise StatisticsContractError("cannot aggregate an empty group sample")
    for record in records:
        validate_group_statistics(record)
    num_actions = int(records[0]["num_actions"])
    cell = tuple(records[0][name] for name in ("variant", "seed", "condition"))
    semantic_output_rate_hz = int(records[0]["semantic_output_rate_hz"])
    if any(
        int(row["num_actions"]) != num_actions
        or int(row["semantic_output_rate_hz"]) != semantic_output_rate_hz
        or tuple(row[name] for name in ("variant", "seed", "condition")) != cell
        for row in records
    ):
        raise StatisticsContractError(
            "mixed cells, output rates, or label spaces in group aggregation"
        )
    confusion = sum(
        (np.asarray(row["confusion"], dtype=np.int64) for row in records),
        start=np.zeros((num_actions, num_actions), dtype=np.int64),
    )
    segment: dict[str, dict[str, int]] = {}
    boundary: dict[str, dict[str, int]] = {}
    for key in records[0]["segment_counts"]:
        segment[key] = {
            name: sum(int(row["segment_counts"][key][name]) for row in records)
            for name in ("tp", "fp", "fn")
        }
    for key in records[0]["boundary_counts"]:
        boundary[key] = {
            name: sum(int(row["boundary_counts"][key][name]) for row in records)
            for name in ("tp", "fp", "fn")
        }
    episodes = sum(int(row["episode_count"]) for row in records)
    total_ticks = int(confusion.sum())
    support = confusion.sum(axis=1)
    class_metrics = []
    for label in range(num_actions):
        tp = int(confusion[label, label])
        counts = _count_record(
            tp,
            int(confusion[:, label].sum()) - tp,
            int(confusion[label, :].sum()) - tp,
        )
        class_metrics.append(
            {
                "class_index": label,
                "support": int(support[label]),
                **_prf(counts),
            }
        )
    supported_class_f1 = [
        float(row["f1"]) for row in class_metrics if int(row["support"]) > 0
    ]
    collapsed_f1 = _f1(segment["collapsed@50"])
    boundary_f1 = _f1(boundary["class_change@250ms"])
    return {
        "variant": cell[0],
        "seed": cell[1],
        "condition": cell[2],
        "num_actions": num_actions,
        "semantic_output_rate_hz": semantic_output_rate_hz,
        "evaluation_timeline_rate_hz": 30,
        "sampled_group_occurrences": len(records),
        "episode_occurrences": episodes,
        "evaluated_ticks": sum(int(row["evaluated_ticks"]) for row in records),
        "frame_accuracy": 100.0 * float(np.trace(confusion)) / max(total_ticks, 1),
        "tick_macro_f1": (
            float(np.mean(supported_class_f1)) if supported_class_f1 else 0.0
        ),
        "per_class": class_metrics,
        "raw_edit": sum(float(row["edit_sums"]["raw"]) for row in records) / episodes,
        "collapsed_edit": sum(float(row["edit_sums"]["collapsed"]) for row in records) / episodes,
        "raw_f1@10": _f1(segment["raw@10"]),
        "raw_f1@25": _f1(segment["raw@25"]),
        "raw_f1@50": _f1(segment["raw@50"]),
        "collapsed_f1@10": _f1(segment["collapsed@10"]),
        "collapsed_f1@25": _f1(segment["collapsed@25"]),
        "collapsed_f1@50": collapsed_f1,
        "class_change_boundary_f1@250ms": boundary_f1,
        "class_change_boundary_f1@500ms": _f1(boundary["class_change@500ms"]),
        "class_change_boundary_f1@1000ms": _f1(boundary["class_change@1000ms"]),
        "raw_boundary_f1@250ms": _f1(boundary["all@250ms"]),
        "raw_boundary_f1@500ms": _f1(boundary["all@500ms"]),
        "raw_boundary_f1@1000ms": _f1(boundary["all@1000ms"]),
        "m2r2_style_adapted_dr_all_boundaries@1000ms": {
            "official_m2r2_reproduction": False,
            "match_scope": "episode_and_contiguous_evaluable_run",
            **_prf(boundary["all@1000ms"]),
        },
        "semantic_segment_count": sum(
            int(segment["collapsed@10"][name]) for name in ("tp", "fp")
        ),
        "target_collapsed_segment_count": sum(
            int(segment["collapsed@10"][name]) for name in ("tp", "fn")
        ),
        "primary_composite": 0.5 * collapsed_f1 + 0.5 * boundary_f1,
    }

def _aggregate_rate_statistics(
    groups: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return only metrics valid for the post-hoc semantic-rate proxy.

    Holding decoded semantic labels cannot preserve same-label instance
    boundaries.  Consequently the secondary rate table intentionally omits
    raw-instance and all/same-label-boundary metrics.
    """

    full = aggregate_group_statistics(groups)
    return {
        "variant": full["variant"],
        "seed": full["seed"],
        "condition": full["condition"],
        "num_actions": full["num_actions"],
        "semantic_output_rate_hz": full["semantic_output_rate_hz"],
        "evaluation_timeline_rate_hz": full["evaluation_timeline_rate_hz"],
        "frame_accuracy": full["frame_accuracy"],
        "tick_macro_f1": full["tick_macro_f1"],
        "per_class": full["per_class"],
        "collapsed_edit": full["collapsed_edit"],
        "collapsed_f1@10": full["collapsed_f1@10"],
        "collapsed_f1@25": full["collapsed_f1@25"],
        "collapsed_f1@50": full["collapsed_f1@50"],
        "class_change_boundary_f1@250ms": full[
            "class_change_boundary_f1@250ms"
        ],
        "class_change_boundary_f1@500ms": full[
            "class_change_boundary_f1@500ms"
        ],
        "class_change_boundary_f1@1000ms": full[
            "class_change_boundary_f1@1000ms"
        ],
        "semantic_segment_count": full["semantic_segment_count"],
        "target_collapsed_segment_count": full["target_collapsed_segment_count"],
        "primary_composite": full["primary_composite"],
        "same_label_instance_boundary_metrics_reportable": False,
        "m2r2_style_adapted_dr_reportable": False,
    }

def aggregate_group_statistics(
    groups: Mapping[str, Mapping[str, Any]],
    sampled_group_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not groups:
        raise StatisticsContractError("condition cell has no group statistics")
    ordered = tuple(sorted(groups))
    sample = ordered if sampled_group_ids is None else tuple(sampled_group_ids)
    if not sample or any(group_id not in groups for group_id in sample):
        raise StatisticsContractError("bootstrap sample contains an unknown group")
    return _merge_records([groups[group_id] for group_id in sample])
