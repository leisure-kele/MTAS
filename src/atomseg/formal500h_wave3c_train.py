"""Formal500h Wave-3C tactile representation/rate/boundary factorial.

The experiment is validation-only and keeps the formal test manifest sealed.
It completes a 2 x 2 x 2 factorial around the Wave-3B VT model:

* tactile representation: raw+signed-delta+absolute-delta versus raw only;
* tactile/event temporal rate: 30 Hz versus mask-aware 15 Hz pooling;
* boundary head and boundary loss: enabled versus removed.

Seven paired seeds are frozen.  The first three baseline cells reuse the exact
Wave-3B VT artifacts; four new seeds form a random-seed replication block.
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
from typing import Any, Callable, Mapping, Sequence

import torch

from .formal500h_evaluation import evaluate_dataset
from .formal500h_infra_runtime import (
    VariantAwareFormal500hDataset,
    collate_formal500h_fast,
)
from .formal500h_protocol import NUM_CLASSES, sha256_file
from .formal500h_runtime import Formal500hModel as OriginalFormal500hModel
from .formal500h_wave1_data import WAVE1_DATA_SCHEMA, WINDOW_SCHEMA
from . import formal500h_wave1_train as wave1
from . import formal500h_wave3b_vt as wave3b


TRAINING_SCHEMA = "formal500h_wave3c_training_v1"
MODEL_GATE_SCHEMA = "formal500h_wave3c_model_gate_v1"
AGGREGATE_SCHEMA = "formal500h_wave3c_factorial_aggregate_v1"
RUNTIME_SCHEMA = "formal500h_wave3c_optimized_runtime_v1"

VT_MODALITIES = ("vision", "tactile")
ORIGINAL_SEEDS = (20260824, 20260825, 20260826)
REPLICATION_SEEDS = (20260827, 20260828, 20260829, 20260830)
SEEDS = ORIGINAL_SEEDS + REPLICATION_SEEDS
BASELINE_CELL = "D30B"


@dataclass(frozen=True)
class CellSpec:
    cell: str
    tactile_representation: str
    event_rate_hz: int
    use_boundary_head: bool

    @property
    def training_variant(self) -> str:
        # Preserve the already-released Wave-3B baseline variant identity.
        return "VT" if self.cell == BASELINE_CELL else self.cell

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "tactile_representation": self.tactile_representation,
            "event_rate_hz": self.event_rate_hz,
            "use_boundary_head": self.use_boundary_head,
            "boundary_weight": 0.5 if self.use_boundary_head else 0.0,
        }


CELL_SPECS: tuple[CellSpec, ...] = (
    CellSpec("D30B", "raw_delta_abs", 30, True),
    CellSpec("R30B", "raw", 30, True),
    CellSpec("D15B", "raw_delta_abs", 15, True),
    CellSpec("R15B", "raw", 15, True),
    CellSpec("D30N", "raw_delta_abs", 30, False),
    CellSpec("R30N", "raw", 30, False),
    CellSpec("D15N", "raw_delta_abs", 15, False),
    CellSpec("R15N", "raw", 15, False),
)
CELL_BY_ID = {value.cell: value for value in CELL_SPECS}

METRICS: Mapping[str, Callable[[Mapping[str, Any]], float]] = {
    "primary_composite": lambda value: float(
        value["calibrated_validation"]["metrics"]["primary"]
        ["checkpoint_composite"]
    ),
    "collapsed_f1@50": lambda value: float(
        value["calibrated_validation"]["metrics"]["primary"]
        ["collapsed_semantic_f1@50"]
    ),
    "class_change_boundary_f1@250ms": lambda value: float(
        value["calibrated_validation"]["metrics"]["primary"]
        ["class_change_boundary_f1@250ms"]
    ),
    "raw_instance_f1@50": lambda value: float(
        value["calibrated_validation"]["metrics"]["primary"]
        ["raw_instance_f1@50"]
    ),
    "frame_accuracy": lambda value: float(
        value["calibrated_validation"]["metrics"]["frame_accuracy"]
    ),
    "tick_macro_f1": lambda value: float(
        value["calibrated_validation"]["metrics"]["tick_macro_f1"]
    ),
    "hand_macro_f1": lambda value: float(
        value["calibrated_validation"]["hand"]["macro_f1"]
    ),
}


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
        raise ValueError(f"{path} is not a JSON object")
    return value


def _release_lineage() -> dict[str, Any]:
    module = Path(__file__).resolve()
    manifest = (
        module.parents[2]
        / "configs"
        / "experimental"
        / "atomic_v1_wave3c_release_sha256.txt"
    )
    if not manifest.is_file():
        raise ValueError(f"Wave-3C release manifest is missing: {manifest}")
    sources = {
        "wave3c_trainer": module,
        "wave3b_vt": module.with_name("formal500h_wave3b_vt.py"),
        "infra_runtime": module.with_name("formal500h_infra_runtime.py"),
        "wave1_trainer": module.with_name("formal500h_wave1_train.py"),
        "evaluator": module.with_name("formal500h_evaluation.py"),
        "model": module.with_name("model.py"),
        "losses": module.with_name("losses.py"),
    }
    return {
        "schema_version": RUNTIME_SCHEMA,
        "release_manifest": str(manifest),
        "release_manifest_sha256": sha256_file(manifest),
        "dataset": "VariantAwareFormal500hDataset",
        "collate": "collate_formal500h_fast",
        "enabled_modalities": list(VT_MODALITIES),
        "hdf5_cache_size": 8,
        "prefetch_factor": 2,
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
    }


def _cell_model_config(
    spec: CellSpec,
    *,
    formal_variant: str | None = None,
    d_model: int = 128,
    dropout: float = 0.15,
) -> dict[str, Any]:
    # This is the frozen Wave-1/Wave-3B architecture with only the three
    # declared factorial fields changed.  Keeping the template local also
    # avoids recursion while wave1._model_config is temporarily patched.
    return {
        "variant": "b2_early_fusion",
        "formal_variant": formal_variant or spec.training_variant,
        "enabled_modalities": list(VT_MODALITIES),
        "vision_dim": 1536,
        "tactile_dim": 360,
        "tactile_representation": spec.tactile_representation,
        "skeleton_joints": 25,
        "skeleton_joint_dim": 7,
        "kinematic_dim": 48,
        "d_model": int(d_model),
        "dropout": float(dropout),
        "modality_dropout": 0.0,
        "stream_layers": 3,
        "decoder_channels": max(64, int(d_model)),
        "decoder_layers": 5,
        "decoder_stages": 2,
        "num_actions": NUM_CLASSES,
        "num_subtasks": 0,
        "use_boundary_head": bool(spec.use_boundary_head),
        "output_rate_hz": 30,
        "event_rate_hz": int(spec.event_rate_hz),
    }


def _remove_boundary_heads(model: OriginalFormal500hModel) -> None:
    decoder = model.backbone.decoder
    decoder.use_boundary_head = False
    stages = [decoder.first, *decoder.refinement]
    for stage in stages:
        stage.use_boundary_head = False
        stage.boundary_head = None


def _matched_model(config: Mapping[str, Any]) -> OriginalFormal500hModel:
    """Consume identical RNG for common parameters before boundary removal."""

    requested = bool(config.get("use_boundary_head", True))
    construction = dict(config)
    construction["use_boundary_head"] = True
    model = OriginalFormal500hModel(construction)
    if not requested:
        _remove_boundary_heads(model)
    return model


_ACTIVE_USE_BOUNDARY_HEAD = True


def _cell_evaluate_predictions(
    predictions,
    truth,
    *,
    boundary_threshold: float,
    boundary_min_distance_frames: int,
    min_segment_frames: int,
) -> dict[str, Any]:
    result = evaluate_dataset(
        predictions,
        truth,
        num_actions=NUM_CLASSES,
        segment_iou_thresholds=(0.10, 0.25, 0.50),
        boundary_tolerances_ms=(50, 100, 250, 500),
        primary_boundary_tolerance_ms=250,
        output_rate_hz=30.0,
        boundary_threshold=boundary_threshold,
        boundary_min_distance_frames=boundary_min_distance_frames,
        min_segment_frames=min_segment_frames,
        use_boundary_head=_ACTIVE_USE_BOUNDARY_HEAD,
        include_action_changes=True,
        strict=True,
        collect_artifacts=False,
    )
    return result.metrics


def _fixture_batch() -> dict[str, torch.Tensor]:
    # The released Wave-3B fixture already exercises all formal input fields.
    return wave3b._fixture_batch()


def _clone_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in batch.items()}


def _max_output_difference(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float:
    values: list[float] = []
    for name in ("atomic_logits", "boundary_logits", "hand_logits"):
        values.append(
            float((left[name].float() - right[name].float()).abs().max().item())
        )
    return max(values)


def _state_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _common_state_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    common = set(left).intersection(right)
    expected_left_only = {name for name in left if ".boundary_head." in name}
    expected_right_only = {name for name in right if ".boundary_head." in name}
    if (set(left) - common) != expected_left_only:
        return False
    if (set(right) - common) != expected_right_only:
        return False
    return all(torch.equal(left[name], right[name]) for name in common)


def build_model_gate(*, output: Path) -> dict[str, Any]:
    module = Path(__file__).resolve()
    if output.is_file():
        existing = _read_json(output)
        checks = {
            "schema": existing.get("schema_version") == MODEL_GATE_SCHEMA,
            "pass": existing.get("status") == "pass",
            "trainer": existing.get("trainer_sha256") == sha256_file(module),
            "test_sealed": existing.get("test_manifest_read") is False,
            "cells": set(existing.get("parameter_counts") or {}) == set(CELL_BY_ID),
        }
        if all(checks.values()):
            print(json.dumps({"event": "wave3c_model_gate_reused", **checks}), flush=True)
            return existing
        raise ValueError(f"stale Wave-3C model gate: {checks}")

    models: dict[str, OriginalFormal500hModel] = {}
    states: dict[str, Mapping[str, torch.Tensor]] = {}
    counts: dict[str, int] = {}
    for spec in CELL_SPECS:
        wave1._seed_everything(20260831)
        model = _matched_model(_cell_model_config(spec))
        model.eval()
        models[spec.cell] = model
        states[spec.cell] = model.state_dict()
        counts[spec.cell] = sum(parameter.numel() for parameter in model.parameters())

    boundary_cells = [spec.cell for spec in CELL_SPECS if spec.use_boundary_head]
    no_boundary_cells = [spec.cell for spec in CELL_SPECS if not spec.use_boundary_head]
    batch = _fixture_batch()
    with torch.no_grad():
        outputs = {name: model(_clone_batch(batch)) for name, model in models.items()}

    checks = {
        "eight_factorial_cells": len(CELL_SPECS) == 8 and len(CELL_BY_ID) == 8,
        "cell_ids_unique": len({value.cell for value in CELL_SPECS}) == 8,
        "boundary_group_parameter_matched": len({counts[name] for name in boundary_cells}) == 1,
        "no_boundary_group_parameter_matched": len({counts[name] for name in no_boundary_cells}) == 1,
        "boundary_group_initial_state_equal": all(
            _state_equal(states[boundary_cells[0]], states[name])
            for name in boundary_cells[1:]
        ),
        "no_boundary_group_initial_state_equal": all(
            _state_equal(states[no_boundary_cells[0]], states[name])
            for name in no_boundary_cells[1:]
        ),
        "common_initial_state_equal_across_boundary_ablation": _common_state_equal(
            states["D30B"], states["D30N"]
        ),
        "boundary_head_is_only_parameter_difference": (
            counts["D30B"] > counts["D30N"]
            and counts["D30B"] - counts["D30N"]
            == sum(
                value.numel()
                for name, value in states["D30B"].items()
                if ".boundary_head." in name
            )
        ),
        "raw_control_is_parameter_matched": counts["D30B"] == counts["R30B"],
        "event_rate_is_parameter_matched": counts["D30B"] == counts["D15B"],
        "raw_delta_representation_changes_output": _max_output_difference(
            outputs["D30B"], outputs["R30B"]
        ) > 0.0,
        "event_rate_changes_output": _max_output_difference(
            outputs["D30B"], outputs["D15B"]
        ) > 0.0,
        "no_boundary_logits_are_fixed": all(
            torch.equal(
                outputs[name]["boundary_logits"],
                torch.full_like(outputs[name]["boundary_logits"], -20.0),
            )
            for name in no_boundary_cells
        ),
        "no_boundary_parameters_removed": all(
            not any(".boundary_head." in name for name in states[cell])
            for cell in no_boundary_cells
        ),
        "modalities_are_exact_vt": all(
            _cell_model_config(spec)["enabled_modalities"] == list(VT_MODALITIES)
            for spec in CELL_SPECS
        ),
        "seven_seeds_frozen": SEEDS
        == (20260824, 20260825, 20260826, 20260827, 20260828, 20260829, 20260830),
    }
    result = {
        "schema_version": MODEL_GATE_SCHEMA,
        "status": "pass" if all(checks.values()) else "fail",
        "purpose": "formal_wave3c_two_by_two_by_two_factorial_model_gate",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "trainer": str(module),
        "trainer_sha256": sha256_file(module),
        "cells": [value.as_dict() for value in CELL_SPECS],
        "seeds": list(SEEDS),
        "original_seeds": list(ORIGINAL_SEEDS),
        "replication_seeds": list(REPLICATION_SEEDS),
        "parameter_counts": counts,
        "initial_model_sha256": {
            name: wave1._model_state_sha256(model) for name, model in models.items()
        },
        "checks": checks,
    }
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result["status"] != "pass":
        raise RuntimeError(f"formal Wave-3C model gate failed: {checks}")
    return result


def train_cell(
    *,
    data_dir: Path,
    output_dir: Path,
    cell: str,
    seed: int,
    device: str = "cuda:0",
    precision: str = "bf16",
    batch_size: int = 16,
    num_workers: int = 20,
    epochs: int = 3,
    learning_rate: float = 3.0e-4,
    calibration_workers: int = 4,
) -> dict[str, Any]:
    cell = cell.upper()
    if cell not in CELL_BY_ID or seed not in SEEDS:
        raise ValueError("Wave-3C requires one frozen cell and seed")
    spec = CELL_BY_ID[cell]
    if seed in ORIGINAL_SEEDS and cell == BASELINE_CELL:
        raise ValueError("Wave-3C must reuse the original three Wave-3B VT baseline runs")
    if (
        precision != "bf16"
        or batch_size != 16
        or num_workers != 20
        or epochs != 3
        or not math.isclose(learning_rate, 3.0e-4, rel_tol=1.0e-12)
        or calibration_workers != 4
    ):
        raise ValueError(
            "released Wave-3C recipe is bf16 batch=16 workers=20 "
            "epochs=3 lr=3e-4 calibration_workers=4"
        )

    original_modalities = wave1.VARIANT_MODALITIES
    original_dataset = wave1._dataset
    original_collate = wave1.collate_formal500h
    original_release = wave1._implementation_release
    original_model_config = wave1._model_config
    original_loss_config = wave1._loss_config
    original_model_class = wave1.Formal500hModel
    original_evaluate = wave1._evaluate_predictions
    lineage = _release_lineage()
    training_variant = spec.training_variant

    def optimized_dataset(manifest: Path, statistics_path: Path):
        rows = wave1._read_jsonl(manifest)
        if any(row.get("schema_version") != WINDOW_SCHEMA for row in rows):
            raise ValueError("Wave-1 window manifest schema changed")
        return VariantAwareFormal500hDataset(
            manifest,
            statistics_path,
            expected_statistics_schema=WAVE1_DATA_SCHEMA,
            enabled_modalities=VT_MODALITIES,
            hdf5_cache_size=8,
        )

    def released_implementation() -> dict[str, Any]:
        upstream = original_release()
        return {
            **upstream,
            "formal_wave1_upstream": {
                "release_manifest": upstream["release_manifest"],
                "release_manifest_sha256": upstream["release_manifest_sha256"],
            },
            "release_manifest": lineage["release_manifest"],
            "release_manifest_sha256": lineage["release_manifest_sha256"],
            "formal_wave3c": lineage,
            "wave3c_cell": spec.as_dict(),
        }

    def model_config(variant: str, *, d_model: int, dropout: float):
        if variant != training_variant:
            raise ValueError("Wave-3C variant dispatch changed")
        return _cell_model_config(
            spec,
            formal_variant=training_variant,
            d_model=d_model,
            dropout=dropout,
        )

    def loss_config(statistics_path: Path):
        value = original_loss_config(statistics_path)
        value["boundary_weight"] = 0.5 if spec.use_boundary_head else 0.0
        return value

    global _ACTIVE_USE_BOUNDARY_HEAD
    prior_active_boundary = _ACTIVE_USE_BOUNDARY_HEAD
    _ACTIVE_USE_BOUNDARY_HEAD = spec.use_boundary_head
    wave1.VARIANT_MODALITIES = {**original_modalities, training_variant: VT_MODALITIES}
    wave1._dataset = optimized_dataset
    wave1.collate_formal500h = collate_formal500h_fast
    wave1._implementation_release = released_implementation
    wave1._model_config = model_config
    wave1._loss_config = loss_config
    wave1.Formal500hModel = _matched_model
    wave1._evaluate_predictions = _cell_evaluate_predictions
    try:
        return wave1.train_formal_variant(
            data_dir=data_dir,
            output_dir=output_dir,
            variant=training_variant,
            seed=seed,
            device_name=device,
            precision=precision,
            batch_size=batch_size,
            num_workers=num_workers,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=1.0e-4,
            d_model=128,
            dropout=0.15,
            hand_weight=0.1,
            checkpoint_interval=2000,
            calibration_workers=calibration_workers,
            resume=True,
        )
    finally:
        _ACTIVE_USE_BOUNDARY_HEAD = prior_active_boundary
        wave1.VARIANT_MODALITIES = original_modalities
        wave1._dataset = original_dataset
        wave1.collate_formal500h = original_collate
        wave1._implementation_release = original_release
        wave1._model_config = original_model_config
        wave1._loss_config = original_loss_config
        wave1.Formal500hModel = original_model_class
        wave1._evaluate_predictions = original_evaluate


def _validated_wave3c_summary(
    root: Path,
    *,
    spec: CellSpec,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = root / "summary.json"
    complete_path = root / "COMPLETE.json"
    summary = _read_json(summary_path)
    complete = _read_json(complete_path)
    resolved_path = Path(str(summary.get("resolved_config", "")))
    checkpoint_path = Path(str(summary.get("best_checkpoint", "")))
    if not resolved_path.is_file() or not checkpoint_path.is_file():
        raise ValueError(f"missing Wave-3C run artifact: {root}")
    resolved = _read_json(resolved_path)
    model = resolved.get("model") or {}
    loss = resolved.get("loss") or {}
    calibration = summary.get("calibrated_validation") or {}
    selected = calibration.get("selected") or {}
    implementation = resolved.get("implementation_release") or {}
    runtime = implementation.get("formal_wave3c") or {}
    release_path = Path(str(runtime.get("release_manifest", "")))
    data_summary_path = Path(str(resolved.get("data_summary", "")))
    data_summary = _read_json(data_summary_path)
    expected_updates = 3 * math.ceil(int(data_summary["train"]["windows"]) / 16)
    training_variant = spec.training_variant
    policy = wave1.FORMAL_EVALUATION_POLICY["decode_calibration"]
    checks = {
        "complete": summary.get("status") == "complete"
        and complete.get("status") == "complete",
        "completion_hash": complete.get("summary_sha256") == sha256_file(summary_path),
        "checkpoint_hash": summary.get("best_checkpoint_sha256")
        == sha256_file(checkpoint_path),
        "resolved_hash": summary.get("resolved_config_sha256")
        == sha256_file(resolved_path),
        "test_sealed": summary.get("test_manifest_read") is False
        and complete.get("test_manifest_read") is False
        and resolved.get("test_manifest_read") is False
        and data_summary.get("test_manifest_read") is False,
        "variant": summary.get("variant") == training_variant
        and resolved.get("variant") == training_variant
        and model.get("formal_variant") == training_variant,
        "modalities": summary.get("enabled_modalities") == list(VT_MODALITIES)
        and resolved.get("enabled_modalities") == list(VT_MODALITIES)
        and model.get("enabled_modalities") == list(VT_MODALITIES),
        "seed": int(summary.get("seed", -1)) == seed
        and int(resolved.get("seed", -1)) == seed,
        "recipe": resolved.get("precision") == "bf16"
        and int(resolved.get("batch_size_per_gpu", -1)) == 16
        and int(resolved.get("num_workers", -1)) == 20
        and int(resolved.get("epochs", -1)) == 3
        and math.isclose(float(resolved.get("learning_rate", -1)), 3.0e-4, rel_tol=1e-12)
        and math.isclose(float(resolved.get("weight_decay", -1)), 1.0e-4, rel_tol=1e-12)
        and math.isclose(float(resolved.get("hand_weight", -1)), 0.1, rel_tol=1e-12),
        "updates": int(resolved.get("total_updates", -1)) == expected_updates
        and int(summary.get("updates", -1)) == expected_updates,
        "factors": model.get("tactile_representation") == spec.tactile_representation
        and int(model.get("event_rate_hz", -1)) == spec.event_rate_hz
        and model.get("use_boundary_head") is spec.use_boundary_head
        and math.isclose(
            float(loss.get("boundary_weight", -1)),
            0.5 if spec.use_boundary_head else 0.0,
            abs_tol=1e-12,
        ),
        "calibration": len(calibration.get("candidates") or []) == 96
        and selected.get("boundary_threshold") in policy["boundary_threshold_candidates"]
        and selected.get("boundary_min_distance_frames")
        in policy["boundary_min_distance_frames_30hz"]
        and selected.get("min_segment_frames") in policy["min_segment_frames_30hz"],
        "hand_order": summary.get("hand_order_policy")
        == "HDF5 already [left,right]; no training-time swap"
        and resolved.get("hand_order_policy")
        == "HDF5 already [left,right]; no training-time swap",
        "wave3c_runtime": runtime.get("schema_version") == RUNTIME_SCHEMA
        and runtime.get("dataset") == "VariantAwareFormal500hDataset"
        and runtime.get("collate") == "collate_formal500h_fast"
        and runtime.get("enabled_modalities") == list(VT_MODALITIES)
        and runtime.get("hdf5_cache_size") == 8
        and runtime.get("prefetch_factor") == 2
        and release_path.is_file()
        and sha256_file(release_path)
        == runtime.get("release_manifest_sha256")
        == implementation.get("release_manifest_sha256")
        == summary.get("release_manifest_sha256"),
        "cell_lineage": implementation.get("wave3c_cell") == spec.as_dict(),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid Wave-3C run {root}: {checks}")
    return summary, resolved


def _run_for_cell_seed(
    *,
    runs_root: Path,
    wave3b_root: Path,
    spec: CellSpec,
    seed: int,
) -> tuple[dict[str, Any], str]:
    if spec.cell == BASELINE_CELL and seed in ORIGINAL_SEEDS:
        root = wave3b_root / "runs" / "VT" / f"seed_{seed}"
        summary, resolved = wave3b._validated_summary(
            root,
            variant="VT",
            seed=seed,
            enabled_modalities=VT_MODALITIES,
            expected_workers=20,
        )
        model = resolved.get("model") or {}
        loss = resolved.get("loss") or {}
        checks = {
            "representation": model.get("tactile_representation") == "raw_delta_abs",
            "event_rate": int(model.get("event_rate_hz", -1)) == 30,
            "boundary_head": model.get("use_boundary_head") is True,
            "boundary_weight": math.isclose(
                float(loss.get("boundary_weight", -1)), 0.5, abs_tol=1e-12
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"Wave-3B baseline factor mismatch: {checks}")
        return summary, "reused_wave3b_vt"
    root = runs_root / spec.cell / f"seed_{seed}"
    summary, _ = _validated_wave3c_summary(root, spec=spec, seed=seed)
    return summary, "trained_wave3c"


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _paired(values: Sequence[float]) -> dict[str, Any]:
    return {
        **_mean_std(values),
        "values": [float(value) for value in values],
        "wins": sum(value > 0.0 for value in values),
        "ties": sum(value == 0.0 for value in values),
        "losses": sum(value < 0.0 for value in values),
    }


def _aggregate_cells(
    summaries: Mapping[str, Mapping[int, Mapping[str, Any]]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for spec in CELL_SPECS:
        output[spec.cell] = {
            metric: _mean_std([METRICS[metric](summaries[spec.cell][seed]) for seed in seeds])
            for metric in METRICS
        }
        output[spec.cell]["parameter_count"] = int(
            summaries[spec.cell][seeds[0]]["parameter_count"]
        )
        output[spec.cell]["factors"] = spec.as_dict()
    return output


def _contrast_vs_baseline(
    summaries: Mapping[str, Mapping[int, Mapping[str, Any]]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for spec in CELL_SPECS:
        if spec.cell == BASELINE_CELL:
            continue
        result[f"{spec.cell}_minus_{BASELINE_CELL}"] = {
            metric: _paired(
                [
                    METRICS[metric](summaries[spec.cell][seed])
                    - METRICS[metric](summaries[BASELINE_CELL][seed])
                    for seed in seeds
                ]
            )
            for metric in METRICS
        }
    return result


def _factor_code(spec: CellSpec) -> dict[str, int]:
    return {
        "raw_minus_delta": 1 if spec.tactile_representation == "raw" else -1,
        "rate15_minus30": 1 if spec.event_rate_hz == 15 else -1,
        "no_boundary_minus_boundary": -1 if spec.use_boundary_head else 1,
    }


def _factorial_effects(
    summaries: Mapping[str, Mapping[int, Mapping[str, Any]]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    terms = (
        ("raw_minus_delta",),
        ("rate15_minus30",),
        ("no_boundary_minus_boundary",),
        ("raw_minus_delta", "rate15_minus30"),
        ("raw_minus_delta", "no_boundary_minus_boundary"),
        ("rate15_minus30", "no_boundary_minus_boundary"),
        ("raw_minus_delta", "rate15_minus30", "no_boundary_minus_boundary"),
    )
    output: dict[str, Any] = {}
    for term in terms:
        label = "__x__".join(term)
        output[label] = {}
        for metric in METRICS:
            values: list[float] = []
            for seed in seeds:
                signed = 0.0
                for spec in CELL_SPECS:
                    code = _factor_code(spec)
                    sign = math.prod(code[name] for name in term)
                    signed += sign * METRICS[metric](summaries[spec.cell][seed])
                # Standard factorial effect: main differences, difference of
                # differences, and the three-way difference respectively.
                signed *= (2 ** len(term)) / len(CELL_SPECS)
                values.append(float(signed))
            output[label][metric] = _paired(values)
    return output


def aggregate_wave3c(
    *,
    runs_root: Path,
    wave3b_root: Path,
    wave3b_aggregate: Path,
    wave1_aggregate: Path,
    model_gate: Path,
    output: Path,
) -> dict[str, Any]:
    if output.is_file():
        existing = _read_json(output)
        complete_path = output.with_name("WAVE3C_COMPLETE.json")
        if complete_path.is_file():
            completion = _read_json(complete_path)
            checks = {
                "status": existing.get("status") == "complete",
                "complete": completion.get("status") == "complete",
                "hash": completion.get("aggregate_sha256") == sha256_file(output),
                "test_sealed": existing.get("test_manifest_read") is False
                and completion.get("test_manifest_read") is False,
            }
            if all(checks.values()):
                print(json.dumps({"event": "wave3c_aggregate_reused", **checks}), flush=True)
                return existing
        raise ValueError("partial or stale Wave-3C aggregate exists")

    gate = _read_json(model_gate)
    wave3b_value = _read_json(wave3b_aggregate)
    wave1_value = _read_json(wave1_aggregate)
    summaries: dict[str, dict[int, dict[str, Any]]] = {
        spec.cell: {} for spec in CELL_SPECS
    }
    sources: dict[str, dict[str, Any]] = {spec.cell: {} for spec in CELL_SPECS}
    for spec in CELL_SPECS:
        for seed in SEEDS:
            summary, source = _run_for_cell_seed(
                runs_root=runs_root,
                wave3b_root=wave3b_root,
                spec=spec,
                seed=seed,
            )
            summaries[spec.cell][seed] = summary
            sources[spec.cell][str(seed)] = {
                "source": source,
                "summary": str(Path(summary["resolved_config"]).parent / "summary.json"),
                "summary_sha256": sha256_file(
                    Path(summary["resolved_config"]).parent / "summary.json"
                ),
            }

    original_aggregate = _aggregate_cells(summaries, ORIGINAL_SEEDS)
    replication_aggregate = _aggregate_cells(summaries, REPLICATION_SEEDS)
    all_aggregate = _aggregate_cells(summaries, SEEDS)
    wave3b_vt = (wave3b_value.get("aggregate") or {}).get("VT") or {}
    preservation = {
        metric: math.isclose(
            float(original_aggregate[BASELINE_CELL][metric]["mean"]),
            float((wave3b_vt.get(metric) or {}).get("mean", math.nan)),
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(original_aggregate[BASELINE_CELL][metric]["std"]),
            float((wave3b_vt.get(metric) or {}).get("std", math.nan)),
            abs_tol=1.0e-12,
        )
        for metric in METRICS
    }
    all_test_sealed = all(
        summary.get("test_manifest_read") is False
        for by_seed in summaries.values()
        for summary in by_seed.values()
    )
    checks = {
        "model_gate_pass": gate.get("status") == "pass"
        and all((gate.get("checks") or {}).values()),
        "eight_cells": set(summaries) == set(CELL_BY_ID),
        "seven_paired_seeds": all(set(by_seed) == set(SEEDS) for by_seed in summaries.values()),
        "wave3b_baseline_reused_exactly": all(preservation.values()),
        "three_baseline_jobs_reused": sum(
            value["source"] == "reused_wave3b_vt"
            for value in sources[BASELINE_CELL].values()
        ) == 3,
        "fifty_three_jobs_trained": sum(
            value["source"] == "trained_wave3c"
            for by_seed in sources.values()
            for value in by_seed.values()
        ) == 53,
        "parameter_count_matches_model_gate": all(
            int(summary["parameter_count"])
            == int((gate.get("parameter_counts") or {}).get(cell, -1))
            for cell, by_seed in summaries.items()
            for summary in by_seed.values()
        ),
        "paired_initialization_within_boundary_group": all(
            len(
                {
                    summaries[spec.cell][seed]["initial_model_sha256"]
                    for spec in CELL_SPECS
                    if spec.use_boundary_head
                }
            )
            == 1
            for seed in SEEDS
        ),
        "paired_initialization_within_no_boundary_group": all(
            len(
                {
                    summaries[spec.cell][seed]["initial_model_sha256"]
                    for spec in CELL_SPECS
                    if not spec.use_boundary_head
                }
            )
            == 1
            for seed in SEEDS
        ),
        "same_data_protocol_audit_lineage": len(
            {
                (
                    summary["data_summary_sha256"],
                    summary["source_protocol_sha256"],
                    summary["source_audit_sha256"],
                )
                for by_seed in summaries.values()
                for summary in by_seed.values()
            }
        )
        == 1,
        "wave3b_complete": wave3b_value.get("status") == "complete",
        "wave1_complete": wave1_value.get("status") == "complete",
        "test_sealed": all_test_sealed
        and wave3b_value.get("test_manifest_read") is False
        and wave1_value.get("test_manifest_read") is False
        and gate.get("test_manifest_read") is False,
        "hand_order_already_corrected": all(
            summary.get("hand_order_policy")
            == "HDF5 already [left,right]; no training-time swap"
            for by_seed in summaries.values()
            for summary in by_seed.values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"formal Wave-3C aggregate checks failed: {checks}")

    leader = max(
        CELL_BY_ID,
        key=lambda cell: float(all_aggregate[cell]["primary_composite"]["mean"]),
    )
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "complete",
        "purpose": "formal_wave3c_tactile_representation_rate_boundary_factorial_validation_only",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "test_selection_deferred": True,
        "cells": [value.as_dict() for value in CELL_SPECS],
        "seeds": list(SEEDS),
        "original_seeds": list(ORIGINAL_SEEDS),
        "replication_seeds": list(REPLICATION_SEEDS),
        "jobs": {"factorial_total": 56, "reused_wave3b": 3, "trained_wave3c": 53},
        "controls": checks,
        "baseline_preservation": preservation,
        "aggregate": {
            "original_three_seed_block": original_aggregate,
            "replication_four_seed_block": replication_aggregate,
            "all_seven_seeds": all_aggregate,
        },
        "paired_vs_baseline": {
            "original_three_seed_block": _contrast_vs_baseline(summaries, ORIGINAL_SEEDS),
            "replication_four_seed_block": _contrast_vs_baseline(summaries, REPLICATION_SEEDS),
            "all_seven_seeds": _contrast_vs_baseline(summaries, SEEDS),
        },
        "factorial_effects": {
            "original_three_seed_block": _factorial_effects(summaries, ORIGINAL_SEEDS),
            "replication_four_seed_block": _factorial_effects(summaries, REPLICATION_SEEDS),
            "all_seven_seeds": _factorial_effects(summaries, SEEDS),
        },
        "validation_leader": {
            "cell": leader,
            "primary_composite": all_aggregate[leader]["primary_composite"],
            "selection_scope": "validation_only",
        },
        "sources": sources,
        "parents": {
            "wave3b_aggregate": {
                "path": str(wave3b_aggregate.resolve()),
                "sha256": sha256_file(wave3b_aggregate),
            },
            "wave1_aggregate": {
                "path": str(wave1_aggregate.resolve()),
                "sha256": sha256_file(wave1_aggregate),
            },
            "model_gate": {
                "path": str(model_gate.resolve()),
                "sha256": sha256_file(model_gate),
            },
        },
        "decision": "factorial_validation_complete_test_selection_deferred",
    }
    _atomic_json(output, result)
    completion = {
        "schema_version": "formal500h_wave3c_complete_v1",
        "status": "complete",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "test_selection_deferred": True,
        "aggregate": str(output.resolve()),
        "aggregate_sha256": sha256_file(output),
        "model_gate": str(model_gate.resolve()),
        "model_gate_sha256": sha256_file(model_gate),
        "validation_leader": leader,
        "decision": result["decision"],
    }
    _atomic_json(output.with_name("WAVE3C_COMPLETE.json"), completion)
    print(
        json.dumps(
            {
                "event": "formal500h_wave3c_aggregate_complete",
                "status": "complete",
                "leader": leader,
                "primary": all_aggregate[leader]["primary_composite"],
                "checks": checks,
                "test_manifest_read": False,
            }
        ),
        flush=True,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("model-gate")
    gate.add_argument("--output", type=Path, required=True)

    train = commands.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--cell", choices=tuple(CELL_BY_ID), required=True)
    train.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--precision", choices=("bf16",), default="bf16")
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--num-workers", type=int, default=20)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=3.0e-4)
    train.add_argument("--calibration-workers", type=int, default=4)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--runs-root", type=Path, required=True)
    aggregate.add_argument("--wave3b-root", type=Path, required=True)
    aggregate.add_argument("--wave3b-aggregate", type=Path, required=True)
    aggregate.add_argument("--wave1-aggregate", type=Path, required=True)
    aggregate.add_argument("--model-gate", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "model-gate":
        build_model_gate(output=arguments.output)
    elif arguments.command == "train":
        train_cell(
            data_dir=arguments.data_dir,
            output_dir=arguments.output_dir,
            cell=arguments.cell,
            seed=arguments.seed,
            device=arguments.device,
            precision=arguments.precision,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            calibration_workers=arguments.calibration_workers,
        )
    elif arguments.command == "aggregate":
        aggregate_wave3c(
            runs_root=arguments.runs_root,
            wave3b_root=arguments.wave3b_root,
            wave3b_aggregate=arguments.wave3b_aggregate,
            wave1_aggregate=arguments.wave1_aggregate,
            model_gate=arguments.model_gate,
            output=arguments.output,
        )
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
