from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from .formal500h_infra_runtime import (
    VariantAwareFormal500hDataset,
    collate_formal500h_fast,
)
from .formal500h_protocol import sha256_file
from .formal500h_wave1_data import WAVE1_DATA_SCHEMA, WINDOW_SCHEMA
from . import formal500h_wave1_train as wave1


TRAINING_SCHEMA = "formal500h_wave3b_vt_training_v1"
AGGREGATE_SCHEMA = "formal500h_wave3b_vt_aggregate_v1"
RUNTIME_SCHEMA = "formal500h_wave3b_vt_optimized_runtime_v1"
VARIANT = "VT"
REFERENCE_VARIANT = "B0"
VT_MODALITIES = ("vision", "tactile")
B0_MODALITIES = ("vision",)
SEEDS = (20260824, 20260825, 20260826)

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


def _vt_model_config(*, d_model: int = 128, dropout: float = 0.15) -> dict[str, Any]:
    """Build VT with the frozen Wave-1 architecture and no persistent mutation."""
    original = wave1.VARIANT_MODALITIES
    wave1.VARIANT_MODALITIES = {**original, VARIANT: VT_MODALITIES}
    try:
        return wave1._model_config(VARIANT, d_model=d_model, dropout=dropout)
    finally:
        wave1.VARIANT_MODALITIES = original


def _release_lineage() -> dict[str, Any]:
    module = Path(__file__).resolve()
    infra_runtime = module.with_name("formal500h_infra_runtime.py")
    release = (
        module.parents[2]
        / "configs"
        / "experimental"
        / "atomic_v1_wave3b_release_sha256.txt"
    )
    if not release.is_file():
        raise ValueError(f"Wave-3B VT release manifest is missing: {release}")
    return {
        "schema_version": RUNTIME_SCHEMA,
        "dataset": "VariantAwareFormal500hDataset",
        "collate": "collate_formal500h_fast",
        "enabled_modalities": list(VT_MODALITIES),
        "hdf5_cache_size": 8,
        "prefetch_factor": 2,
        "wave3b_vt_trainer": str(module),
        "wave3b_vt_trainer_sha256": sha256_file(module),
        "infra_runtime": str(infra_runtime),
        "infra_runtime_sha256": sha256_file(infra_runtime),
        "release_manifest": str(release),
        "release_manifest_sha256": sha256_file(release),
    }


def _fixture_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(8317)
    batch_size, steps, vision_steps = 2, 6, 3
    tactile_quality = torch.zeros(batch_size, steps, 2, 5)
    # quality[..., 0] is packet validity; fields 2:5 are anomaly fractions.
    tactile_quality[..., 0] = 1.0
    return {
        "sequence_mask": torch.ones(batch_size, steps, dtype=torch.bool),
        "vision_features": torch.randn(
            batch_size, vision_steps, 3, 1536, generator=generator
        ),
        "vision_cache_valid": torch.ones(
            batch_size, vision_steps, 3, dtype=torch.bool
        ),
        "vision_index_60": torch.arange(steps)
        .div(2, rounding_mode="floor")
        .reshape(1, steps, 1)
        .expand(batch_size, -1, 3),
        "vision_valid_60": torch.ones(batch_size, steps, 3, dtype=torch.bool),
        "vision_freshness_ms": torch.zeros(batch_size, steps, 3),
        "tactile": torch.randn(
            batch_size, steps, 2, 360, generator=generator
        ),
        "tactile_valid": torch.ones(batch_size, steps, 2, dtype=torch.bool),
        "tactile_quality": tactile_quality,
        "skeleton": torch.randn(
            batch_size, steps, 2, 25, 7, generator=generator
        ),
        "skeleton_valid": torch.ones(batch_size, steps, 2, dtype=torch.bool),
        "skeleton_joint_valid": torch.ones(
            batch_size, steps, 2, 25, dtype=torch.bool
        ),
        "kinematics": torch.randn(
            batch_size, steps, 48, generator=generator
        ),
        "kinematics_valid": torch.ones(batch_size, steps, dtype=torch.bool),
        "pose_quality": torch.ones(batch_size, steps, 6),
    }


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def _max_output_difference(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
) -> float:
    return max(
        float((first[name].float() - second[name].float()).abs().max())
        for name in ("atomic_logits", "boundary_logits", "hand_logits")
    )


def run_model_gate(*, output: Path) -> dict[str, Any]:
    """Prove that VT changes only the enabled input set relative to B0."""
    module = Path(__file__).resolve()
    if output.is_file():
        existing = _read_json(output)
        if (
            existing.get("status") == "pass"
            and existing.get("test_manifest_read") is False
            and existing.get("trainer_sha256") == sha256_file(module)
            and bool(existing.get("checks"))
            and all(value is True for value in existing["checks"].values())
        ):
            print(json.dumps({"event": "wave3b_vt_model_gate_reused", **existing}))
            return existing
        raise ValueError("existing Wave-3B VT model gate is stale or invalid")

    # Exercise the exact released model size.  A reduced diagnostic model could
    # prove the input mask logic, but would not prove parameter matching for the
    # formal d_model=128 jobs that this gate releases.
    b0_config = wave1._model_config(
        REFERENCE_VARIANT, d_model=128, dropout=0.15
    )
    vt_config = _vt_model_config(d_model=128, dropout=0.15)
    models: dict[str, torch.nn.Module] = {}
    for name, config in ((REFERENCE_VARIANT, b0_config), (VARIANT, vt_config)):
        torch.manual_seed(SEEDS[0])
        models[name] = wave1.Formal500hModel(config).eval()

    states = {name: model.state_dict() for name, model in models.items()}
    parameter_names = {
        name: [key for key, _ in model.named_parameters()]
        for name, model in models.items()
    }
    parameter_counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    initial_state_equal = list(states[REFERENCE_VARIANT]) == list(states[VARIANT]) and all(
        torch.equal(states[REFERENCE_VARIANT][key], states[VARIANT][key])
        for key in states[REFERENCE_VARIANT]
    )

    comparable_b0 = {
        key: value
        for key, value in b0_config.items()
        if key not in {"formal_variant", "enabled_modalities"}
    }
    comparable_vt = {
        key: value
        for key, value in vt_config.items()
        if key not in {"formal_variant", "enabled_modalities"}
    }

    batch = _fixture_batch()
    disabled_changed = _clone_batch(batch)
    disabled_changed["skeleton"].mul_(17.0).add_(9.0)
    disabled_changed["skeleton_valid"].logical_not_()
    disabled_changed["skeleton_joint_valid"].logical_not_()
    disabled_changed["kinematics"].mul_(-13.0).add_(5.0)
    disabled_changed["kinematics_valid"].logical_not_()
    disabled_changed["pose_quality"].zero_()
    tactile_missing = _clone_batch(batch)
    tactile_missing["tactile"].zero_()
    tactile_missing["tactile_valid"].zero_()
    tactile_missing["tactile_quality"].zero_()
    with torch.no_grad():
        b0_base = models[REFERENCE_VARIANT](batch)
        b0_tactile_missing = models[REFERENCE_VARIANT](tactile_missing)
        vt_base = models[VARIANT](batch)
        vt_disabled_changed = models[VARIANT](disabled_changed)
        vt_tactile_missing = models[VARIANT](tactile_missing)

    differences = {
        "b0_tactile_missing": _max_output_difference(
            b0_base, b0_tactile_missing
        ),
        "vt_disabled_pose_changed": _max_output_difference(
            vt_base, vt_disabled_changed
        ),
        "vt_tactile_missing": _max_output_difference(
            vt_base, vt_tactile_missing
        ),
    }
    checks = {
        "parameter_count_equal": parameter_counts[REFERENCE_VARIANT]
        == parameter_counts[VARIANT],
        "parameter_names_equal": parameter_names[REFERENCE_VARIANT]
        == parameter_names[VARIANT],
        "initial_state_bitwise_equal": initial_state_equal,
        "model_config_equal_except_variant_and_modalities": comparable_b0
        == comparable_vt,
        "b0_modalities_exact": b0_config["enabled_modalities"]
        == list(B0_MODALITIES),
        "vt_modalities_exact": vt_config["enabled_modalities"]
        == list(VT_MODALITIES),
        "b0_ignores_tactile_bitwise": differences["b0_tactile_missing"] == 0.0,
        "vt_ignores_pose_bitwise": differences["vt_disabled_pose_changed"] == 0.0,
        "vt_uses_tactile": differences["vt_tactile_missing"] > 0.0,
        "all_parameters_trainable": all(
            parameter.requires_grad
            for model in models.values()
            for parameter in model.parameters()
        ),
    }
    result = {
        "schema_version": TRAINING_SCHEMA,
        "status": "pass" if all(checks.values()) else "fail",
        "purpose": "formal_wave3b_vt_parameter_input_and_initialization_gate",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "trainer": str(module),
        "trainer_sha256": sha256_file(module),
        "formal_model_recipe": {"d_model": 128, "dropout": 0.15},
        "parameter_counts": parameter_counts,
        "initial_model_sha256": {
            name: wave1._model_state_sha256(model)
            for name, model in models.items()
        },
        "output_max_abs": differences,
        "checks": checks,
    }
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result["status"] != "pass":
        raise RuntimeError("formal Wave-3B VT model gate failed")
    return result


def train_variant(
    *,
    data_dir: Path,
    output_dir: Path,
    variant: str,
    seed: int,
    device: str = "cuda:0",
    precision: str = "bf16",
    batch_size: int = 16,
    num_workers: int = 20,
    epochs: int = 3,
    learning_rate: float = 3.0e-4,
    calibration_workers: int = 4,
) -> dict[str, Any]:
    variant = variant.upper()
    if variant != VARIANT or seed not in SEEDS:
        raise ValueError("formal Wave-3B VT requires VT and the frozen three seeds")
    if (
        precision != "bf16"
        or batch_size != 16
        or num_workers != 20
        or epochs != 3
        or not math.isclose(learning_rate, 3.0e-4, rel_tol=1.0e-12)
        or calibration_workers != 4
    ):
        raise ValueError(
            "released Wave-3B VT recipe is bf16 batch=16 workers=20 "
            "epochs=3 lr=3e-4 calibration_workers=4"
        )

    original_modalities = wave1.VARIANT_MODALITIES
    original_dataset = wave1._dataset
    original_collate = wave1.collate_formal500h
    original_release = wave1._implementation_release
    lineage = _release_lineage()

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
        value = original_release()
        value["formal_wave1_upstream"] = {
            "release_manifest": value["release_manifest"],
            "release_manifest_sha256": value["release_manifest_sha256"],
        }
        value["release_manifest"] = lineage["release_manifest"]
        value["release_manifest_sha256"] = lineage["release_manifest_sha256"]
        value["formal_wave3b_vt"] = lineage
        return value

    wave1.VARIANT_MODALITIES = {
        **original_modalities,
        VARIANT: VT_MODALITIES,
    }
    wave1._dataset = optimized_dataset
    wave1.collate_formal500h = collate_formal500h_fast
    wave1._implementation_release = released_implementation
    try:
        return wave1.train_formal_variant(
            data_dir=data_dir,
            output_dir=output_dir,
            variant=variant,
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
        wave1.VARIANT_MODALITIES = original_modalities
        wave1._dataset = original_dataset
        wave1.collate_formal500h = original_collate
        wave1._implementation_release = original_release


def _validated_summary(
    root: Path,
    *,
    variant: str,
    seed: int,
    enabled_modalities: Sequence[str],
    expected_workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = root / "summary.json"
    complete_path = root / "COMPLETE.json"
    summary = _read_json(summary_path)
    complete = _read_json(complete_path)
    resolved_path = Path(str(summary.get("resolved_config", "")))
    checkpoint_path = Path(str(summary.get("best_checkpoint", "")))
    if not resolved_path.is_file() or not checkpoint_path.is_file():
        raise ValueError(f"missing resolved/checkpoint artifact: {root}")
    resolved = _read_json(resolved_path)
    data_summary_path = Path(str(resolved.get("data_summary", "")))
    if not data_summary_path.is_file():
        raise ValueError(f"missing data summary: {data_summary_path}")
    data_summary = _read_json(data_summary_path)
    train_windows = int((data_summary.get("train") or {}).get("windows", -1))
    expected_updates = 3 * math.ceil(train_windows / 16)
    calibration = summary.get("calibrated_validation") or {}
    selected = calibration.get("selected") or {}
    policy = wave1.FORMAL_EVALUATION_POLICY["decode_calibration"]
    model = resolved.get("model") or {}
    checks = {
        "schema": summary.get("schema_version") == wave1.TRAINING_SCHEMA
        and resolved.get("schema_version") == wave1.TRAINING_SCHEMA,
        "complete": summary.get("status") == "complete"
        and complete.get("status") == "complete",
        "completion_hash": complete.get("summary_sha256")
        == sha256_file(summary_path),
        "checkpoint_hash": summary.get("best_checkpoint_sha256")
        == sha256_file(checkpoint_path),
        "resolved_hash": summary.get("resolved_config_sha256")
        == sha256_file(resolved_path),
        "test_sealed": summary.get("test_manifest_read") is False
        and complete.get("test_manifest_read") is False
        and resolved.get("test_manifest_read") is False
        and data_summary.get("test_manifest_read") is False,
        "episode_contract": summary.get("episode_contract")
        == "formal500h_training_episode_v3"
        and resolved.get("episode_contract") == "formal500h_training_episode_v3",
        "variant": summary.get("variant") == variant
        and resolved.get("variant") == variant
        and model.get("formal_variant") == variant,
        "modalities": summary.get("enabled_modalities")
        == list(enabled_modalities)
        and resolved.get("enabled_modalities") == list(enabled_modalities)
        and model.get("enabled_modalities") == list(enabled_modalities),
        "seed": int(summary.get("seed", -1)) == seed
        and int(resolved.get("seed", -1)) == seed,
        "recipe": resolved.get("precision") == "bf16"
        and int(resolved.get("batch_size_per_gpu", -1)) == 16
        and int(resolved.get("num_workers", -1)) == expected_workers
        and int(resolved.get("epochs", -1)) == 3
        and math.isclose(
            float(resolved.get("learning_rate", -1)), 3.0e-4, rel_tol=1.0e-12
        )
        and math.isclose(
            float(resolved.get("weight_decay", -1)), 1.0e-4, rel_tol=1.0e-12
        )
        and math.isclose(
            float(resolved.get("hand_weight", -1)), 0.1, rel_tol=1.0e-12
        ),
        "updates": int(resolved.get("total_updates", -1)) == expected_updates
        and int(summary.get("updates", -1)) == expected_updates,
        "data_summary": data_summary.get("status") == "complete"
        and sha256_file(data_summary_path)
        == resolved.get("data_summary_sha256")
        == summary.get("data_summary_sha256"),
        "calibration": len(calibration.get("candidates") or []) == 96
        and selected.get("boundary_threshold")
        in policy["boundary_threshold_candidates"]
        and selected.get("boundary_min_distance_frames")
        in policy["boundary_min_distance_frames_30hz"]
        and selected.get("min_segment_frames")
        in policy["min_segment_frames_30hz"],
        "hand_order": summary.get("hand_order_policy")
        == "HDF5 already [left,right]; no training-time swap"
        and resolved.get("hand_order_policy")
        == "HDF5 already [left,right]; no training-time swap",
    }
    if variant == VARIANT:
        implementation = resolved.get("implementation_release") or {}
        runtime = implementation.get("formal_wave3b_vt") or {}
        release_path = Path(str(runtime.get("release_manifest", "")))
        trainer_path = Path(str(runtime.get("wave3b_vt_trainer", "")))
        infra_path = Path(str(runtime.get("infra_runtime", "")))
        checks["optimized_runtime"] = (
            runtime.get("schema_version") == RUNTIME_SCHEMA
            and runtime.get("dataset") == "VariantAwareFormal500hDataset"
            and runtime.get("collate") == "collate_formal500h_fast"
            and runtime.get("enabled_modalities") == list(VT_MODALITIES)
            and runtime.get("hdf5_cache_size") == 8
            and runtime.get("prefetch_factor") == 2
            and trainer_path.is_file()
            and sha256_file(trainer_path)
            == runtime.get("wave3b_vt_trainer_sha256")
            and infra_path.is_file()
            and sha256_file(infra_path) == runtime.get("infra_runtime_sha256")
            and release_path.is_file()
            and sha256_file(release_path)
            == runtime.get("release_manifest_sha256")
            == summary.get("release_manifest_sha256")
            == implementation.get("release_manifest_sha256")
        )
    if not all(checks.values()):
        raise ValueError(f"invalid formal {variant} run {root}: {checks}")
    return summary, resolved


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def aggregate_vt(
    *,
    vt_runs_root: Path,
    wave1_b0_root: Path,
    wave1_aggregate: Path,
    model_gate: Path,
    output: Path,
) -> dict[str, Any]:
    wave1_result = _read_json(wave1_aggregate)
    wave1_complete_path = wave1_aggregate.parent / "WAVE1_COMPLETE.json"
    wave1_complete = _read_json(wave1_complete_path)
    gate = _read_json(model_gate)
    wave1_checks = {
        "schema": wave1_result.get("schema_version") == wave1.AGGREGATE_SCHEMA,
        "complete": wave1_result.get("status") == "complete"
        and wave1_complete.get("status") == "complete",
        "aggregate_hash": wave1_complete.get("aggregate_sha256")
        == sha256_file(wave1_aggregate),
        "b0_present": REFERENCE_VARIANT in (wave1_result.get("variants") or []),
        "frozen_seeds": wave1_result.get("seeds") == list(SEEDS),
        "test_sealed": wave1_result.get("test_manifest_read") is False
        and wave1_complete.get("test_manifest_read") is False
        and (wave1_result.get("controls") or {}).get("test_sealed") is True,
        "full_episode_validation": (wave1_result.get("controls") or {}).get(
            "full_episode_validation"
        )
        is True,
    }
    gate_checks = {
        "schema": gate.get("schema_version") == TRAINING_SCHEMA,
        "pass": gate.get("status") == "pass",
        "test_sealed": gate.get("test_manifest_read") is False,
        "formal_recipe": gate.get("formal_model_recipe")
        == {"d_model": 128, "dropout": 0.15},
        "formal_parameter_count": gate.get("parameter_counts")
        == {REFERENCE_VARIANT: 1463482, VARIANT: 1463482},
        "paired_gate_initialization": (
            (gate.get("initial_model_sha256") or {}).get(REFERENCE_VARIANT)
            == (gate.get("initial_model_sha256") or {}).get(VARIANT)
            and bool(
                (gate.get("initial_model_sha256") or {}).get(
                    REFERENCE_VARIANT
                )
            )
        ),
        "trainer_hash": gate.get("trainer_sha256")
        == sha256_file(Path(__file__).resolve()),
        "all_checks": bool(gate.get("checks"))
        and all(value is True for value in gate["checks"].values()),
    }
    if not all(wave1_checks.values()) or not all(gate_checks.values()):
        raise ValueError(
            f"Wave-3B VT aggregate preflight failed: "
            f"wave1={wave1_checks}, model_gate={gate_checks}"
        )

    summaries: dict[str, dict[int, dict[str, Any]]] = {
        REFERENCE_VARIANT: {},
        VARIANT: {},
    }
    configs: dict[str, dict[int, dict[str, Any]]] = {
        REFERENCE_VARIANT: {},
        VARIANT: {},
    }
    for seed in SEEDS:
        summaries[REFERENCE_VARIANT][seed], configs[REFERENCE_VARIANT][seed] = (
            _validated_summary(
                wave1_b0_root / f"seed_{seed}",
                variant=REFERENCE_VARIANT,
                seed=seed,
                enabled_modalities=B0_MODALITIES,
                expected_workers=16,
            )
        )
        summaries[VARIANT][seed], configs[VARIANT][seed] = _validated_summary(
            vt_runs_root / VARIANT / f"seed_{seed}",
            variant=VARIANT,
            seed=seed,
            enabled_modalities=VT_MODALITIES,
            expected_workers=20,
        )

    recomputed_b0 = {
        metric: _mean_std(
            [getter(summaries[REFERENCE_VARIANT][seed]) for seed in SEEDS]
        )
        for metric, getter in METRICS.items()
    }
    frozen_b0 = (wave1_result.get("aggregate") or {}).get(
        REFERENCE_VARIANT
    ) or {}

    def same_metric(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
        return all(
            key in first
            and key in second
            and math.isclose(
                float(first[key]),
                float(second[key]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            for key in ("mean", "std")
        )

    def model_except_intervention(config: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in (config.get("model") or {}).items()
            if key not in {"formal_variant", "enabled_modalities"}
        }

    wave1_release_hash = (wave1_result.get("controls") or {}).get(
        "release_manifest_sha256"
    )
    controls = {
        "b0_reproduces_frozen_wave1_aggregate": all(
            same_metric(recomputed_b0[metric], frozen_b0.get(metric) or {})
            for metric in METRICS
        ),
        "paired_initialization_within_seed": all(
            summaries[REFERENCE_VARIANT][seed]["initial_model_sha256"]
            == summaries[VARIANT][seed]["initial_model_sha256"]
            for seed in SEEDS
        ),
        "same_parameter_count": all(
            summaries[REFERENCE_VARIANT][seed]["parameter_count"]
            == summaries[VARIANT][seed]["parameter_count"]
            for seed in SEEDS
        ),
        "same_model_except_enabled_modalities": all(
            model_except_intervention(configs[REFERENCE_VARIANT][seed])
            == model_except_intervention(configs[VARIANT][seed])
            for seed in SEEDS
        ),
        "same_loss": all(
            configs[REFERENCE_VARIANT][seed]["loss"]
            == configs[VARIANT][seed]["loss"]
            for seed in SEEDS
        ),
        "same_evaluation_policy": all(
            configs[REFERENCE_VARIANT][seed]["evaluation_policy"]
            == configs[VARIANT][seed]["evaluation_policy"]
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
        "same_seen_examples_and_optimization": all(
            summaries[REFERENCE_VARIANT][seed]["epochs"]
            == summaries[VARIANT][seed]["epochs"]
            == 3
            and summaries[REFERENCE_VARIANT][seed]["updates"]
            == summaries[VARIANT][seed]["updates"]
            and configs[REFERENCE_VARIANT][seed]["precision"]
            == configs[VARIANT][seed]["precision"]
            == "bf16"
            and configs[REFERENCE_VARIANT][seed]["batch_size_per_gpu"]
            == configs[VARIANT][seed]["batch_size_per_gpu"]
            == 16
            and math.isclose(
                float(configs[REFERENCE_VARIANT][seed]["learning_rate"]),
                float(configs[VARIANT][seed]["learning_rate"]),
                rel_tol=1.0e-12,
            )
            and math.isclose(
                float(configs[REFERENCE_VARIANT][seed]["weight_decay"]),
                float(configs[VARIANT][seed]["weight_decay"]),
                rel_tol=1.0e-12,
            )
            and math.isclose(
                float(configs[REFERENCE_VARIANT][seed]["hand_weight"]),
                float(configs[VARIANT][seed]["hand_weight"]),
                rel_tol=1.0e-12,
            )
            for seed in SEEDS
        ),
        "vt_uses_released_optimized_reader": all(
            configs[VARIANT][seed]["num_workers"] == 20
            and (
                configs[VARIANT][seed]["implementation_release"]
                ["formal_wave3b_vt"]["dataset"]
                == "VariantAwareFormal500hDataset"
            )
            for seed in SEEDS
        ),
        "vt_links_frozen_wave1_upstream": all(
            summaries[REFERENCE_VARIANT][seed]["release_manifest_sha256"]
            == wave1_release_hash
            and (
                configs[VARIANT][seed]["implementation_release"]
                ["formal_wave1_upstream"]["release_manifest_sha256"]
                == wave1_release_hash
            )
            for seed in SEEDS
        ),
        "full_episode_validation": all(
            len(summaries[name][seed]["calibrated_validation"]["candidates"])
            == 96
            for name in summaries
            for seed in SEEDS
        ),
        "test_sealed": all(
            summaries[name][seed]["test_manifest_read"] is False
            and configs[name][seed]["test_manifest_read"] is False
            for name in summaries
            for seed in SEEDS
        ),
        "model_gate_passed": all(gate_checks.values()),
    }
    if not all(controls.values()):
        raise ValueError(f"formal Wave-3B VT controls failed: {controls}")

    aggregate: dict[str, Any] = {}
    for name, by_seed in summaries.items():
        aggregate[name] = {
            metric: _mean_std([getter(by_seed[seed]) for seed in SEEDS])
            for metric, getter in METRICS.items()
        }
        aggregate[name]["parameter_count"] = int(
            by_seed[SEEDS[0]]["parameter_count"]
        )
    paired: dict[str, Any] = {"VT_minus_B0": {}}
    for metric, getter in METRICS.items():
        values = [
            getter(summaries[VARIANT][seed])
            - getter(summaries[REFERENCE_VARIANT][seed])
            for seed in SEEDS
        ]
        paired["VT_minus_B0"][metric] = {
            **_mean_std(values),
            "values": values,
            "wins": sum(value > 0 for value in values),
        }

    primary = paired["VT_minus_B0"]["primary_composite"]
    if primary["mean"] > 0 and primary["wins"] >= 2:
        decision = "vt_positive_direction_on_clean_validation"
    elif primary["mean"] < 0 and primary["wins"] <= 1:
        decision = "vt_not_supported_on_clean_validation"
    else:
        decision = "vt_clean_validation_inconclusive"
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "complete",
        "purpose": "formal_wave3b_vt_vs_b0_validation_only",
        "created_unix_s": time.time(),
        "test_manifest_read": False,
        "test_selection_deferred": True,
        "variants": [REFERENCE_VARIANT, VARIANT],
        "seeds": list(SEEDS),
        "jobs_trained_in_wave3b_vt": 3,
        "controls": controls,
        "definitions": {
            "B0": "frozen Wave-1 vision-only reference",
            "VT": "same Wave-1 architecture with vision and tactile enabled",
            "primary_causal_comparison": "VT_minus_B0",
            "runtime_note": (
                "B0 used the frozen legacy reader with 16 workers; VT uses the "
                "released bitwise-equivalent optimized reader with 20 workers"
            ),
        },
        "wave1_aggregate": str(wave1_aggregate.resolve()),
        "wave1_aggregate_sha256": sha256_file(wave1_aggregate),
        "model_gate": str(model_gate.resolve()),
        "model_gate_sha256": sha256_file(model_gate),
        "aggregate": aggregate,
        "paired": paired,
        "decision": decision,
    }
    _atomic_json(output, result)
    complete = {
        "schema_version": TRAINING_SCHEMA,
        "status": "complete",
        "test_manifest_read": False,
        "test_selection_deferred": True,
        "aggregate": str(output.resolve()),
        "aggregate_sha256": sha256_file(output),
        "decision": decision,
    }
    _atomic_json(output.parent / "WAVE3B_VT_COMPLETE.json", complete)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Formal500h Wave-3B strict vision+tactile experiment"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    model_gate = commands.add_parser("model-gate")
    model_gate.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--variant", choices=(VARIANT,), default=VARIANT)
    train.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--precision", choices=("bf16",), default="bf16")
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--num-workers", type=int, default=20)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=3.0e-4)
    train.add_argument("--calibration-workers", type=int, default=4)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--vt-runs-root", type=Path, required=True)
    aggregate.add_argument("--wave1-b0-root", type=Path, required=True)
    aggregate.add_argument("--wave1-aggregate", type=Path, required=True)
    aggregate.add_argument("--model-gate", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "model-gate":
        run_model_gate(output=arguments.output)
    elif arguments.command == "train":
        train_variant(
            data_dir=arguments.data_dir,
            output_dir=arguments.output_dir,
            variant=arguments.variant,
            seed=arguments.seed,
            device=arguments.device,
            precision=arguments.precision,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            calibration_workers=arguments.calibration_workers,
        )
    else:
        aggregate_vt(
            vt_runs_root=arguments.vt_runs_root,
            wave1_b0_root=arguments.wave1_b0_root,
            wave1_aggregate=arguments.wave1_aggregate,
            model_gate=arguments.model_gate,
            output=arguments.output,
        )


if __name__ == "__main__":
    main()
