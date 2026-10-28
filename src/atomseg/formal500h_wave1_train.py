from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import socket
import time
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler

from .formal500h_evaluation import GroundTruthEpisode, evaluate_dataset
from .formal500h_clock import clock_metadata, downsample_targets_60_to_30
from .formal500h_protocol import FORMAL_EVALUATION_POLICY, NUM_CLASSES, sha256_file
from .formal500h_runtime import (
    Formal500hDataset,
    Formal500hModel,
    collate_formal500h,
    formal500h_class_weights,
    formal500h_loss_terms,
)
from .formal500h_wave1_data import WAVE1_DATA_SCHEMA, WINDOW_SCHEMA
from .inference import StitchedEpisode, WindowPrediction, stitch_window_predictions
from .losses import ContactTASLoss


TRAINING_SCHEMA = "formal500h_wave1_training_v1"
AGGREGATE_SCHEMA = "formal500h_wave1_aggregate_v1"
VARIANT_MODALITIES = {
    "B0": ("vision",),
    "B1": ("vision", "skeleton", "kinematics"),
    "B2": ("vision", "tactile", "skeleton", "kinematics"),
}
DEFAULT_SEEDS = (20260824, 20260825, 20260826)
_CALIBRATION_PREDICTIONS: Mapping[str, StitchedEpisode] | None = None
_CALIBRATION_TRUTH: Mapping[str, GroundTruthEpisode] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


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
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and value.get("cuda") is not None:
        torch.cuda.set_rng_state_all(value["cuda"])


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _implementation_release() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[2]
    manifest = (
        package_root
        / "configs"
        / "experimental"
        / "atomic_v1_wave1_release_sha256.txt"
    )
    if not manifest.is_file():
        raise ValueError(f"Wave-1 release manifest is missing: {manifest}")
    module_dir = Path(__file__).resolve().parent
    paths = {
        "trainer": module_dir / "formal500h_wave1_train.py",
        "data": module_dir / "formal500h_wave1_data.py",
        "runtime": module_dir / "formal500h_runtime.py",
        "evaluator": module_dir / "formal500h_evaluation.py",
        "inference": module_dir / "inference.py",
        "model": module_dir / "model.py",
        "baselines": module_dir / "baselines.py",
        "losses": module_dir / "losses.py",
    }
    return {
        "release_manifest": str(manifest),
        "release_manifest_sha256": sha256_file(manifest),
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }


def _model_config(variant: str, *, d_model: int, dropout: float) -> dict[str, Any]:
    if variant not in VARIANT_MODALITIES:
        raise ValueError(f"unknown Wave-1 variant: {variant}")
    return {
        "variant": "b2_early_fusion",
        "formal_variant": variant,
        "enabled_modalities": list(VARIANT_MODALITIES[variant]),
        "vision_dim": 1536,
        "tactile_dim": 360,
        "tactile_representation": "raw_delta_abs",
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
        "use_boundary_head": True,
        "output_rate_hz": 30,
        "event_rate_hz": 30,
        "clock_metadata": dict(clock_metadata()),
    }


class _IndexSampler(Sampler[int]):
    def __init__(self, indices: Sequence[int]):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _epoch_order(length: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed + 1_000_003 * epoch)
    return torch.randperm(length, generator=generator).tolist()


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _checkpoint_save(path: Path, value: Mapping[str, Any]) -> str:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    os.replace(temporary_sidecar, sidecar)
    return digest


def _checkpoint_load(path: Path, device: torch.device) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"checkpoint digest sidecar missing: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != expected:
        raise ValueError(f"checkpoint digest mismatch: {path}")
    value = torch.load(path, map_location=device, weights_only=False)
    if value.get("schema_version") != TRAINING_SCHEMA:
        raise ValueError("resume checkpoint has wrong schema")
    return value


def _validated_data(data_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    summary_path = data_dir / "summary.json"
    summary = _read_json(summary_path)
    if (
        summary.get("schema_version") != WAVE1_DATA_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("test_manifest_read") is not False
        or summary.get("class_count") != NUM_CLASSES
        or summary.get("episode_contract") != "formal500h_training_episode_v3"
    ):
        raise ValueError("Wave-1 formal data preflight failed")
    artifacts: dict[str, Path] = {}
    for name in (
        "train_windows",
        "val_windows",
        "train_episodes",
        "val_episodes",
        "train_statistics",
    ):
        record = (summary.get("artifacts") or {}).get(name) or {}
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Wave-1 data artifact changed: {name}")
        artifacts[name] = path
    if artifacts["train_windows"].parent != data_dir.resolve():
        raise ValueError("Wave-1 data summary points outside its frozen directory")
    return summary, artifacts


def _dataset(manifest: Path, statistics: Path) -> Formal500hDataset:
    rows = _read_jsonl(manifest)
    if any(row.get("schema_version") != WINDOW_SCHEMA for row in rows):
        raise ValueError("Wave-1 window manifest schema changed")
    return Formal500hDataset(
        manifest,
        statistics,
        expected_statistics_schema=WAVE1_DATA_SCHEMA,
    )


def _ground_truth(rows: Sequence[Mapping[str, Any]]) -> dict[str, GroundTruthEpisode]:
    output: dict[str, GroundTruthEpisode] = {}
    for index, row in enumerate(rows, start=1):
        episode_id = str(row.get("episode_id") or row.get("capture_key"))
        path = Path(str(row["cache_path"]))
        with h5py.File(path, "r") as handle:
            schema = handle.attrs.get("schema_version", "")
            if isinstance(schema, bytes):
                schema = schema.decode("utf-8")
            if str(schema) != "formal500h_training_episode_v3" or not bool(
                handle.attrs.get("complete", False)
            ):
                raise ValueError(f"invalid Formal500h validation HDF5: {path}")
            action_60 = np.asarray(handle["targets/action_class"], dtype=np.int64)
            instance_60 = np.asarray(
                handle["targets/action_instance"], dtype=np.int64
            )
            boundary_instance_60 = np.asarray(
                handle["targets/boundary_instance"], dtype=np.bool_
            )
            boundary_class_change_60 = np.asarray(
                handle["targets/boundary_class_change"], dtype=np.bool_
            )
            targets = downsample_targets_60_to_30(
                action_class_60=action_60,
                action_instance_60=instance_60,
                boundary_instance_60=boundary_instance_60,
                boundary_class_change_60=boundary_class_change_60,
            )
            action = np.asarray(targets["action_class"], dtype=np.int64)
            instance = np.asarray(targets["action_instance"], dtype=np.int64)
            boundary_hard = np.asarray(targets["boundary_instance"], dtype=np.bool_)
            boundary_change = np.asarray(
                targets["boundary_class_change"], dtype=np.bool_
            )
            time_ns = np.asarray(handle["timeline/time_ns"][::2], dtype=np.int64)
        if action.size < 2 or instance.shape != action.shape or time_ns.shape != action.shape:
            raise ValueError(f"invalid Formal500h validation target shapes: {path}")
        valid = (action >= 0) & (action < NUM_CLASSES) & (instance >= 0)
        if not valid.all():
            raise ValueError(f"invalid Formal500h validation target values: {path}")
        boundary_type = np.zeros(action.size, dtype=np.int8)
        boundary_type[boundary_hard & boundary_change] = 1
        boundary_type[boundary_hard & ~boundary_change] = 2
        truth = GroundTruthEpisode(
            episode_id=episode_id,
            global_frame_60=np.arange(action.size, dtype=np.int64),
            atomic_target=action,
            atomic_instance_id=instance,
            boundary_hard=boundary_hard,
            boundary_type=boundary_type,
            label_mask=valid,
            boundary_eval_mask=valid,
            time_ns=time_ns,
            metadata={
                "formal_split_group_id": row.get("formal_split_group_id"),
                "clock_metadata": dict(clock_metadata()),
            },
        )
        if truth.episode_id in output:
            raise ValueError(f"duplicate validation episode id: {truth.episode_id}")
        output[truth.episode_id] = truth
        if index % 500 == 0:
            print(
                json.dumps(
                    {
                        "event": "wave1_ground_truth_progress",
                        "episodes": index,
                        "total": len(rows),
                    }
                ),
                flush=True,
            )
    return output


@torch.no_grad()
def _predict_full_validation(
    model: nn.Module,
    dataset: Formal500hDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    precision: str,
) -> tuple[dict[str, StitchedEpisode], dict[str, float]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_formal500h,
    )
    amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    windows: list[WindowPrediction] = []
    hand_confusion = np.zeros((3, 3), dtype=np.int64)
    model.eval()
    for batch_index, batch in enumerate(loader, start=1):
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            output = model(batch)
        atomic = output["atomic_logits"].float().detach().cpu().numpy()
        boundary = torch.sigmoid(output["boundary_logits"].float()).detach().cpu().numpy()
        hand = output["hand_logits"].argmax(dim=-1).detach().cpu().numpy()
        hand_target = batch["hand_target"].detach().cpu().numpy()
        mask = batch["sequence_mask"].detach().cpu().numpy().astype(bool)
        global_frame = batch["global_frame_60"].detach().cpu().numpy()
        time_ns = batch["time_ns"].detach().cpu().numpy()
        for row_index, episode_id in enumerate(batch["episode_id"]):
            used = mask[row_index]
            truth_hand = hand_target[row_index, used]
            predicted_hand = hand[row_index, used]
            valid_hand = (truth_hand >= 0) & (truth_hand < 3)
            np.add.at(
                hand_confusion,
                (truth_hand[valid_hand], predicted_hand[valid_hand]),
                1,
            )
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
                        "event": "wave1_full_validation_progress",
                        "batches": batch_index,
                        "windows": len(windows),
                    }
                ),
                flush=True,
            )
    stitched = stitch_window_predictions(
        windows, weighting="uniform", require_full_coverage=True
    )
    support = hand_confusion.sum(axis=1)
    f1_values: list[float] = []
    for label in range(3):
        if support[label] <= 0:
            continue
        tp = float(hand_confusion[label, label])
        denominator = 2 * tp + float(hand_confusion[:, label].sum() - tp) + float(
            hand_confusion[label, :].sum() - tp
        )
        f1_values.append(2 * tp / denominator if denominator else 0.0)
    hand_total = int(hand_confusion.sum())
    return stitched, {
        "accuracy": 100.0 * float(np.trace(hand_confusion)) / max(hand_total, 1),
        "macro_f1": 100.0 * float(np.mean(f1_values)) if f1_values else 0.0,
    }


def _evaluate_predictions(
    predictions: Mapping[str, StitchedEpisode],
    truth: Mapping[str, GroundTruthEpisode],
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
        use_boundary_head=True,
        include_action_changes=True,
        strict=True,
        collect_artifacts=False,
    )
    return result.metrics


def _calibration_case(
    parameters: tuple[float, int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _CALIBRATION_PREDICTIONS is None or _CALIBRATION_TRUTH is None:
        raise RuntimeError("formal calibration worker was not initialized")
    threshold, nms, minimum = parameters
    metrics = _evaluate_predictions(
        _CALIBRATION_PREDICTIONS,
        _CALIBRATION_TRUTH,
        boundary_threshold=threshold,
        boundary_min_distance_frames=nms,
        min_segment_frames=minimum,
    )
    primary = metrics["primary"]
    record = {
        "boundary_threshold": threshold,
        "boundary_min_distance_frames": nms,
        "min_segment_frames": minimum,
        "primary_composite": float(primary["checkpoint_composite"]),
        "collapsed_semantic_f1@50": float(primary["collapsed_semantic_f1@50"]),
        "class_change_boundary_f1@250ms": float(
            primary["class_change_boundary_f1@250ms"]
        ),
        "raw_instance_f1@50": float(primary["raw_instance_f1@50"]),
    }
    return record, metrics


def _calibrate(
    predictions: Mapping[str, StitchedEpisode],
    truth: Mapping[str, GroundTruthEpisode],
    *,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("calibration workers must be positive")
    policy = FORMAL_EVALUATION_POLICY["decode_calibration"]
    cases = [
        (float(threshold), int(nms), int(minimum))
        for threshold in policy["boundary_threshold_candidates"]
        for nms in policy["boundary_min_distance_frames_30hz"]
        for minimum in policy["min_segment_frames_30hz"]
    ]
    if len(cases) != 96:
        raise AssertionError("frozen Formal500h calibration grid must have 96 cases")
    global _CALIBRATION_PREDICTIONS, _CALIBRATION_TRUTH
    _CALIBRATION_PREDICTIONS = predictions
    _CALIBRATION_TRUTH = truth
    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        try:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("fork"),
            )
            results: Iterable[tuple[dict[str, Any], dict[str, Any]]] = executor.map(
                _calibration_case,
                cases,
                chunksize=1,
            )
        except OSError as error:
            executor = None
            results = map(_calibration_case, cases)
            print(
                json.dumps(
                    {
                        "event": "wave1_calibration_parallel_fallback",
                        "requested_workers": workers,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                ),
                flush=True,
            )
    else:
        results = map(_calibration_case, cases)
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    try:
        for record, metrics in results:
            minimum = int(record["min_segment_frames"])
            nms = int(record["boundary_min_distance_frames"])
            threshold = float(record["boundary_threshold"])
            candidates.append(record)
            key = (
                record["primary_composite"],
                record["collapsed_semantic_f1@50"],
                record["class_change_boundary_f1@250ms"],
                record["raw_instance_f1@50"],
                -float(minimum),
                -float(nms),
                -abs(float(threshold) - 0.5),
                -float(threshold),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {"decoder": record, "metrics": metrics}
            print(
                json.dumps(
                    {
                        "event": "wave1_calibration_progress",
                        "completed": len(candidates),
                        "total": 96,
                        "current_primary": record["primary_composite"],
                        "best_primary": best["decoder"]["primary_composite"],
                    }
                ),
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        _CALIBRATION_PREDICTIONS = None
        _CALIBRATION_TRUTH = None
    assert best is not None
    return {
        "split": "validation_only",
        "test_manifest_read": False,
        "tie_break": policy["tie_break"],
        "workers_requested": workers,
        "workers_used": workers if executor is not None else 1,
        "candidates": candidates,
        "selected": best["decoder"],
        "metrics": best["metrics"],
    }


def _loss_config(statistics: Path) -> dict[str, Any]:
    return {
        "action_weight": 1.0,
        "smooth_weight": 0.1,
        "boundary_weight": 0.5,
        "subtask_weight": 0.0,
        "boundary_positive_weight": 8.0,
        "action_class_weights": formal500h_class_weights(statistics),
    }


def train_formal_variant(
    *,
    data_dir: Path,
    output_dir: Path,
    variant: str,
    seed: int,
    device_name: str = "cuda:0",
    precision: str = "bf16",
    batch_size: int = 16,
    num_workers: int = 16,
    epochs: int = 3,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-4,
    d_model: int = 128,
    dropout: float = 0.15,
    hand_weight: float = 0.1,
    checkpoint_interval: int = 2000,
    calibration_workers: int = 4,
    resume: bool = True,
) -> dict[str, Any]:
    variant = variant.upper()
    if variant not in VARIANT_MODALITIES:
        raise ValueError("variant must be B0, B1, or B2")
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16")
    if (
        batch_size < 1
        or num_workers < 0
        or epochs < 1
        or checkpoint_interval < 1
        or calibration_workers < 1
    ):
        raise ValueError("invalid positive training setting")
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        value = _read_json(summary_path)
        if value.get("status") == "complete":
            print(json.dumps({"event": "wave1_job_already_complete", "output_dir": str(output_dir)}), flush=True)
            return value
        raise ValueError(f"non-complete summary exists: {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_summary, artifacts = _validated_data(data_dir)
    train_dataset = _dataset(artifacts["train_windows"], artifacts["train_statistics"])
    val_dataset = _dataset(artifacts["val_windows"], artifacts["train_statistics"])
    val_episode_rows = _read_jsonl(artifacts["val_episodes"])
    if any(row.get("cache_path", "").endswith("test.jsonl") for row in val_episode_rows):
        raise ValueError("test leakage marker detected")
    # Load validation targets once; repeated epoch-level HDF5 scans would only
    # waste local-NVMe bandwidth and cannot change the frozen ground truth.
    truth = _ground_truth(val_episode_rows)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _seed_everything(seed)
    model_config = _model_config(variant, d_model=d_model, dropout=dropout)
    model = Formal500hModel(model_config).to(device)
    initial_model_sha256 = _model_state_sha256(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    loss_config = _loss_config(artifacts["train_statistics"])
    criterion = ContactTASLoss(loss_config).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    updates_per_epoch = math.ceil(len(train_dataset) / batch_size)
    total_updates = epochs * updates_per_epoch
    warmup = max(1, int(round(0.05 * total_updates)))

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1.0e-3, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = LambdaLR(optimizer, schedule)
    amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and precision == "fp16")
    implementation = _implementation_release()
    resolved = {
        "schema_version": TRAINING_SCHEMA,
        "purpose": "formal_reportable_wave1",
        "episode_contract": "formal500h_training_episode_v3",
        "test_manifest_read": False,
        "variant": variant,
        "enabled_modalities": list(VARIANT_MODALITIES[variant]),
        "seed": seed,
        "hostname": socket.gethostname(),
        "device": str(device),
        "precision": precision,
        "batch_size_per_gpu": batch_size,
        "num_workers": num_workers,
        "epochs": epochs,
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hand_weight": hand_weight,
        "checkpoint_interval": checkpoint_interval,
        "calibration_workers": calibration_workers,
        "model": model_config,
        "loss": loss_config,
        "parameter_count": parameter_count,
        "initial_model_sha256": initial_model_sha256,
        "data_summary": str((data_dir / "summary.json").resolve()),
        "data_summary_sha256": sha256_file(data_dir / "summary.json"),
        "source_protocol_sha256": data_summary["source_protocol_sha256"],
        "source_audit_sha256": data_summary["source_audit_sha256"],
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
        "evaluation_policy": FORMAL_EVALUATION_POLICY,
        "implementation_release": implementation,
        "hand_order_policy": "HDF5 already [left,right]; no training-time swap",
        "clock_metadata": dict(clock_metadata()),
    }
    resolved_path = output_dir / "resolved_config.json"
    if resolved_path.exists() and _read_json(resolved_path) != resolved:
        raise ValueError("resume configuration differs from frozen run configuration")
    _atomic_json(resolved_path, resolved)

    metrics_path = output_dir / "metrics.jsonl"
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    epoch = 0
    sample_offset = 0
    update = 0
    best_score = float("-inf")
    best_epoch = -1
    best_fixed_metrics: dict[str, Any] = {}
    if resume and last_path.is_file():
        checkpoint = _checkpoint_load(last_path, device)
        if checkpoint.get("resolved_config") != resolved:
            raise ValueError("resume checkpoint does not belong to this formal run")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        epoch = int(checkpoint["next_epoch"])
        sample_offset = int(checkpoint["next_sample_offset"])
        update = int(checkpoint["update"])
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        best_fixed_metrics = dict(checkpoint.get("best_fixed_metrics") or {})
        _restore_rng_state(checkpoint["rng_state"])
        print(json.dumps({"event": "wave1_resume", "variant": variant, "seed": seed, "epoch": epoch, "sample_offset": sample_offset, "update": update}), flush=True)

    def checkpoint_value(next_epoch: int, next_offset: int) -> dict[str, Any]:
        return {
            "schema_version": TRAINING_SCHEMA,
            "resolved_config": resolved,
            "variant": variant,
            "seed": seed,
            "update": update,
            "next_epoch": next_epoch,
            "next_sample_offset": next_offset,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_fixed_metrics": best_fixed_metrics,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_state": _rng_state(),
        }

    started = time.time()
    while epoch < epochs:
        order = _epoch_order(len(train_dataset), seed, epoch)
        if not 0 <= sample_offset <= len(order):
            raise ValueError("resume sample offset is outside the epoch")
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=_IndexSampler(order[sample_offset:]),
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=collate_formal500h,
        )
        model.train()
        for batch in loader:
            actual = len(batch["episode_id"])
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                output = model(batch)
                terms = formal500h_loss_terms(criterion, output, batch, hand_weight)
            scaler.scale(terms["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update += 1
            sample_offset += actual
            if update % 100 == 0:
                record = {
                    "event": "wave1_train_progress",
                    "variant": variant,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "update": update,
                    "total_updates": total_updates,
                    "sample_offset": sample_offset,
                    "epoch_samples": len(train_dataset),
                    "loss": float(terms["loss"].detach().float().cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_s": time.time() - started,
                }
                _append_jsonl(metrics_path, record)
                print(json.dumps(record), flush=True)
            if update % checkpoint_interval == 0 and sample_offset < len(order):
                _checkpoint_save(last_path, checkpoint_value(epoch, sample_offset))

        epoch += 1
        sample_offset = 0
        predictions, hand_metrics = _predict_full_validation(
            model,
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            precision=precision,
        )
        fixed_metrics = _evaluate_predictions(
            predictions,
            truth,
            boundary_threshold=0.5,
            boundary_min_distance_frames=6,
            min_segment_frames=9,
        )
        score = float(fixed_metrics["primary"]["checkpoint_composite"])
        validation_record = {
            "event": "wave1_full_episode_validation",
            "variant": variant,
            "seed": seed,
            "epoch": epoch,
            "update": update,
            "decoder": {
                "boundary_threshold": 0.5,
                "boundary_min_distance_frames": 6,
                "min_segment_frames": 9,
            },
            "metrics": fixed_metrics,
            "hand": hand_metrics,
            "elapsed_s": time.time() - started,
        }
        _append_jsonl(metrics_path, validation_record)
        print(
            json.dumps(
                {
                    "event": validation_record["event"],
                    "variant": variant,
                    "seed": seed,
                    "epoch": epoch,
                    "update": update,
                    "primary": fixed_metrics["primary"],
                    "hand": hand_metrics,
                    "elapsed_s": validation_record["elapsed_s"],
                }
            ),
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_fixed_metrics = fixed_metrics
            _checkpoint_save(best_path, checkpoint_value(epoch, 0))
        _checkpoint_save(last_path, checkpoint_value(epoch, 0))

    best_checkpoint = _checkpoint_load(best_path, device)
    model.load_state_dict(best_checkpoint["model_state"])
    predictions, hand_metrics = _predict_full_validation(
        model,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        precision=precision,
    )
    # Calibration is CPU-only.  Release the scarce GPU before the frozen
    # 96-case validation grid so the ninth Wave-1 job can start immediately.
    if device.type == "cuda":
        model.to("cpu")
        criterion.to("cpu")
        del optimizer, scheduler, scaler
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    gpu_release = {
        "schema_version": TRAINING_SCHEMA,
        "status": "gpu_released_before_validation_calibration",
        "variant": variant,
        "seed": seed,
        "update": update,
        "test_manifest_read": False,
        "created_unix_s": time.time(),
    }
    _atomic_json(output_dir / "GPU_RELEASED.json", gpu_release)
    print(json.dumps({"event": "wave1_gpu_released", **gpu_release}), flush=True)
    calibration = _calibrate(predictions, truth, workers=calibration_workers)
    calibration["hand"] = hand_metrics
    calibration_path = output_dir / "validation_calibration.json"
    _atomic_json(calibration_path, calibration)
    summary = {
        "schema_version": TRAINING_SCHEMA,
        "status": "complete",
        "purpose": "formal_reportable_wave1_validation_selected",
        "episode_contract": "formal500h_training_episode_v3",
        "test_manifest_read": False,
        "variant": variant,
        "enabled_modalities": list(VARIANT_MODALITIES[variant]),
        "seed": seed,
        "epochs": epochs,
        "updates": update,
        "best_epoch": best_epoch,
        "best_fixed_primary_composite": best_score,
        "best_fixed_metrics": best_fixed_metrics,
        "calibrated_validation": calibration,
        "elapsed_s": time.time() - started,
        "parameter_count": parameter_count,
        "initial_model_sha256": initial_model_sha256,
        "final_model_sha256": _model_state_sha256(model),
        "resolved_config": str(resolved_path.resolve()),
        "resolved_config_sha256": sha256_file(resolved_path),
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path.resolve()),
        "last_checkpoint_sha256": sha256_file(last_path),
        "validation_calibration": str(calibration_path.resolve()),
        "validation_calibration_sha256": sha256_file(calibration_path),
        "data_summary_sha256": resolved["data_summary_sha256"],
        "source_protocol_sha256": resolved["source_protocol_sha256"],
        "source_audit_sha256": resolved["source_audit_sha256"],
        "release_manifest_sha256": implementation["release_manifest_sha256"],
        "hand_order_policy": resolved["hand_order_policy"],
        "gpu_released_before_validation_calibration": True,
        "clock_metadata": dict(clock_metadata()),
    }
    _atomic_json(summary_path, summary)
    completion = {
        "schema_version": TRAINING_SCHEMA,
        "status": "complete",
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256_file(best_path),
        "test_manifest_read": False,
    }
    _atomic_json(output_dir / "COMPLETE.json", completion)
    print(
        json.dumps(
            {
                "event": "formal500h_wave1_training_complete",
                "status": "complete",
                "variant": variant,
                "seed": seed,
                "epochs": epochs,
                "updates": update,
                "best_epoch": best_epoch,
                "calibrated_primary": calibration["metrics"]["primary"],
                "test_manifest_read": False,
                "summary": str(summary_path.resolve()),
            }
        ),
        flush=True,
    )
    return summary


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def aggregate_wave1(*, runs_root: Path, output: Path, seeds: Sequence[int] = DEFAULT_SEEDS) -> dict[str, Any]:
    summaries: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in VARIANT_MODALITIES}
    for variant in VARIANT_MODALITIES:
        for seed in seeds:
            path = runs_root / variant / f"seed_{seed}" / "summary.json"
            value = _read_json(path)
            if (
                value.get("schema_version") != TRAINING_SCHEMA
                or value.get("status") != "complete"
                or value.get("variant") != variant
                or value.get("seed") != seed
                or value.get("test_manifest_read") is not False
                or value.get("episode_contract") != "formal500h_training_episode_v3"
            ):
                raise ValueError(f"invalid Wave-1 completion: {path}")
            complete_path = path.parent / "COMPLETE.json"
            complete = _read_json(complete_path)
            resolved_path = Path(str(value.get("resolved_config", "")))
            best_path = Path(str(value.get("best_checkpoint", "")))
            calibration = value.get("calibrated_validation") or {}
            selected = calibration.get("selected") or {}
            policy = FORMAL_EVALUATION_POLICY["decode_calibration"]
            if (
                complete.get("summary_sha256") != sha256_file(path)
                or complete.get("test_manifest_read") is not False
                or not resolved_path.is_file()
                or sha256_file(resolved_path) != value.get("resolved_config_sha256")
                or not best_path.is_file()
                or sha256_file(best_path) != value.get("best_checkpoint_sha256")
                or value.get("enabled_modalities") != list(VARIANT_MODALITIES[variant])
                or int(value.get("epochs", -1)) != 3
                or len(calibration.get("candidates") or []) != 96
                or selected.get("boundary_threshold") not in policy["boundary_threshold_candidates"]
                or selected.get("boundary_min_distance_frames") not in policy["boundary_min_distance_frames_30hz"]
                or selected.get("min_segment_frames") not in policy["min_segment_frames_30hz"]
            ):
                raise ValueError(f"incomplete formal provenance/calibration: {path}")
            summaries[variant][seed] = value
    lineage = {
        (
            value["data_summary_sha256"],
            value["source_protocol_sha256"],
            value["source_audit_sha256"],
            value["release_manifest_sha256"],
        )
        for by_seed in summaries.values() for value in by_seed.values()
    }
    if len(lineage) != 1:
        raise ValueError("Wave-1 runs do not share one frozen data/protocol lineage")
    parameter_counts = {
        int(value["parameter_count"])
        for by_seed in summaries.values()
        for value in by_seed.values()
    }
    updates = {
        int(value["updates"])
        for by_seed in summaries.values()
        for value in by_seed.values()
    }
    paired_initialization = all(
        len({summaries[variant][seed]["initial_model_sha256"] for variant in VARIANT_MODALITIES}) == 1
        for seed in seeds
    )
    if len(parameter_counts) != 1 or len(updates) != 1 or not paired_initialization:
        raise ValueError("Wave-1 variants are not parameter/update/initialization matched")
    metrics = {
        "primary_composite": lambda value: value["calibrated_validation"]["metrics"]["primary"]["checkpoint_composite"],
        "collapsed_f1@50": lambda value: value["calibrated_validation"]["metrics"]["primary"]["collapsed_semantic_f1@50"],
        "class_change_boundary_f1@250ms": lambda value: value["calibrated_validation"]["metrics"]["primary"]["class_change_boundary_f1@250ms"],
        "raw_instance_f1@50": lambda value: value["calibrated_validation"]["metrics"]["primary"]["raw_instance_f1@50"],
        "frame_accuracy": lambda value: value["calibrated_validation"]["metrics"]["frame_accuracy"],
        "tick_macro_f1": lambda value: value["calibrated_validation"]["metrics"]["tick_macro_f1"],
        "hand_macro_f1": lambda value: value["calibrated_validation"]["hand"]["macro_f1"],
    }
    aggregate: dict[str, Any] = {}
    for variant, by_seed in summaries.items():
        aggregate[variant] = {
            name: _mean_std([float(getter(by_seed[seed])) for seed in seeds])
            for name, getter in metrics.items()
        }
    paired: dict[str, Any] = {}
    for first, second in (("B0", "B1"), ("B1", "B2"), ("B0", "B2")):
        key = f"{second}_minus_{first}"
        paired[key] = {}
        for name, getter in metrics.items():
            differences = [float(getter(summaries[second][seed])) - float(getter(summaries[first][seed])) for seed in seeds]
            paired[key][name] = {**_mean_std(differences), "values": differences, "wins": sum(value > 0 for value in differences)}
    data_hash, protocol_hash, audit_hash, release_hash = next(iter(lineage))
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "complete",
        "purpose": "formal_wave1_validation_result_test_still_sealed",
        "episode_contract": "formal500h_training_episode_v3",
        "test_manifest_read": False,
        "variants": list(VARIANT_MODALITIES),
        "seeds": list(seeds),
        "jobs": len(VARIANT_MODALITIES) * len(seeds),
        "controls": {
            "parameter_count": next(iter(parameter_counts)),
            "updates_per_job": next(iter(updates)),
            "paired_initialization_within_seed": paired_initialization,
            "data_summary_sha256": data_hash,
            "source_protocol_sha256": protocol_hash,
            "source_audit_sha256": audit_hash,
            "release_manifest_sha256": release_hash,
            "full_episode_validation": True,
            "validation_calibration_candidates": 96,
            "test_sealed": True,
        },
        "aggregate": aggregate,
        "paired": paired,
        "decision_scope": "select tactile baseline conclusion and inputs to Wave 2 using validation only",
    }
    _atomic_json(output, result)
    _atomic_json(output.parent / "WAVE1_COMPLETE.json", {**result, "aggregate_path": str(output.resolve()), "aggregate_sha256": sha256_file(output)})
    print(
        json.dumps(
            {
                "event": "formal500h_wave1_aggregate_complete",
                "status": "complete",
                "jobs": result["jobs"],
                "seeds": result["seeds"],
                "controls": result["controls"],
                "primary": {
                    variant: result["aggregate"][variant]["primary_composite"]
                    for variant in VARIANT_MODALITIES
                },
                "paired_primary": {
                    name: values["primary_composite"]
                    for name, values in result["paired"].items()
                },
                "test_manifest_read": False,
                "output": str(output.resolve()),
            }
        ),
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent Formal500h 26-class Wave-1 trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--variant", choices=sorted(VARIANT_MODALITIES), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--num-workers", type=int, default=16)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=3.0e-4)
    train.add_argument("--weight-decay", type=float, default=1.0e-4)
    train.add_argument("--d-model", type=int, default=128)
    train.add_argument("--dropout", type=float, default=0.15)
    train.add_argument("--hand-weight", type=float, default=0.1)
    train.add_argument("--checkpoint-interval", type=int, default=2000)
    train.add_argument("--calibration-workers", type=int, default=4)
    train.add_argument("--no-resume", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--runs-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--seeds", default=" ".join(map(str, DEFAULT_SEEDS)))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "train":
        train_formal_variant(
            data_dir=arguments.data_dir,
            output_dir=arguments.output_dir,
            variant=arguments.variant,
            seed=arguments.seed,
            device_name=arguments.device,
            precision=arguments.precision,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            d_model=arguments.d_model,
            dropout=arguments.dropout,
            hand_weight=arguments.hand_weight,
            checkpoint_interval=arguments.checkpoint_interval,
            calibration_workers=arguments.calibration_workers,
            resume=not arguments.no_resume,
        )
    else:
        seeds = tuple(int(value) for value in arguments.seeds.replace(",", " ").split())
        aggregate_wave1(runs_root=arguments.runs_root, output=arguments.output, seeds=seeds)


if __name__ == "__main__":
    main()
