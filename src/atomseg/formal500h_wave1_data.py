from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np

from .formal500h_protocol import NUM_CLASSES, PROTOCOL_SCHEMA, SPLIT_SCHEMA, sha256_file
from .formal500h_clock import clock_metadata, downsample_targets_60_to_30
from .formal500h_runtime import EPISODE_SCHEMA, REQUIRED_DATASETS


WAVE1_DATA_SCHEMA = "formal500h_wave1_data_v1"
WINDOW_SCHEMA = "formal500h_wave1_window_v1"
OUTPUT_RATE_HZ = 30
SOURCE_RATE_HZ = 60
DEFAULT_WINDOW_SECONDS = 8.0
DEFAULT_STRIDE_SECONDS = 4.0


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
        raise ValueError(f"empty split manifest: {path}")
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


def _artifact(protocol: Mapping[str, Any], name: str, protocol_dir: Path) -> Path:
    record = (protocol.get("artifacts") or {}).get(name) or {}
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        path = protocol_dir / path
    path = path.resolve()
    expected = str(record.get("sha256", ""))
    if not path.is_file() or len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"frozen protocol artifact changed or is absent: {name}")
    return path


def _window_starts(output_steps: int, window_steps: int, stride_steps: int) -> list[int]:
    if output_steps <= 0:
        raise ValueError("episode contains no 30 Hz output ticks")
    if output_steps <= window_steps:
        return [0]
    starts = list(range(0, output_steps - window_steps + 1, stride_steps))
    final = output_steps - window_steps
    if starts[-1] != final:
        starts.append(final)
    return starts


def _empty_statistics() -> dict[str, np.ndarray | int]:
    return {
        "tactile_count": np.zeros((2, 360), dtype=np.float64),
        "tactile_sum": np.zeros((2, 360), dtype=np.float64),
        "tactile_square": np.zeros((2, 360), dtype=np.float64),
        "skeleton_count": np.zeros((2, 25, 7), dtype=np.float64),
        "skeleton_sum": np.zeros((2, 25, 7), dtype=np.float64),
        "skeleton_square": np.zeros((2, 25, 7), dtype=np.float64),
        "kinematics_count": np.zeros((48,), dtype=np.float64),
        "kinematics_sum": np.zeros((48,), dtype=np.float64),
        "kinematics_square": np.zeros((48,), dtype=np.float64),
        "action_count": np.zeros((NUM_CLASSES,), dtype=np.int64),
        "hand_count": np.zeros((3,), dtype=np.int64),
        "boundary_count": 0,
        "boundary_count_60": 0,
        "boundary_count_30": 0,
        "output_ticks": 0,
    }


def _update_moments(
    statistics: dict[str, np.ndarray | int],
    prefix: str,
    values: np.ndarray,
    valid: np.ndarray,
) -> None:
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float64, copy=False
    )
    weight = valid.astype(np.float64, copy=False)
    while weight.ndim < clean.ndim:
        weight = weight[..., None]
    weight = np.broadcast_to(weight, clean.shape)
    statistics[f"{prefix}_count"] += weight.sum(axis=0)
    statistics[f"{prefix}_sum"] += (clean * weight).sum(axis=0)
    statistics[f"{prefix}_square"] += (clean * clean * weight).sum(axis=0)


def _validate_split_row(row: Mapping[str, Any], split: str) -> None:
    if row.get("formal_split_schema") != SPLIT_SCHEMA:
        raise ValueError("formal split row schema changed")
    if row.get("formal_split_name") != split:
        raise ValueError(f"row escaped frozen {split} split")
    if not str(row.get("formal_split_group_id", "")):
        raise ValueError("formal split row lacks group id")
    summary = row.get("formal_episode_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("formal split row lacks episode summary")
    classes = [int(value) for value in summary.get("class_indices", [])]
    if not classes or min(classes) < 0 or max(classes) >= NUM_CLASSES:
        raise ValueError("formal split row has invalid 26-class labels")


def _process_chunk(
    payload: tuple[str, list[dict[str, Any]], int, int, bool]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray | int], list[dict[str, Any]]]:
    split, rows, window_steps, stride_steps, collect_statistics = payload
    statistics = _empty_statistics()
    windows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for row in rows:
        _validate_split_row(row, split)
        path = Path(str(row.get("cache_path", "")))
        if not path.is_file():
            raise ValueError(f"formal HDF5 does not exist: {path}")
        with h5py.File(path, "r") as handle:
            schema = handle.attrs.get("schema_version", "")
            if isinstance(schema, bytes):
                schema = schema.decode("utf-8")
            if str(schema) != EPISODE_SCHEMA or not bool(
                handle.attrs.get("complete", False)
            ):
                raise ValueError(f"invalid formal HDF5 completion/schema: {path}")
            missing = [name for name in REQUIRED_DATASETS if name not in handle]
            if missing:
                raise ValueError(f"formal HDF5 lacks {missing}: {path}")
            metadata_raw = handle.attrs.get("metadata_json", "{}")
            if isinstance(metadata_raw, bytes):
                metadata_raw = metadata_raw.decode("utf-8")
            metadata = json.loads(str(metadata_raw))
            if metadata.get("tactile_hand_order") != ["left", "right"]:
                raise ValueError(f"HDF5 hand order is not frozen [left,right]: {path}")
            ticks_60 = int(handle["targets/action_class"].shape[0])
            expected_ticks = int(
                (row.get("formal_episode_summary") or {}).get("ticks_60", -1)
            )
            if ticks_60 < 1 or expected_ticks != ticks_60:
                raise ValueError(f"HDF5 length disagrees with frozen protocol: {path}")
            output_steps = (ticks_60 + 1) // 2
            targets = downsample_targets_60_to_30(
                action_class_60=np.asarray(
                    handle["targets/action_class"], dtype=np.int64
                ),
                action_instance_60=np.asarray(
                    handle["targets/action_instance"], dtype=np.int64
                ),
                boundary_instance_60=np.asarray(
                    handle["targets/boundary_instance"], dtype=np.bool_
                ),
                boundary_class_change_60=np.asarray(
                    handle["targets/boundary_class_change"], dtype=np.bool_
                ),
            )
            action_all = np.asarray(targets["action_class"], dtype=np.int64)
            boundary_all = np.asarray(targets["boundary_instance"], dtype=np.bool_)
            change_all = np.asarray(
                targets["boundary_class_change"], dtype=np.bool_
            )
            if action_all.shape != (output_steps,) or np.any(
                (action_all < 0) | (action_all >= NUM_CLASSES)
            ):
                raise ValueError(f"invalid 26-class targets: {path}")
            episode_id = str(row.get("episode_id") or row.get("capture_key"))
            for start_30 in _window_starts(
                output_steps, window_steps, stride_steps
            ):
                end_30 = min(start_30 + window_steps, output_steps)
                start_60 = 2 * start_30
                end_60 = min(2 * end_30, ticks_60)
                actions = action_all[start_30:end_30]
                boundaries = boundary_all[start_30:end_30]
                changes = change_all[start_30:end_30]
                raw_boundaries = np.asarray(
                    handle["targets/boundary_instance"][start_60:end_60],
                    dtype=np.bool_,
                )
                windows.append(
                    {
                        "schema_version": WINDOW_SCHEMA,
                        "formal_split": split,
                        "episode_id": episode_id,
                        "capture_key": str(row.get("capture_key")),
                        "task_id": str(row.get("task_id", "")),
                        "group_id": str(row["formal_split_group_id"]),
                        "mcap_path": str(row.get("mcap_path", "")),
                        "cache_path": str(path),
                        "start_frame_30": int(start_30),
                        "end_frame_30_exclusive": int(end_30),
                        "start_frame_60": int(start_60),
                        "end_frame_60_exclusive": int(end_60),
                        "output_ticks_30": int(end_30 - start_30),
                        "clock_metadata": dict(clock_metadata()),
                        "class_indices": sorted(
                            {int(value) for value in actions.tolist()}
                        ),
                        "boundary_count_60": int(raw_boundaries.sum()),
                        "boundary_count_30": int(boundaries.sum()),
                        "boundary_class_change_count_30": int(changes.sum()),
                        # Retain the raw count for audits that compare source
                        # and output clocks; this is intentionally distinct
                        # from the model-facing 30 Hz count above.
                        "boundary_source_count_60": int(raw_boundaries.sum()),
                    }
                )
            episodes.append(
                {
                    "episode_id": episode_id,
                    "capture_key": str(row.get("capture_key")),
                    "group_id": str(row["formal_split_group_id"]),
                    "task_id": str(row.get("task_id", "")),
                    "cache_path": str(path),
                    "ticks_60": ticks_60,
                    "ticks_30": output_steps,
                    "classes": sorted({int(value) for value in action_all.tolist()}),
                    "clock_metadata": dict(clock_metadata()),
                }
            )
            if collect_statistics:
                tactile = np.asarray(
                    handle["sensors/tactile/value"][::2], dtype=np.float32
                )
                tactile_valid = np.asarray(
                    handle["sensors/tactile/valid"][::2], dtype=np.bool_
                )
                skeleton = np.asarray(
                    handle["sensors/skeleton/relative"][::2], dtype=np.float32
                )
                skeleton_valid = np.asarray(
                    handle["sensors/skeleton/valid"][::2], dtype=np.bool_
                )
                joint_valid = np.asarray(
                    handle["sensors/skeleton/joint_valid"][::2], dtype=np.bool_
                )
                kinematics = np.asarray(
                    handle["sensors/kinematics/value"][::2], dtype=np.float32
                )
                kinematics_valid = np.asarray(
                    handle["sensors/kinematics/valid"][::2], dtype=np.bool_
                )
                hands = np.asarray(
                    handle["targets/action_hand"][::2], dtype=np.int64
                )
                _update_moments(
                    statistics, "tactile", tactile, tactile_valid
                )
                _update_moments(
                    statistics,
                    "skeleton",
                    skeleton,
                    joint_valid & skeleton_valid[..., None],
                )
                _update_moments(
                    statistics, "kinematics", kinematics, kinematics_valid
                )
                statistics["action_count"] += np.bincount(
                    action_all, minlength=NUM_CLASSES
                )[:NUM_CLASSES]
                statistics["hand_count"] += np.bincount(
                    hands, minlength=3
                )[:3]
                statistics["boundary_count"] += int(
                    np.asarray(
                        handle["targets/boundary_instance"], dtype=np.bool_
                    ).sum()
                )
                statistics["boundary_count_60"] += int(
                    np.asarray(
                        handle["targets/boundary_instance"], dtype=np.bool_
                    ).sum()
                )
                statistics["boundary_count_30"] += int(boundary_all.sum())
                statistics["output_ticks"] += output_steps
    return windows, statistics, episodes


def _merge_statistics(
    destination: dict[str, np.ndarray | int], source: Mapping[str, np.ndarray | int]
) -> None:
    for name in destination:
        destination[name] += source[name]


def _finish_moments(
    statistics: Mapping[str, np.ndarray | int], prefix: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = np.asarray(statistics[f"{prefix}_count"], dtype=np.float64)
    total = np.asarray(statistics[f"{prefix}_sum"], dtype=np.float64)
    square = np.asarray(statistics[f"{prefix}_square"], dtype=np.float64)
    denominator = np.maximum(count, 1.0)
    mean = total / denominator
    variance = np.maximum(square / denominator - mean * mean, 1.0e-8)
    standard = np.sqrt(variance)
    standard[count < 2] = 1.0
    return (
        mean.astype(np.float32),
        standard.astype(np.float32),
        count.astype(np.int64),
    )


def _write_statistics(
    path: Path, statistics: Mapping[str, np.ndarray | int]
) -> dict[str, Any]:
    tactile_mean, tactile_std, tactile_count = _finish_moments(
        statistics, "tactile"
    )
    skeleton_mean, skeleton_std, skeleton_count = _finish_moments(
        statistics, "skeleton"
    )
    kinematics_mean, kinematics_std, kinematics_count = _finish_moments(
        statistics, "kinematics"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(WAVE1_DATA_SCHEMA),
        tactile_mean=tactile_mean,
        tactile_std=tactile_std,
        tactile_count=tactile_count,
        skeleton_mean=skeleton_mean,
        skeleton_std=skeleton_std,
        skeleton_count=skeleton_count,
        kinematics_mean=kinematics_mean,
        kinematics_std=kinematics_std,
        kinematics_count=kinematics_count,
        action_count=np.asarray(statistics["action_count"], dtype=np.int64),
        hand_count=np.asarray(statistics["hand_count"], dtype=np.int64),
        boundary_count=np.asarray(int(statistics["boundary_count"])),
        boundary_count_60=np.asarray(int(statistics["boundary_count_60"])),
        boundary_count_30=np.asarray(int(statistics["boundary_count_30"])),
        clock_metadata_json=np.asarray(
            json.dumps(dict(clock_metadata()), sort_keys=True)
        ),
        output_ticks_30=np.asarray(int(statistics["output_ticks"])),
    )
    os.replace(temporary, path)
    ticks = int(statistics["output_ticks"])
    return {
        "output_ticks_30": ticks,
        "source_hours": ticks / OUTPUT_RATE_HZ / 3600.0,
        "action_count": np.asarray(statistics["action_count"]).astype(int).tolist(),
        "hand_count": np.asarray(statistics["hand_count"]).astype(int).tolist(),
        "boundary_count_60": int(statistics["boundary_count_60"]),
        "boundary_count_30": int(statistics["boundary_count_30"]),
        "clock_metadata": dict(clock_metadata()),
        "sha256": sha256_file(path),
    }


def _chunks(rows: Sequence[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    count = max(1, min(len(rows), workers * 4))
    size = int(math.ceil(len(rows) / count))
    return [list(rows[start : start + size]) for start in range(0, len(rows), size)]


def _process_split(
    rows: list[dict[str, Any]],
    *,
    split: str,
    window_steps: int,
    stride_steps: int,
    workers: int,
    collect_statistics: bool,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray | int], list[dict[str, Any]]]:
    payloads = [
        (split, chunk, window_steps, stride_steps, collect_statistics)
        for chunk in _chunks(rows, workers)
    ]
    if workers == 1:
        results = [_process_chunk(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_process_chunk, payloads))
    windows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    statistics = _empty_statistics()
    for index, (part_windows, part_statistics, part_episodes) in enumerate(
        results, start=1
    ):
        windows.extend(part_windows)
        episodes.extend(part_episodes)
        _merge_statistics(statistics, part_statistics)
        print(
            json.dumps(
                {
                    "event": "wave1_data_progress",
                    "split": split,
                    "chunks": index,
                    "total_chunks": len(results),
                    "episodes": len(episodes),
                    "windows": len(windows),
                }
            ),
            flush=True,
        )
    windows.sort(key=lambda row: (str(row["capture_key"]), int(row["start_frame_30"])))
    episodes.sort(key=lambda row: str(row["capture_key"]))
    return windows, statistics, episodes


def prepare_wave1_data(
    *,
    protocol_dir: Path,
    output_dir: Path,
    workers: int = 16,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
) -> dict[str, Any]:
    protocol_dir = protocol_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Wave-1 data: {output_dir}")
    if workers < 1:
        raise ValueError("workers must be positive")
    window_steps = int(round(window_seconds * OUTPUT_RATE_HZ))
    stride_steps = int(round(stride_seconds * OUTPUT_RATE_HZ))
    if (
        window_steps < 2
        or stride_steps < 1
        or stride_steps > window_steps
        or not math.isclose(window_steps / OUTPUT_RATE_HZ, window_seconds)
        or not math.isclose(stride_steps / OUTPUT_RATE_HZ, stride_seconds)
    ):
        raise ValueError("window/stride must be exact positive 30 Hz durations")

    protocol_path = protocol_dir / "protocol.json"
    protocol = _read_json(protocol_path)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA or protocol.get("status") != "frozen":
        raise ValueError("Formal500h protocol is not frozen v2")
    audit_record = protocol.get("audit") or {}
    audit_path = Path(str(audit_record.get("path", "")))
    if not audit_path.is_absolute():
        audit_path = protocol_dir / audit_path
    audit_path = audit_path.resolve()
    if not audit_path.is_file() or sha256_file(audit_path) != audit_record.get("sha256"):
        raise ValueError("frozen protocol audit changed")
    audit = _read_json(audit_path)
    checks = audit.get("checks") or {}
    if audit.get("status") != "pass" or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("frozen protocol audit did not pass")

    # Deliberately resolve and read only train/validation.  The sealed test
    # artifact is not opened, hashed, stat'ed, or copied in Wave 1.
    train_path = _artifact(protocol, "train", protocol_dir)
    val_path = _artifact(protocol, "val", protocol_dir)
    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    train_groups = {str(row.get("formal_split_group_id")) for row in train_rows}
    val_groups = {str(row.get("formal_split_group_id")) for row in val_rows}
    if train_groups & val_groups:
        raise ValueError("frozen train/validation groups overlap")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".tmp", dir=output_dir.parent)
    )
    try:
        started = time.time()
        train_windows, statistics, train_episodes = _process_split(
            train_rows,
            split="train",
            window_steps=window_steps,
            stride_steps=stride_steps,
            workers=workers,
            collect_statistics=True,
        )
        val_windows, _, val_episodes = _process_split(
            val_rows,
            split="val",
            window_steps=window_steps,
            stride_steps=stride_steps,
            workers=workers,
            collect_statistics=False,
        )
        if len(train_episodes) != len(train_rows) or len(val_episodes) != len(val_rows):
            raise AssertionError("episode processing was not complete")
        train_classes = sorted(
            {value for row in train_episodes for value in row["classes"]}
        )
        if train_classes != list(range(NUM_CLASSES)):
            raise ValueError(f"full train split does not contain all 26 classes: {train_classes}")
        if len({(row["cache_path"], row["start_frame_30"]) for row in train_windows}) != len(train_windows):
            raise ValueError("duplicate full-train windows")
        if len({(row["cache_path"], row["start_frame_30"]) for row in val_windows}) != len(val_windows):
            raise ValueError("duplicate full-validation windows")

        train_manifest = temporary / "train_windows.jsonl"
        val_manifest = temporary / "val_windows.jsonl"
        train_episode_manifest = temporary / "train_episodes.jsonl"
        val_episode_manifest = temporary / "val_episodes.jsonl"
        statistics_path = temporary / "train_statistics.npz"
        _atomic_jsonl(train_manifest, train_windows)
        _atomic_jsonl(val_manifest, val_windows)
        _atomic_jsonl(train_episode_manifest, train_episodes)
        _atomic_jsonl(val_episode_manifest, val_episodes)
        statistics_summary = _write_statistics(statistics_path, statistics)
        summary = {
            "schema_version": WAVE1_DATA_SCHEMA,
            "status": "complete",
            "purpose": "formal_reportable_wave1_training",
            "created_unix_s": time.time(),
            "elapsed_s": time.time() - started,
            "hostname": socket.gethostname(),
            "source_protocol": str(protocol_path),
            "source_protocol_sha256": sha256_file(protocol_path),
            "source_audit": str(audit_path),
            "source_audit_sha256": sha256_file(audit_path),
            "test_manifest_read": False,
            "test_policy": "sealed test artifact was not opened, hashed, stat'ed, or copied",
            "class_count": NUM_CLASSES,
            "output_rate_hz": OUTPUT_RATE_HZ,
            "source_rate_hz": SOURCE_RATE_HZ,
            "clock_metadata": dict(clock_metadata()),
            "clock_implementation_sha256": sha256_file(
                Path(__file__).with_name("formal500h_clock.py")
            ),
            "data_builder_sha256": sha256_file(Path(__file__)),
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "window_frames_30": window_steps,
            "stride_frames_30": stride_steps,
            "workers": workers,
            "train": {
                "episodes": len(train_episodes),
                "windows": len(train_windows),
                "groups": len(train_groups),
                "class_indices": train_classes,
                "source_hours": sum(row["ticks_30"] for row in train_episodes) / OUTPUT_RATE_HZ / 3600.0,
            },
            "val": {
                "episodes": len(val_episodes),
                "windows": len(val_windows),
                "groups": len(val_groups),
                "class_indices": sorted({value for row in val_episodes for value in row["classes"]}),
                "source_hours": sum(row["ticks_30"] for row in val_episodes) / OUTPUT_RATE_HZ / 3600.0,
            },
            "statistics": statistics_summary,
            "artifacts": {
                "train_windows": {"path": str(output_dir / train_manifest.name), "sha256": sha256_file(train_manifest)},
                "val_windows": {"path": str(output_dir / val_manifest.name), "sha256": sha256_file(val_manifest)},
                "train_episodes": {"path": str(output_dir / train_episode_manifest.name), "sha256": sha256_file(train_episode_manifest)},
                "val_episodes": {"path": str(output_dir / val_episode_manifest.name), "sha256": sha256_file(val_episode_manifest)},
                "train_statistics": {"path": str(output_dir / statistics_path.name), "sha256": sha256_file(statistics_path)},
            },
            "training_budget": {
                "epochs": 3,
                "batch_size_per_gpu": 16,
                "updates_per_epoch": math.ceil(len(train_windows) / 16),
                "total_updates": 3 * math.ceil(len(train_windows) / 16),
                "definition": "three deterministic complete passes over the frozen full-train window universe",
            },
            "hand_order_policy": "HDF5 already [left,right]; no training-time swap",
            "episode_contract": EPISODE_SCHEMA,
        }
        _atomic_json(temporary / "summary.json", summary)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "event": "formal500h_wave1_data_ready",
                "status": summary["status"],
                "train": summary["train"],
                "val": summary["val"],
                "training_budget": summary["training_budget"],
                "test_manifest_read": False,
                "episode_contract": EPISODE_SCHEMA,
                "summary": str(output_dir / "summary.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the frozen full Formal500h Wave-1 train/validation universe")
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--stride-seconds", type=float, default=DEFAULT_STRIDE_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    prepare_wave1_data(
        protocol_dir=arguments.protocol_dir,
        output_dir=arguments.output_dir,
        workers=arguments.workers,
        window_seconds=arguments.window_seconds,
        stride_seconds=arguments.stride_seconds,
    )


if __name__ == "__main__":
    main()
