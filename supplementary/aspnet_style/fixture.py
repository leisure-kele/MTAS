"""Synthetic HDF5 fixture reused from project tests; no real data."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import h5py
import numpy as np
from atomseg.formal500h_runtime import EPISODE_SCHEMA
from atomseg.formal500h_protocol import PROTOCOL_SCHEMA, SPLIT_SCHEMA, sha256_file
def _put(handle: h5py.File, name: str, value: np.ndarray) -> None:
    parent, leaf = name.rsplit("/", 1)
    handle.require_group(parent).create_dataset(leaf, data=value)

def _episode(path: Path, episode_id: str, labels: np.ndarray) -> None:
    ticks = int(labels.size)
    rng = np.random.default_rng(int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16))
    instance = np.arange(ticks, dtype=np.int32) // 4
    boundary = np.zeros(ticks, dtype=np.bool_)
    boundary[1:] = instance[1:] != instance[:-1]
    change = np.zeros(ticks, dtype=np.bool_)
    change[1:] = labels[1:] != labels[:-1]
    vision_steps = (ticks + 1) // 2
    vision_index = np.repeat((np.arange(ticks) // 2)[:, None], 3, axis=1)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = EPISODE_SCHEMA
        handle.attrs["complete"] = True
        handle.attrs["metadata_json"] = json.dumps(
            {
                "episode_id": episode_id,
                "task_id": "T001",
                "task_name": "fixture",
                "tactile_hand_order": ["left", "right"],
            }
        )
        _put(handle, "timeline/time_ns", np.arange(ticks, dtype=np.int64) * 16666667)
        _put(handle, "targets/action_class", labels.astype(np.int16))
        _put(handle, "targets/action_hand", (np.arange(ticks) % 3).astype(np.int8))
        _put(handle, "targets/action_instance", instance)
        _put(handle, "targets/boundary_instance", boundary)
        _put(handle, "targets/boundary_class_change", change & boundary)
        _put(handle, "annotations/atomic_segments/class_index", np.unique(labels).astype(np.int16))
        _put(handle, "vision/features", rng.normal(size=(vision_steps, 3, 1536)).astype(np.float16))
        _put(handle, "vision/valid", np.ones((vision_steps, 3), dtype=np.bool_))
        _put(handle, "vision/index_60", vision_index.astype(np.int32))
        _put(handle, "vision/aligned_valid_60", np.ones((ticks, 3), dtype=np.bool_))
        _put(handle, "vision/freshness_ms_60", np.zeros((ticks, 3), dtype=np.float32))
        _put(handle, "sensors/tactile/value", rng.normal(size=(ticks, 2, 360)).astype(np.float32))
        _put(handle, "sensors/tactile/valid", np.ones((ticks, 2), dtype=np.bool_))
        quality = np.zeros((ticks, 2, 5), dtype=np.float32)
        quality[..., :2] = 1.0
        _put(handle, "sensors/tactile/quality_raw460", quality)
        _put(handle, "sensors/skeleton/relative", rng.normal(size=(ticks, 2, 25, 7)).astype(np.float32))
        _put(handle, "sensors/skeleton/valid", np.ones((ticks, 2), dtype=np.bool_))
        _put(handle, "sensors/skeleton/joint_valid", np.ones((ticks, 2, 25), dtype=np.bool_))
        _put(handle, "sensors/kinematics/value", rng.normal(size=(ticks, 48)).astype(np.float32))
        _put(handle, "sensors/kinematics/valid", np.ones(ticks, dtype=np.bool_))
        _put(handle, "sensors/pose/quality", np.ones((ticks, 6), dtype=np.float32))

def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

def _protocol(root: Path) -> tuple[Path, Path]:
    protocol_dir = root / "protocol"
    episodes = root / "episodes"
    protocol_dir.mkdir()
    episodes.mkdir()
    rows_by_split: dict[str, list[dict[str, object]]] = {"train": [], "val": []}
    for split, count in (("train", 2), ("val", 1)):
        for index in range(count):
            episode_id = f"{split}_{index}"
            labels = (
                np.repeat(np.arange(26, dtype=np.int16), 4)
                if split == "train"
                else np.repeat(np.arange(8, dtype=np.int16), 8)
            )
            path = episodes / f"{episode_id}.h5"
            _episode(path, episode_id, labels)
            group = f"{split}_group_{index}"
            rows_by_split[split].append(
                {
                    "capture_key": episode_id,
                    "episode_id": episode_id,
                    "task_id": "T001",
                    "mcap_path": f"/source/{episode_id}.mcap",
                    "cache_path": str(path),
                    "formal_split_schema": SPLIT_SCHEMA,
                    "formal_split_name": split,
                    "formal_split_group_id": group,
                    "formal_episode_summary": {
                        "ticks_60": int(labels.size),
                        "duration_s": labels.size / 60.0,
                        "class_indices": sorted({int(value) for value in labels}),
                    },
                }
            )
    train = protocol_dir / "train.jsonl"
    val = protocol_dir / "val.jsonl"
    test = protocol_dir / "test.jsonl"
    _jsonl(train, rows_by_split["train"])
    _jsonl(val, rows_by_split["val"])
    test.write_text("sealed and intentionally invalid\n", encoding="utf-8")
    audit = protocol_dir / "audit.json"
    audit.write_text(json.dumps({"schema_version": PROTOCOL_SCHEMA, "status": "pass", "checks": {"all": True}}), encoding="utf-8")
    protocol = protocol_dir / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema_version": PROTOCOL_SCHEMA,
                "status": "frozen",
                "audit": {"path": str(audit), "sha256": sha256_file(audit)},
                "artifacts": {
                    "train": {"path": str(train), "sha256": sha256_file(train)},
                    "val": {"path": str(val), "sha256": sha256_file(val)},
                    "test": {"path": str(test), "sha256": "0" * 64},
                },
            }
        ),
        encoding="utf-8",
    )
    return protocol_dir, test
