"""Eight-GPU, resumable exporter for the frozen HumanTouch1 formal-500h corpus.

The source MCAP files are opened read-only.  The authoritative correction
manifest is keyed by the *exact absolute MCAP path*.  Tactile correction is
performed on complete 460-channel left/right streams before the hand-specific
360-channel mappings are applied.

One HDF5 file is written per episode.  Labels and sensors use the 60 Hz label
clock; DINOv3 features use a compact 30 Hz clock and include precomputed gather
indices back to 60 Hz.  No normalization is performed here because train-only
statistics cannot be known before the leakage-safe split is frozen.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import queue
import re
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .exporter import _build_kinematics, _relative_skeleton


SCHEMA_VERSION = "formal500h_training_episode_v3"
RANK_STATE_SCHEMA = "formal500h_export_rank_state_v1"
PAUSE_SCHEMA = "formal500h_export_pause_v1"
CORRECTION_SCHEMA = "formal500h_tactile_correction_v1"
MAPPING_SCHEMA = "humantouch_360_mapping_v1"
MCAP_CENSUS_SCHEMA = "formal500h_mcap_schema_census_v1"
TACTILE_LAYOUT = "humantouch-360-v1"
EXPECTED_CORRECTION_SHA256 = (
    "91271fbf49e1bdbd1902c11fd7d7d89953ab847a52ce09caa4022ef73f294829"
)
EXPECTED_MAPPING_SHA256 = (
    "d1d43a0d34f94af2ee37d0cd14ff3233b17460cdfe7f11b43951327d2d806679"
)
EXPECTED_DINOV3_SHA256 = (
    "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b"
)
EXPECTED_ONTOLOGY_SHA256 = (
    "7c1f1a7a391a9488334d9ababbb5605451a041c522b81ec39ef7721f4a54f246"
)

VIDEO_TOPICS = ("/ego_video/head", "/ego_video/left", "/ego_video/right")
TACTILE_TOPICS = ("/tactile/raw/left", "/tactile/raw/right")
SKELETON_TOPIC = "/tracking/manus_skeleton_raw"
POSE_TOPIC = "/tracking/vive_tracker_manus"
TIME_TOPIC = "/time_sync/info"
INPUT_TOPICS = VIDEO_TOPICS + TACTILE_TOPICS + (SKELETON_TOPIC, POSE_TOPIC, TIME_TOPIC)
VIEW_INDEX = {topic: index for index, topic in enumerate(VIDEO_TOPICS)}
HAND_INDEX = {"left": 0, "right": 1}
POSE_BLOCKS = ("head", "left_wrist", "right_wrist")
HAND_TARGET = {"left": 0, "right": 1, "both": 2}
TACTILE_FIELD_CODE = {"missing": 0, "pressureData": 1, "values": 2}
TACTILE_TOPIC_CODE = {TACTILE_TOPICS[0]: 0, TACTILE_TOPICS[1]: 1}
TACTILE_DEVICE_CODE = {"left": 1, "right": 2}
TACTILE_PROTOCOL_CODE = {"moxian": 1}


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Publish a small control-plane JSON file without exposing a partial write."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{os.getpid()}.partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output)


def _validated_stop_file(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("--stop-file must be an absolute path shared by every rank")
    return path


def _stop_requested(stop_file: Path | None) -> bool:
    """Return true only at an explicit episode boundary.

    Callers deliberately invoke this before starting an episode, never from
    inside MCAP decoding, GPU inference, or HDF5 writing.  A request arriving
    during an episode is therefore observed only after that episode has been
    atomically committed and its result row has been flushed.
    """

    return stop_file is not None and stop_file.is_file()


def _rank_state_path(run_dir: str | Path, rank: int) -> Path:
    return Path(run_dir) / f"rank{rank:02d}.state.json"


def _write_rank_state(
    run_dir: str | Path,
    *,
    rank: int,
    world_size: int,
    state: str,
    shard_size: int,
    successful_rows: int,
    failed_rows: int,
    stop_file: Path | None,
    next_shard_index: int | None = None,
    next_mcap_path: str | None = None,
) -> dict[str, Any]:
    if state not in {"running", "complete", "planned_pause", "failed"}:
        raise ValueError(f"invalid rank export state: {state!r}")
    payload = {
        "schema_version": RANK_STATE_SCHEMA,
        "state": state,
        "rank": int(rank),
        "world_size": int(world_size),
        "shard_size": int(shard_size),
        "successful_rows": int(successful_rows),
        "failed_rows": int(failed_rows),
        "next_shard_index": next_shard_index,
        "next_mcap_path": next_mcap_path,
        "stop_file": str(stop_file) if stop_file is not None else None,
        "updated_unix_s": time.time(),
    }
    _atomic_write_json(_rank_state_path(run_dir, rank), payload)
    return payload


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _require_absolute_posix_path(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    parsed = PurePosixPath(value)
    if not parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"{field_name} must be an absolute normalized path: {value!r}")
    return value


@dataclass(frozen=True)
class Authority:
    rows: tuple[dict[str, Any], ...]
    by_mcap_path: Mapping[str, dict[str, Any]]
    left_indices: np.ndarray
    right_indices: np.ndarray
    correction_sha256: str
    mapping_sha256: str


@dataclass(frozen=True)
class McapSchemaContract:
    """Frozen server-formal MCAP schemas accepted by the exporter."""

    report_path: str
    report_sha256: str
    authority_manifest_sha256: str
    inspected_file_count: int
    allowed_channel_signatures: Mapping[str, frozenset[tuple[str, str, str, str]]]


def _channel_signature(schema: Any, channel: Any) -> tuple[str, str, str, str]:
    schema_data = bytes(schema.data) if schema is not None else b""
    return (
        str(channel.message_encoding),
        str(schema.name) if schema is not None else "",
        str(schema.encoding) if schema is not None else "",
        hashlib.sha256(schema_data).hexdigest() if schema is not None else "",
    )


def load_mcap_schema_contract(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_authority_sha256: str,
    required_topics: Sequence[str] = INPUT_TOPICS,
    min_inspected_files: int = 32,
) -> McapSchemaContract:
    """Load a server-generated schema census and reject guesses/local samples."""

    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(
            "expected_mcap_schema_report_sha256 is not frozen; run the formal server "
            "MCAP schema census before export"
        )
    report_path = Path(path)
    observed_sha256 = sha256_file(report_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"formal MCAP schema report SHA256 mismatch: {observed_sha256} != {expected_sha256}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != MCAP_CENSUS_SCHEMA:
        raise ValueError("unexpected formal MCAP schema census version")
    if report.get("authority_manifest_sha256") != expected_authority_sha256:
        raise ValueError("schema census was not generated from the frozen correction manifest")
    if report.get("errors"):
        raise ValueError("formal MCAP schema census contains file inspection errors")
    inspected = int((report.get("selection") or {}).get("sample_files_inspected", 0))
    if inspected < min_inspected_files:
        raise ValueError(
            f"schema census inspected only {inspected} files; require at least {min_inspected_files}"
        )
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("formal MCAP schema census lacks aggregate results")
    if aggregate.get("formal_tactile_contract_violations"):
        raise ValueError("formal tactile schema contract violations exist in the census")
    if aggregate.get("formal_atomic_contract_violations"):
        raise ValueError("formal atomic JSON contract violations exist in the census")
    missing_tactile = aggregate.get("missing_formal_tactile_topics") or {}
    if any(missing_tactile.values()):
        raise ValueError("formal tactile topics are missing from one or more census files")
    topic_counts = aggregate.get("topic_file_counts") or {}
    variants = aggregate.get("variants_by_topic") or {}
    files = report.get("files")
    if not isinstance(files, list) or len(files) != inspected:
        raise ValueError("formal MCAP schema census file details are incomplete")
    for file_record in files:
        topics = file_record.get("topics") if isinstance(file_record, Mapping) else None
        if not isinstance(topics, Mapping):
            raise ValueError("formal MCAP schema census contains an invalid file record")
        for topic in required_topics:
            topic_record = topics.get(topic)
            if not isinstance(topic_record, Mapping):
                raise ValueError(f"required topic {topic!r} lacks per-file census details")
            if topic_record.get("decode_errors"):
                raise ValueError(
                    f"required topic {topic!r} has payload decode errors in the schema census"
                )
            if not topic_record.get("samples"):
                raise ValueError(
                    f"required topic {topic!r} has no decoded payload sample in the schema census"
                )
    allowed: dict[str, frozenset[tuple[str, str, str, str]]] = {}
    for topic in required_topics:
        if int(topic_counts.get(topic, 0)) != inspected:
            raise ValueError(
                f"required formal topic {topic!r} was present in {topic_counts.get(topic, 0)}/"
                f"{inspected} census files"
            )
        topic_variants = variants.get(topic)
        if not isinstance(topic_variants, list) or not topic_variants:
            raise ValueError(f"schema census has no variants for required topic {topic!r}")
        signatures = {
            (
                str(variant.get("message_encoding") or ""),
                str(variant.get("schema_name") or ""),
                str(variant.get("schema_encoding") or ""),
                str(variant.get("schema_sha256") or ""),
            )
            for variant in topic_variants
        }
        if any(not all(signature) for signature in signatures):
            raise ValueError(f"incomplete channel signature for formal topic {topic!r}")
        allowed[topic] = frozenset(signatures)
    return McapSchemaContract(
        report_path=str(report_path),
        report_sha256=observed_sha256,
        authority_manifest_sha256=expected_authority_sha256,
        inspected_file_count=inspected,
        allowed_channel_signatures=allowed,
    )


def assert_formal_channel(
    contract: McapSchemaContract, schema: Any, channel: Any
) -> tuple[str, str, str, str]:
    signature = _channel_signature(schema, channel)
    allowed = contract.allowed_channel_signatures.get(channel.topic)
    if allowed is None or signature not in allowed:
        raise ValueError(
            f"unapproved formal MCAP channel schema for {channel.topic}: {signature}"
        )
    return signature


def _mapping_indices(regions: Sequence[Mapping[str, Any]]) -> np.ndarray:
    indices: list[int] = []
    for region in regions:
        kind = region.get("type")
        if kind == "grid":
            rows = region.get("row_start_indices")
            columns = region.get("columns")
            if not isinstance(rows, Mapping) or not isinstance(columns, Mapping):
                raise ValueError("grid mapping region lacks row/column ranges")
            for row in range(int(rows["start"]), int(rows["stop_exclusive"]), int(rows["step"])):
                for column in range(int(columns["start"]), int(columns["stop_exclusive"])):
                    indices.append(row + column)
        elif kind == "contiguous_range":
            indices.extend(range(int(region["start"]), int(region["stop_exclusive"])))
        else:
            raise ValueError(f"unknown tactile mapping region type: {kind!r}")
    output = np.asarray(indices, dtype=np.int64)
    if output.shape != (360,):
        raise ValueError(f"mapping must select exactly 360 points, got {output.shape}")
    if np.any(np.diff(output) <= 0):
        raise ValueError("within-hand tactile mapping must be strictly ascending")
    if int(output[0]) < 0 or int(output[-1]) >= 460:
        raise ValueError("tactile mapping contains an index outside [0,460)")
    return output


def load_authority(
    correction_path: str | Path,
    mapping_path: str | Path,
    *,
    expected_correction_sha256: str = EXPECTED_CORRECTION_SHA256,
    expected_mapping_sha256: str = EXPECTED_MAPPING_SHA256,
    require_frozen_counts: bool = True,
) -> Authority:
    correction = Path(correction_path)
    mapping_file = Path(mapping_path)
    correction_hash = sha256_file(correction)
    mapping_hash = sha256_file(mapping_file)
    if expected_correction_sha256 and correction_hash != expected_correction_sha256:
        raise ValueError(
            f"correction manifest SHA256 mismatch: {correction_hash} != "
            f"{expected_correction_sha256}"
        )
    if expected_mapping_sha256 and mapping_hash != expected_mapping_sha256:
        raise ValueError(
            f"tactile mapping SHA256 mismatch: {mapping_hash} != {expected_mapping_sha256}"
        )

    rows: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    atomic_paths: set[str] = set()
    actions = {"keep": 0, "swap_left_right": 0}
    with correction.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"correction line {line_number} is not an object")
            path = _require_absolute_posix_path(row.get("mcap_path"), "mcap_path")
            atomic_path = _require_absolute_posix_path(
                row.get("atomic_json_path"), "atomic_json_path"
            )
            if path in by_path:
                raise ValueError(f"duplicate absolute mcap_path at line {line_number}: {path}")
            if atomic_path in atomic_paths:
                raise ValueError(
                    f"duplicate absolute atomic_json_path at line {line_number}: {atomic_path}"
                )
            if row.get("tactile_correction_schema") != CORRECTION_SCHEMA:
                raise ValueError(f"wrong correction schema at line {line_number}")
            if row.get("tactile_layout") != TACTILE_LAYOUT:
                raise ValueError(f"wrong tactile layout at line {line_number}")
            if row.get("tactile_processing_unit") != "whole_mcap":
                raise ValueError(f"segmented tactile correction is forbidden: {path}")
            if any(key in row for key in ("segments", "intervals", "swap_intervals")):
                raise ValueError(f"correction row contains forbidden segment fields: {path}")
            swap = row.get("tactile_swap_left_right")
            action = row.get("tactile_hand_correction")
            expected_action = "swap_left_right" if swap is True else "keep"
            if type(swap) is not bool or action != expected_action:
                raise ValueError(f"inconsistent tactile decision for {path}")
            actions[action] += 1
            rows.append(row)
            by_path[path] = row
            atomic_paths.add(atomic_path)
    if require_frozen_counts:
        expected = {"keep": 49_736, "swap_left_right": 7_839}
        if len(rows) != 57_575 or actions != expected:
            raise ValueError(
                f"frozen correction counts changed: rows={len(rows)}, actions={actions}"
            )

    mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
    if mapping.get("schema_version") != MAPPING_SCHEMA:
        raise ValueError("unexpected tactile mapping schema")
    if mapping.get("layout_version") != TACTILE_LAYOUT:
        raise ValueError("correction and mapping tactile layouts differ")
    if mapping.get("raw_dimension_per_hand") != 460:
        raise ValueError("raw tactile dimension must be 460 per hand")
    if mapping.get("output_dimension_per_hand") != 360:
        raise ValueError("output tactile dimension must be 360 per hand")
    output = mapping.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("mapping lacks output contract")
    if output.get("hand_order") != ["left", "right"]:
        raise ValueError("output tactile hand order must be [left,right]")
    if output.get("within_hand_order") != "ascending_raw_index":
        raise ValueError("tactile output order must preserve ascending raw indices")
    if output.get("mirror") is not False or output.get("reorder_after_selection") is not False:
        raise ValueError("tactile mapping must not mirror or reorder within a hand")
    left = _mapping_indices(mapping["left"]["valid_regions"])
    right = _mapping_indices(mapping["right"]["valid_regions"])
    return Authority(
        rows=tuple(rows),
        by_mcap_path=by_path,
        left_indices=left,
        right_indices=right,
        correction_sha256=correction_hash,
        mapping_sha256=mapping_hash,
    )


def _first_present(payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _frame_index(payload: Mapping[str, Any]) -> int | None:
    value = _first_present(payload, ("frame_60", "frameIndex", "frame_index"))
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _sensor_time_ns(payload: Mapping[str, Any], log_time_ns: int) -> int:
    value = _first_present(
        payload,
        ("session_time_ns", "timestampNs", "timestamp_ns", "sessionTimeNs"),
    )
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else int(log_time_ns)


def _valid_flag(payload: Mapping[str, Any], default: bool = True) -> bool:
    value = _first_present(payload, ("valid", "isValid", "is_valid"))
    return bool(value) if isinstance(value, (bool, int)) else default


def _numeric_quality(payload: Mapping[str, Any]) -> float:
    value = _first_present(
        payload, ("quality", "confidence", "qualityScore", "signalQuality")
    )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float("nan")


def _vector(value: Any, size: int, field_name: str) -> np.ndarray:
    if isinstance(value, Mapping):
        names = ("x", "y", "z", "w")[:size]
        value = [value.get(name) for name in names]
    output = np.asarray(value, dtype=np.float32)
    if output.shape != (size,) or not np.all(np.isfinite(output)):
        raise ValueError(f"{field_name} must be a finite {size}-vector")
    return output


@dataclass
class Sample:
    value: np.ndarray
    valid: bool
    frame_index: int | None
    log_time_ns: int
    sensor_time_ns: int
    source_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def tactile_values(
    payload: Mapping[str, Any],
    topic: str,
    *,
    allow_legacy_values: bool = False,
) -> tuple[np.ndarray, str]:
    """Read formal pressureData; an explicitly approved legacy schema may use values."""

    if "pressureData" in payload:
        field_name = "pressureData"
    elif "values" in payload and allow_legacy_values:
        field_name = "values"
    else:
        raise ValueError(
            f"{topic} lacks pressureData; values fallback is not approved for this schema"
        )
    candidate = payload[field_name]
    if not isinstance(candidate, list):
        raise ValueError(f"{topic}.{field_name} must be a list")
    if field_name == "pressureData" and "pressureDataLength" in payload:
        if payload["pressureDataLength"] != len(candidate):
            raise ValueError(
                f"{topic} pressureDataLength={payload['pressureDataLength']!r}, "
                f"actual={len(candidate)}"
            )
    output = np.asarray(candidate, dtype=np.float32)
    if output.shape != (460,):
        raise ValueError(f"{topic}.{field_name} has shape {output.shape}, expected (460,)")
    if not np.all(np.isfinite(output)):
        raise ValueError(f"{topic}.{field_name} contains NaN or infinity")
    return output, field_name


def _parse_tactile(
    payload: Mapping[str, Any],
    topic: str,
    log_time_ns: int,
    index: int,
    *,
    schema_name: str,
    legacy_schema_names: frozenset[str],
) -> Sample:
    if schema_name == "egopro.ProcessedTactileSample":
        required = {
            "frameIndex",
            "timestampNs",
            "hand",
            "device",
            "tactileProtocol",
            "pressureDataLength",
            "pressureData",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"{topic} formal tactile payload lacks fields: {missing}")
    values, source_field = tactile_values(
        payload,
        topic,
        allow_legacy_values=schema_name in legacy_schema_names,
    )
    expected_hand = topic.rsplit("/", 1)[-1]
    payload_hand = payload.get("hand")
    hand_match = payload_hand == expected_hand
    device = str(payload.get("device", "")).lower()
    protocol = str(payload.get("tactileProtocol", "")).lower()
    payload_hand_code = TACTILE_DEVICE_CODE.get(str(payload_hand).lower(), 3 if payload_hand else 0)
    device_code = TACTILE_DEVICE_CODE.get(device, 3 if device else 0)
    protocol_code = TACTILE_PROTOCOL_CODE.get(protocol, 2 if protocol else 0)
    return Sample(
        value=values,
        valid=_valid_flag(payload) and hand_match,
        frame_index=_frame_index(payload),
        log_time_ns=int(log_time_ns),
        sensor_time_ns=_sensor_time_ns(payload, log_time_ns),
        source_index=index,
        metadata={
            "field_code": TACTILE_FIELD_CODE[source_field],
            "topic_code": TACTILE_TOPIC_CODE[topic],
            "payload_hand_match": hand_match,
            "payload_hand_code": payload_hand_code,
            "device_code": device_code,
            "protocol_code": protocol_code,
            "payload_quality": _numeric_quality(payload),
        },
    )


def _required_int(payload: Mapping[str, Any], name: str, context: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context}.{name} must be an integer")
    return int(value)


def _parse_formal_pose_block(
    payload: Mapping[str, Any], block_name: str, log_time_ns: int, index: int
) -> Sample:
    """Parse one exact block from egopro.ProcessedViveFrame."""

    topic = POSE_TOPIC
    outer_frame = _required_int(payload, "frameIndex", topic)
    outer_time = _required_int(payload, "timestampNs", topic)
    block = payload.get(block_name)
    if not isinstance(block, Mapping):
        raise ValueError(f"{topic}.{block_name} must be an object")
    required = {
        "frameIndex",
        "timestampNs",
        "role",
        "position",
        "orientation",
        "quality",
        "qualityId",
        "trackerType",
        "trackerTypeId",
        "coordinateFrame",
    }
    missing = sorted(required - set(block))
    if missing:
        raise ValueError(f"{topic}.{block_name} lacks fields: {missing}")
    native_frame = _required_int(block, "frameIndex", f"{topic}.{block_name}")
    native_time = _required_int(block, "timestampNs", f"{topic}.{block_name}")
    if native_frame != outer_frame:
        raise ValueError(f"{topic}.{block_name}.frameIndex differs from outer frameIndex")
    if block.get("role") != block_name:
        raise ValueError(f"{topic}.{block_name}.role is {block.get('role')!r}")
    expected_tracker = {
        "head": ("head", 1),
        "left_wrist": ("left_hand", 3),
        "right_wrist": ("right_hand", 4),
    }[block_name]
    if (block.get("trackerType"), block.get("trackerTypeId")) != expected_tracker:
        raise ValueError(f"{topic}.{block_name} tracker type is inconsistent")
    if block.get("coordinateFrame") != "manus_core_unity_world":
        raise ValueError(f"{topic}.{block_name} uses an unexpected coordinate frame")
    value = np.concatenate(
        [
            _vector(block["position"], 3, f"{topic}.{block_name}.position"),
            _vector(block["orientation"], 4, f"{topic}.{block_name}.orientation"),
        ]
    )
    quality_id = _required_int(block, "qualityId", f"{topic}.{block_name}")
    valid = block.get("quality") == "trackable" and quality_id == 2
    return Sample(
        value=value.astype(np.float32),
        valid=valid,
        frame_index=outer_frame,
        log_time_ns=int(log_time_ns),
        # The outer timestamp is the 60 Hz aggregate-frame clock.  The nested
        # timestamp is the tracker observation time and is allowed to differ.
        sensor_time_ns=native_time,
        source_index=index,
        metadata={
            "quality_id": quality_id,
            "outer_timestamp_ns": outer_time,
            "native_frame_index": native_frame,
            "native_minus_outer_time_ns": native_time - outer_time,
        },
    )


def _parse_formal_skeleton_hand(
    payload: Mapping[str, Any], hand: str, log_time_ns: int, index: int, joints: int
) -> Sample:
    """Parse one exact hand block from egopro.ProcessedManusFrame."""

    topic = SKELETON_TOPIC
    outer_frame = _required_int(payload, "frameIndex", topic)
    outer_time = _required_int(payload, "timestampNs", topic)
    hand_payload = payload.get(hand)
    if not isinstance(hand_payload, Mapping):
        raise ValueError(f"{topic}.{hand} must be an object")
    required = {"frameIndex", "timestampNs", "hand", "bones"}
    missing = sorted(required - set(hand_payload))
    if missing:
        raise ValueError(f"{topic}.{hand} lacks fields: {missing}")
    if hand_payload.get("hand") != hand:
        raise ValueError(f"{topic}.{hand}.hand is {hand_payload.get('hand')!r}")
    native_frame = _required_int(hand_payload, "frameIndex", f"{topic}.{hand}")
    native_time = _required_int(hand_payload, "timestampNs", f"{topic}.{hand}")
    if native_frame != outer_frame:
        raise ValueError(f"{topic}.{hand}.frameIndex differs from outer frameIndex")
    rows = hand_payload.get("bones")
    if not isinstance(rows, list) or len(rows) != joints:
        raise ValueError(f"{topic}.{hand}.bones must contain exactly {joints} rows")
    value = np.zeros((joints, 7), dtype=np.float32)
    joint_valid = np.ones(joints, dtype=np.bool_)
    bone_names: list[str] = []
    for joint_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{topic}.{hand}.bones[{joint_index}] is not an object")
        required_bone = {"name", "position", "rotation"}
        missing_bone = sorted(required_bone - set(row))
        if missing_bone:
            raise ValueError(
                f"{topic}.{hand}.bones[{joint_index}] lacks fields: {missing_bone}"
            )
        value[joint_index] = np.concatenate(
            [
                _vector(
                    row["position"],
                    3,
                    f"{topic}.{hand}.bones[{joint_index}].position",
                ),
                _vector(
                    row["rotation"],
                    4,
                    f"{topic}.{hand}.bones[{joint_index}].rotation",
                ),
            ]
        )
        bone_names.append(str(row["name"]))
    return Sample(
        value=value,
        valid=True,
        frame_index=outer_frame,
        log_time_ns=int(log_time_ns),
        # The outer timestamp locates the combined 60 Hz frame.  Preserve the
        # nested hand timestamp as the actual source-observation timestamp.
        sensor_time_ns=native_time,
        source_index=index,
        metadata={
            "joint_valid": joint_valid,
            "bone_names": tuple(bone_names),
            "outer_timestamp_ns": outer_time,
            "native_frame_index": native_frame,
            "native_minus_outer_time_ns": native_time - outer_time,
        },
    )


def _parse_formal_time_sync(
    payload: Mapping[str, Any], log_time_ns: int, index: int
) -> Sample:
    required = {"frameIndex", "timestampNs", "masterUnixNs"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{TIME_TOPIC} lacks fields: {missing}")
    frame_index = _required_int(payload, "frameIndex", TIME_TOPIC)
    timestamp_ns = _required_int(payload, "timestampNs", TIME_TOPIC)
    master_unix_ns = _required_int(payload, "masterUnixNs", TIME_TOPIC)
    return Sample(
        value=np.zeros(1, dtype=np.float32),
        valid=True,
        frame_index=frame_index,
        log_time_ns=int(log_time_ns),
        sensor_time_ns=timestamp_ns,
        source_index=index,
        metadata={"master_unix_ns": master_unix_ns},
    )


@dataclass
class AlignedStream:
    value: np.ndarray
    valid: np.ndarray
    source_time_ns: np.ndarray
    source_log_time_ns: np.ndarray
    source_frame_index: np.ndarray
    source_message_index: np.ndarray
    freshness_ms: np.ndarray
    extras: dict[str, np.ndarray]


def _alignment_locations(
    samples: Sequence[Sample], time_ns: np.ndarray, tolerance_ns: int
) -> list[tuple[int, Sample]]:
    if not samples:
        return []
    explicit = [sample.frame_index for sample in samples if sample.frame_index is not None]
    frame_offset = 0
    if len(explicit) == len(samples) and explicit:
        unique = sorted(set(int(value) for value in explicit))
        if unique[0] != 0:
            if len(unique) == len(time_ns) and unique[-1] - unique[0] == len(time_ns) - 1:
                frame_offset = unique[0]
            else:
                raise ValueError(
                    f"non-zero/non-contiguous frame index cannot be aligned safely: "
                    f"min={unique[0]}, max={unique[-1]}, count={len(unique)}, T={len(time_ns)}"
                )
    output: list[tuple[int, Sample]] = []
    for sample in samples:
        if sample.frame_index is not None:
            location = int(sample.frame_index) - frame_offset
        else:
            insertion = int(np.searchsorted(time_ns, sample.log_time_ns, side="left"))
            right = min(max(insertion, 0), len(time_ns) - 1)
            left = min(max(insertion - 1, 0), len(time_ns) - 1)
            location = right if abs(int(time_ns[right]) - sample.log_time_ns) < abs(
                int(time_ns[left]) - sample.log_time_ns
            ) else left
            if abs(int(time_ns[location]) - sample.log_time_ns) > tolerance_ns:
                continue
        if 0 <= location < len(time_ns):
            output.append((location, sample))
    return output


def align_samples(
    samples: Sequence[Sample],
    time_ns: np.ndarray,
    value_shape: tuple[int, ...],
    *,
    tolerance_fraction: float = 0.45,
    extra_specs: Mapping[str, tuple[Any, Any]] | None = None,
) -> AlignedStream:
    period_ns = int(round(1.0e9 / 60.0))
    locations = _alignment_locations(samples, time_ns, int(period_ns * tolerance_fraction))
    value = np.zeros((len(time_ns),) + value_shape, dtype=np.float32)
    valid = np.zeros(len(time_ns), dtype=np.bool_)
    source_time = np.full(len(time_ns), -1, dtype=np.int64)
    source_log_time = np.full(len(time_ns), -1, dtype=np.int64)
    source_frame = np.full(len(time_ns), -1, dtype=np.int64)
    source_message = np.full(len(time_ns), -1, dtype=np.int64)
    freshness = np.full(len(time_ns), np.inf, dtype=np.float32)
    extras: dict[str, np.ndarray] = {}
    for name, (dtype, fill_value) in (extra_specs or {}).items():
        extras[name] = np.full(len(time_ns), fill_value, dtype=dtype)
    occupied: dict[int, int] = {}
    for location, sample in locations:
        distance = abs(sample.log_time_ns - int(time_ns[location]))
        if location in occupied and occupied[location] <= distance:
            continue
        occupied[location] = distance
        value[location] = sample.value
        valid[location] = sample.valid
        source_time[location] = sample.sensor_time_ns
        source_log_time[location] = sample.log_time_ns
        source_frame[location] = -1 if sample.frame_index is None else sample.frame_index
        source_message[location] = sample.source_index
        freshness[location] = distance / 1.0e6
        for name in extras:
            if name in sample.metadata:
                extras[name][location] = sample.metadata[name]
    return AlignedStream(
        value=value,
        valid=valid,
        source_time_ns=source_time,
        source_log_time_ns=source_log_time,
        source_frame_index=source_frame,
        source_message_index=source_message,
        freshness_ms=freshness,
        extras=extras,
    )


def _tactile_quality(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    quality = np.zeros((len(value), 5), dtype=np.float32)
    quality[:, 0] = valid.astype(np.float32)
    quality[:, 1] = np.mean(np.isfinite(value), axis=1)
    quality[:, 2] = np.mean(value == 0, axis=1)
    if len(value) > 1:
        delta = value[1:] - value[:-1]
        pair_valid = valid[1:] & valid[:-1]
        quality[1:, 3] = np.mean(np.abs(delta) <= 1.0e-6, axis=1) * pair_valid
        quality[1:, 4] = np.mean(np.abs(delta) > 1000.0, axis=1) * pair_valid
    quality[~valid] = 0.0
    return quality


def correct_and_map_tactile(
    left: AlignedStream,
    right: AlignedStream,
    *,
    swap_left_right: bool,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Swap full 460-D streams and metadata, then select logical-hand mappings."""

    streams = [left, right]
    raw = np.stack([left.value, right.value], axis=1)
    valid = np.stack([left.valid, right.valid], axis=1)
    quality = np.stack(
        [_tactile_quality(left.value, left.valid), _tactile_quality(right.value, right.valid)],
        axis=1,
    )
    source_time = np.stack([left.source_time_ns, right.source_time_ns], axis=1)
    source_log_time = np.stack([left.source_log_time_ns, right.source_log_time_ns], axis=1)
    source_frame = np.stack([left.source_frame_index, right.source_frame_index], axis=1)
    source_message = np.stack([left.source_message_index, right.source_message_index], axis=1)
    freshness = np.stack([left.freshness_ms, right.freshness_ms], axis=1)
    field_code = np.stack(
        [stream.extras["field_code"] for stream in streams], axis=1
    )
    topic_code = np.stack(
        [stream.extras["topic_code"] for stream in streams], axis=1
    )
    hand_match = np.stack(
        [stream.extras["payload_hand_match"] for stream in streams], axis=1
    )
    payload_hand_code = np.stack(
        [stream.extras["payload_hand_code"] for stream in streams], axis=1
    )
    device_code = np.stack(
        [stream.extras["device_code"] for stream in streams], axis=1
    )
    protocol_code = np.stack(
        [stream.extras["protocol_code"] for stream in streams], axis=1
    )
    payload_quality = np.stack(
        [stream.extras["payload_quality"] for stream in streams], axis=1
    )
    if swap_left_right:
        order = np.asarray([1, 0])
        raw = raw[:, order]
        valid = valid[:, order]
        quality = quality[:, order]
        source_time = source_time[:, order]
        source_log_time = source_log_time[:, order]
        source_frame = source_frame[:, order]
        source_message = source_message[:, order]
        freshness = freshness[:, order]
        field_code = field_code[:, order]
        topic_code = topic_code[:, order]
        hand_match = hand_match[:, order]
        payload_hand_code = payload_hand_code[:, order]
        device_code = device_code[:, order]
        protocol_code = protocol_code[:, order]
        payload_quality = payload_quality[:, order]
    # This line is intentionally after the complete-stream swap above.
    tactile360 = np.stack(
        [raw[:, 0, left_indices], raw[:, 1, right_indices]], axis=1
    ).astype(np.float32)
    return {
        "value": tactile360,
        "valid": valid,
        "quality_raw460": quality,
        "source_time_ns": source_time,
        "source_log_time_ns": source_log_time,
        "source_frame_index": source_frame,
        "source_message_index": source_message,
        "freshness_ms": freshness,
        "source_field_code": field_code,
        "source_topic_code": topic_code,
        "payload_hand_match": hand_match,
        "source_payload_hand_code": payload_hand_code,
        "source_device_code": device_code,
        "source_protocol_code": protocol_code,
        "source_payload_quality": payload_quality,
    }


def load_ontology(
    path: str | Path, expected_sha256: str = EXPECTED_ONTOLOGY_SHA256
) -> tuple[dict[int, dict[str, Any]], str]:
    observed_sha256 = sha256_file(path)
    if expected_sha256 and observed_sha256 != expected_sha256:
        raise ValueError(
            f"ontology SHA256 mismatch: {observed_sha256} != {expected_sha256}"
        )
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or document.get("ontology_version") != "atomic_v0.9":
        raise ValueError("unexpected formal500h ontology document")
    rows = document.get("actions")
    if not isinstance(rows, list) or len(rows) != 26:
        raise ValueError("formal500h ontology must contain exactly 26 actions")
    actions = {int(row["class_index"]): dict(row) for row in rows}
    if sorted(actions) != list(range(26)):
        raise ValueError("formal500h ontology class indices must be contiguous [0,26)")
    return actions, observed_sha256


def load_annotation(
    row: Mapping[str, Any],
    ontology_actions: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(str(row["atomic_json_path"]))
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "atomic_annotation_schema_v1":
        raise ValueError(f"unexpected annotation schema: {path}")
    if document.get("ontology_version") != "atomic_v0.9":
        raise ValueError(f"unexpected annotation ontology: {path}")
    if float(document.get("fps")) != 60.0:
        raise ValueError(f"annotation fps is not 60: {path}")
    if document.get("interval_convention") != "[start_frame, end_frame_exclusive)":
        raise ValueError(f"annotation interval convention changed: {path}")
    if document.get("episode_id") != row.get("episode_id"):
        raise ValueError(f"annotation episode_id differs from authoritative manifest: {path}")
    qa = document.get("qa")
    if not isinstance(qa, Mapping):
        raise ValueError(f"annotation lacks qa object: {path}")
    ticks = int(qa.get("annotation_frame_count_60"))
    if ticks != int(row["annotation_frame_count"]):
        raise ValueError(f"annotation tick count differs from authoritative manifest: {path}")
    if qa.get("coverage_complete") is not True or any(
        int(qa.get(name, -1)) != 0
        for name in ("overlap_count", "gap_count", "invalid_segment_count", "review_required_count")
    ):
        raise ValueError(f"annotation QA is not clean: {path}")

    action = np.full(ticks, -1, dtype=np.int16)
    hand = np.full(ticks, -1, dtype=np.int8)
    instance = np.full(ticks, -1, dtype=np.int32)
    boundary_instance = np.zeros(ticks, dtype=np.bool_)
    boundary_class_change = np.zeros(ticks, dtype=np.bool_)
    previous_end = 0
    previous_class: int | None = None
    segments = document.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"annotation has no atomic segments: {path}")
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ValueError(f"atomic segment {index} is not an object: {path}")
        start = int(segment["start_frame_60"])
        end = int(segment["end_frame_60_exclusive"])
        class_index = int(segment["class_index"])
        if start != previous_end or not start < end <= ticks:
            raise ValueError(f"atomic labels are not dense at segment {index}: {path}")
        if not 0 <= class_index < 26:
            raise ValueError(f"atomic class index outside [0,26): {path}")
        if ontology_actions is not None:
            expected_action = ontology_actions[class_index]
            changed = [
                name
                for name in ("label_id", "action_name", "group_id")
                if segment.get(name) != expected_action.get(name)
            ]
            if changed:
                raise ValueError(
                    f"atomic segment {index} disagrees with ontology fields {changed}: {path}"
                )
        if segment.get("review_required") is not False:
            raise ValueError(f"review-required segment reached exporter: {path}")
        hand_name = segment.get("hand")
        if hand_name not in HAND_TARGET:
            raise ValueError(f"unknown hand target {hand_name!r}: {path}")
        action[start:end] = class_index
        hand[start:end] = HAND_TARGET[hand_name]
        instance[start:end] = index
        if index:
            boundary_instance[start] = True
            boundary_class_change[start] = class_index != previous_class
        previous_end = end
        previous_class = class_index
    if previous_end != ticks or np.any(action < 0):
        raise ValueError(f"atomic labels do not cover [0,T): {path}")

    subtasks = document.get("subtasks")
    if not isinstance(subtasks, list):
        raise ValueError(f"subtasks is not a list: {path}")
    subtask_instance = np.full(ticks, -1, dtype=np.int32)
    for index, segment in enumerate(subtasks):
        start = int(segment["start_frame_60"])
        end = int(segment["end_frame_60_exclusive"])
        if not 0 <= start < end <= ticks:
            raise ValueError(f"invalid subtask interval {index}: {path}")
        if np.any(subtask_instance[start:end] >= 0):
            raise ValueError(f"overlapping subtask interval {index}: {path}")
        subtask_instance[start:end] = index
    arrays = {
        "action_class": action,
        "action_hand": hand,
        "action_instance": instance,
        "boundary_instance": boundary_instance,
        "boundary_class_change": boundary_class_change,
        "subtask_instance": subtask_instance,
    }
    return document, arrays


class DinoV3CLSMeanPatch:
    """Frozen local Transformers DINOv3, returning CLS || mean(non-register patches)."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str,
        *,
        precision: str = "bf16",
        expected_weights_sha256: str = EXPECTED_DINOV3_SHA256,
        image_processor_use_fast: bool = True,
        channels_last: bool = False,
    ):
        import torch
        import transformers
        from PIL import __version__ as pillow_version
        from PIL import features as pillow_features
        from transformers import AutoImageProcessor, AutoModel

        self.model_dir = Path(model_dir)
        self.weights_path = self.model_dir / "model.safetensors"
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {self.weights_path}")
        self.weights_sha256 = sha256_file(self.weights_path)
        if expected_weights_sha256 and self.weights_sha256 != expected_weights_sha256:
            raise ValueError(
                f"DINOv3 weights SHA256 mismatch: {self.weights_sha256} != "
                f"{expected_weights_sha256}"
            )
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_dir,
            local_files_only=True,
            use_fast=image_processor_use_fast,
        )
        self.model = AutoModel.from_pretrained(
            self.model_dir, local_files_only=True
        ).eval()
        config = self.model.config
        if config.model_type != "dinov3_vit" or int(config.hidden_size) != 768:
            raise ValueError("local checkpoint is not the frozen DINOv3 ViT-B/16")
        if int(config.patch_size) != 16:
            raise ValueError("DINOv3 patch size must be 16")
        self.register_tokens = int(getattr(config, "num_register_tokens", 4))
        if self.register_tokens != 4:
            raise ValueError(f"expected four DINOv3 register tokens, got {self.register_tokens}")
        self.device = torch.device(device)
        self.precision = precision
        self.image_processor_use_fast = bool(image_processor_use_fast)
        self.channels_last = bool(channels_last)
        self.model.to(self.device)
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)
        self.output_dim = 1536
        self.lineage = {
            "weights_sha256": self.weights_sha256,
            "model_config_sha256": sha256_file(self.model_dir / "config.json"),
            "preprocessor_config_sha256": sha256_file(
                self.model_dir / "preprocessor_config.json"
            ),
            "processor_class": type(self.processor).__name__,
            "image_processor_use_fast_requested": self.image_processor_use_fast,
            "transformers_version": str(transformers.__version__),
            "torch_version": str(torch.__version__),
            "pillow_version": str(pillow_version),
            "libjpeg_version": str(pillow_features.version_codec("jpg") or "unknown"),
        }
        # Processor objects are cheap relative to the model but constructing
        # them once per episode would accumulate over ~57k episodes. Grow this
        # rank-local pool only when an optimization candidate requests more
        # parallel workers, then reuse it for every subsequent episode.
        self._worker_image_processors: list[Any] = [self.processor]
        torch.backends.cuda.matmul.allow_tf32 = True

    def new_image_processor(self) -> Any:
        """Build a worker-local processor with the frozen local configuration."""

        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(
            self.model_dir,
            local_files_only=True,
            use_fast=self.image_processor_use_fast,
        )
        if type(processor) is not type(self.processor):
            raise RuntimeError(
                "worker-local image processor class differs from the frozen "
                f"reference: {type(processor).__name__} != "
                f"{type(self.processor).__name__}"
            )
        return processor

    def image_processor_pool(self, count: int) -> tuple[Any, ...]:
        if count < 1:
            raise ValueError("image processor pool count must be positive")
        while len(self._worker_image_processors) < count:
            self._worker_image_processors.append(self.new_image_processor())
        return tuple(self._worker_image_processors[:count])

    def preprocess_images(
        self,
        images: Sequence[Any],
        *,
        processor: Any | None = None,
        pin_memory: bool = False,
    ) -> Any:
        import torch

        selected_processor = self.processor if processor is None else processor
        inputs = selected_processor(images=list(images), return_tensors="pt")
        pixels = inputs["pixel_values"]
        if self.channels_last:
            pixels = pixels.contiguous(memory_format=torch.channels_last)
        if pin_memory and self.device.type == "cuda" and not pixels.is_pinned():
            pixels = pixels.pin_memory()
        return pixels

    @contextlib.contextmanager
    def inference_session(self) -> Iterable[None]:
        import torch

        autocast: Any = contextlib.nullcontext()
        if self.device.type == "cuda" and self.precision != "fp32":
            dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
            autocast = torch.autocast("cuda", dtype=dtype)
        with torch.inference_mode(), autocast:
            yield

    def encode_preprocessed(
        self,
        pixels: Any,
        *,
        within_inference_session: bool = False,
    ) -> np.ndarray:
        import torch

        pixels = pixels.to(self.device, non_blocking=bool(pixels.is_pinned()))

        def forward() -> np.ndarray:
            tokens = self.model(pixel_values=pixels).last_hidden_state
            patch_start = 1 + self.register_tokens
            if (
                tokens.ndim != 3
                or tokens.shape[-1] != 768
                or tokens.shape[1] <= patch_start
            ):
                raise ValueError(
                    f"unexpected DINOv3 token shape: {tuple(tokens.shape)}"
                )
            cls = tokens[:, 0]
            mean_patch = tokens[:, patch_start:].mean(dim=1)
            # Convert on-device so D2H transfers the final fp16 cache payload,
            # rather than an unnecessary float32 intermediate twice its size.
            feature = torch.cat([cls, mean_patch], dim=-1).to(torch.float16)
            return feature.cpu().numpy()

        if within_inference_session:
            return forward()
        with self.inference_session():
            return forward()

    def __call__(self, images: Sequence[Any]) -> np.ndarray:
        return self.encode_preprocessed(self.preprocess_images(images))


@dataclass(frozen=True)
class _VisionInput:
    payload: bytes
    topic: str
    frame_index_60: int
    log_time_ns: int
    slot: int
    view: int
    acquisition_ns: int


@dataclass(frozen=True)
class _PreparedVisionBatch:
    pixels: Any
    items: tuple[_VisionInput, ...]


_VISION_PIPELINE_END = object()


class VisionPipelineFatalError(RuntimeError):
    """A worker lifecycle failure after which the rank must not continue."""


class _VisionPipeline:
    """Bounded CPU-decode/preprocess -> GPU inference pipeline for one episode.

    The MCAP reader remains on the caller thread.  A CPU worker decodes each
    JPEG batch in a small thread pool and runs the frozen Hugging Face image
    processor.  A separate GPU worker consumes prepared tensors.  The bounded
    queues overlap these stages without allowing an episode to accumulate an
    unbounded number of decoded frames in RAM.
    """

    def __init__(
        self,
        *,
        backbone: DinoV3CLSMeanPatch,
        vision: np.ndarray,
        vision_valid: np.ndarray,
        acquisition: np.ndarray,
        decode_errors: list[dict[str, Any]],
        batch_size: int,
        jpeg_decode_workers: int,
        prefetch_batches: int,
        pin_memory: bool,
        preprocess_workers: int = 1,
    ) -> None:
        if jpeg_decode_workers < 1:
            raise ValueError("jpeg_decode_workers must be at least one")
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        if prefetch_batches < 1:
            raise ValueError("vision_prefetch_batches must be at least one")
        if preprocess_workers < 1:
            raise ValueError("vision_preprocess_workers_per_rank must be at least one")
        self.backbone = backbone
        self.vision = vision
        self.vision_valid = vision_valid
        self.acquisition = acquisition
        self.decode_errors = decode_errors
        self.batch_size = int(batch_size)
        self.jpeg_decode_workers = int(jpeg_decode_workers)
        self.preprocess_workers = int(preprocess_workers)
        self.pin_memory = bool(pin_memory)
        self.raw_queue: queue.Queue[Any] = queue.Queue(maxsize=prefetch_batches)
        self.prepared_queue: queue.Queue[Any] = queue.Queue(maxsize=prefetch_batches)
        self.stop_event = threading.Event()
        self.error_lock = threading.Lock()
        self.error: BaseException | None = None
        self.timings: dict[str, float] = {
            "vision_batches": 0.0,
            "vision_images": 0.0,
            "vision_decoded_images": 0.0,
            "vision_decode_error_images": 0.0,
            "jpeg_decode_wall_s": 0.0,
            "vision_preprocess_wall_s": 0.0,
            "vision_preprocess_critical_span_s": 0.0,
            "vision_preprocess_future_wait_s": 0.0,
            "vision_gpu_and_d2h_wall_s": 0.0,
            "vision_submit_blocked_s": 0.0,
            "vision_prepared_queue_blocked_s": 0.0,
        }
        self.preprocess_thread = threading.Thread(
            target=self._preprocess_worker,
            name="formal500h-vision-preprocess",
            daemon=False,
        )
        self.gpu_thread = threading.Thread(
            target=self._gpu_worker,
            name="formal500h-vision-gpu",
            daemon=False,
        )
        self.preprocess_thread.start()
        self.gpu_thread.start()

    def _record_error(self, error: BaseException) -> None:
        with self.error_lock:
            if self.error is None:
                self.error = error
        self.stop_event.set()

    def _raise_if_failed(self) -> None:
        if self.error is not None:
            raise VisionPipelineFatalError("vision pipeline worker failed") from self.error
        if self.stop_event.is_set():
            raise VisionPipelineFatalError("vision pipeline was aborted")

    def _put(self, target: queue.Queue[Any], value: Any) -> float:
        started = time.perf_counter()
        while True:
            self._raise_if_failed()
            try:
                target.put(value, timeout=0.1)
                return time.perf_counter() - started
            except queue.Full:
                continue

    def _get(self, source: queue.Queue[Any]) -> Any:
        while True:
            if self.stop_event.is_set():
                self._raise_if_failed()
            try:
                return source.get(timeout=0.1)
            except queue.Empty:
                continue

    @staticmethod
    def _decode_one(item: _VisionInput) -> tuple[Any | None, dict[str, Any] | None]:
        from PIL import Image

        try:
            with Image.open(io.BytesIO(item.payload)) as image:
                image.load()
                rgb = image.convert("RGB")
            return rgb, None
        except (OSError, SyntaxError) as error:
            return None, {
                "topic": item.topic,
                "frame_index_60": item.frame_index_60,
                "log_time_ns": item.log_time_ns,
                "error": f"{type(error).__name__}: {error}",
            }

    def _preprocess_worker(self) -> None:
        try:
            processor_tls = threading.local()
            available_processors: queue.SimpleQueue[Any] = queue.SimpleQueue()
            for processor in self.backbone.image_processor_pool(
                self.preprocess_workers
            ):
                available_processors.put(processor)
            inflight: deque[Any] = deque()
            preprocess_span_started: float | None = None

            def preprocess_complete_batch(
                images: tuple[Any, ...],
            ) -> tuple[Any, float]:
                processor = getattr(processor_tls, "processor", None)
                if processor is None:
                    processor = available_processors.get()
                    processor_tls.processor = processor
                started = time.perf_counter()
                try:
                    pixels = self.backbone.preprocess_images(
                        images,
                        processor=processor,
                        pin_memory=self.pin_memory,
                    )
                    return pixels, time.perf_counter() - started
                finally:
                    for image in images:
                        image.close()

            with ThreadPoolExecutor(
                max_workers=self.jpeg_decode_workers,
                thread_name_prefix="formal500h-jpeg",
            ) as decode_executor, ThreadPoolExecutor(
                max_workers=self.preprocess_workers,
                thread_name_prefix="formal500h-processor",
            ) as processor_executor:
                pending_images: list[Any] = []
                pending_items: list[_VisionInput] = []

                def retire_oldest() -> None:
                    future, items = inflight.popleft()
                    wait_started = time.perf_counter()
                    pixels, service_seconds = future.result()
                    self.timings["vision_preprocess_future_wait_s"] += (
                        time.perf_counter() - wait_started
                    )
                    self.timings["vision_preprocess_wall_s"] += service_seconds
                    blocked = self._put(
                        self.prepared_queue,
                        _PreparedVisionBatch(pixels=pixels, items=items),
                    )
                    self.timings["vision_prepared_queue_blocked_s"] += blocked

                def submit_preprocess(count: int) -> None:
                    nonlocal preprocess_span_started
                    images = tuple(pending_images[:count])
                    items = tuple(pending_items[:count])
                    del pending_images[:count]
                    del pending_items[:count]
                    if preprocess_span_started is None:
                        preprocess_span_started = time.perf_counter()
                    inflight.append(
                        (
                            processor_executor.submit(
                                preprocess_complete_batch,
                                images,
                            ),
                            items,
                        )
                    )
                    # Bound both full-resolution RGB images and processor
                    # temporaries. Futures are deliberately retired in input
                    # order so the GPU sees exactly the original batch order.
                    if len(inflight) >= self.preprocess_workers:
                        retire_oldest()

                while True:
                    batch = self._get(self.raw_queue)
                    if batch is _VISION_PIPELINE_END:
                        break
                    items = tuple(batch)
                    started = time.perf_counter()
                    decoded = list(decode_executor.map(self._decode_one, items))
                    self.timings["jpeg_decode_wall_s"] += time.perf_counter() - started
                    images: list[Any] = []
                    valid_items: list[_VisionInput] = []
                    for item, (image, error) in zip(items, decoded):
                        if error is not None:
                            self.decode_errors.append(error)
                        elif image is not None:
                            images.append(image)
                            valid_items.append(item)
                    self.timings["vision_decoded_images"] += float(len(images))
                    self.timings["vision_decode_error_images"] += float(
                        len(items) - len(images)
                    )
                    pending_images.extend(images)
                    pending_items.extend(valid_items)
                    while len(pending_images) >= self.batch_size:
                        submit_preprocess(self.batch_size)
                if pending_images:
                    submit_preprocess(len(pending_images))
                while inflight:
                    retire_oldest()
                if preprocess_span_started is not None:
                    self.timings["vision_preprocess_critical_span_s"] += (
                        time.perf_counter() - preprocess_span_started
                    )
            self._put(self.prepared_queue, _VISION_PIPELINE_END)
        except BaseException as error:  # propagate worker failures to caller
            self._record_error(error)

    def _gpu_worker(self) -> None:
        try:
            if self.backbone.device.type == "cuda":
                import torch

                torch.cuda.set_device(self.backbone.device)
            # Keep a rank-long autocast context so PyTorch can retain its
            # autocast weight cache across batches.
            with self.backbone.inference_session():
                while True:
                    prepared = self._get(self.prepared_queue)
                    if prepared is _VISION_PIPELINE_END:
                        break
                    started = time.perf_counter()
                    features = self.backbone.encode_preprocessed(
                        prepared.pixels,
                        within_inference_session=True,
                    )
                    self.timings["vision_gpu_and_d2h_wall_s"] += (
                        time.perf_counter() - started
                    )
                    if features.shape != (len(prepared.items), 1536):
                        raise ValueError(f"DINOv3 output shape is {features.shape}")
                    slots = np.fromiter(
                        (item.slot for item in prepared.items),
                        dtype=np.int64,
                        count=len(prepared.items),
                    )
                    views = np.fromiter(
                        (item.view for item in prepared.items),
                        dtype=np.int64,
                        count=len(prepared.items),
                    )
                    acquisition_ns = np.fromiter(
                        (item.acquisition_ns for item in prepared.items),
                        dtype=np.int64,
                        count=len(prepared.items),
                    )
                    self.vision[slots, views] = features
                    self.vision_valid[slots, views] = True
                    self.acquisition[slots, views] = acquisition_ns
        except BaseException as error:  # propagate worker failures to caller
            self._record_error(error)

    def submit(self, items: Sequence[_VisionInput]) -> None:
        if not items:
            return
        blocked = self._put(self.raw_queue, tuple(items))
        self.timings["vision_submit_blocked_s"] += blocked
        self.timings["vision_batches"] += 1.0
        self.timings["vision_images"] += float(len(items))

    def close(self) -> dict[str, float]:
        self._put(self.raw_queue, _VISION_PIPELINE_END)
        self.preprocess_thread.join()
        self._raise_if_failed()
        self.gpu_thread.join()
        self._raise_if_failed()
        return dict(self.timings)

    def abort(self) -> None:
        self.stop_event.set()
        for target in (self.raw_queue, self.prepared_queue):
            try:
                target.put_nowait(_VISION_PIPELINE_END)
            except queue.Full:
                pass
        self.preprocess_thread.join(timeout=30.0)
        self.gpu_thread.join(timeout=30.0)
        alive = [
            thread.name
            for thread in (self.preprocess_thread, self.gpu_thread)
            if thread.is_alive()
        ]
        if alive:
            raise VisionPipelineFatalError(
                f"vision pipeline threads did not terminate after abort: {alive}"
            )


@dataclass
class EpisodeRead:
    vision_features: np.ndarray
    vision_valid: np.ndarray
    video_source_time_ns_60: np.ndarray
    vision_source_acquisition_time_ns: np.ndarray
    tactile: dict[int, list[Sample]]
    skeleton: dict[int, list[Sample]]
    pose: dict[int, list[Sample]]
    time_samples: list[Sample]
    skeleton_bone_names: tuple[tuple[str, ...], tuple[str, ...]]
    topic_counts: dict[str, int]
    decode_errors: list[dict[str, Any]]
    performance: dict[str, float] = field(default_factory=dict)


def _acquisition_ns(decoded: Any) -> int:
    timestamp = getattr(decoded, "timestamp", None)
    if timestamp is None:
        return -1
    return int(getattr(timestamp, "seconds", 0)) * 1_000_000_000 + int(
        getattr(timestamp, "nanos", 0)
    )


def validate_episode_topic_counts(
    topic_counts: Mapping[str, int],
    *,
    ticks: int,
    max_endpoint_delta: int,
) -> dict[str, int]:
    """Validate recorder endpoint counts without changing the annotation axis.

    The annotation movie was produced at 30 Hz and its frame count was doubled
    to recover the formal 60 Hz half-open label interval.  Depending on recorder
    endpoint inclusion, a source stream can consequently contain T-1, T, or T+1
    messages.  Only samples whose global frame lies in [0, T) are exported.
    """

    if ticks <= 0:
        raise ValueError("annotation tick count must be positive")
    if max_endpoint_delta < 0:
        raise ValueError("max_endpoint_delta must be non-negative")
    counts = {topic: int(topic_counts.get(topic, 0)) for topic in INPUT_TOPICS}
    view_counts = [counts[topic] for topic in VIDEO_TOPICS]
    if len(set(view_counts)) != 1:
        raise ValueError(f"three 60 Hz video streams have unequal counts: {view_counts}")
    outside = {
        topic: count
        for topic, count in counts.items()
        if abs(count - ticks) > max_endpoint_delta
    }
    if outside:
        raise ValueError(
            "formal 60 Hz topic counts exceed annotation endpoint tolerance "
            f"T={ticks}, allowed_delta={max_endpoint_delta}: {outside}"
        )
    expected_visual_frames = len(range(0, ticks, 2))
    available_visual_frames = len(range(0, min(view_counts[0], ticks), 2))
    if available_visual_frames != expected_visual_frames:
        raise ValueError(
            "video endpoint count does not contain every required 30 Hz sample: "
            f"count={view_counts[0]}, T={ticks}, available={available_visual_frames}, "
            f"required={expected_visual_frames}"
        )
    return {topic: count - ticks for topic, count in counts.items()}


def read_episode_once(
    mcap_path: str | Path,
    *,
    ticks: int,
    backbone: DinoV3CLSMeanPatch,
    batch_size: int,
    skeleton_joints: int,
    schema_contract: McapSchemaContract,
    legacy_tactile_schema_names: frozenset[str],
    max_topic_endpoint_delta: int,
    jpeg_decode_workers: int = 4,
    vision_prefetch_batches: int = 2,
    vision_pin_memory: bool = True,
    vision_preprocess_workers: int = 1,
) -> EpisodeRead:
    """Scan one MCAP once, extracting video and collecting all model sensors."""

    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
        from PIL import Image
    except ImportError as error:  # pragma: no cover - production dependency
        raise RuntimeError(
            "formal500h export requires mcap, mcap-protobuf-support, and Pillow"
        ) from error

    source = Path(mcap_path)
    visual_ticks = np.arange(0, ticks, 2, dtype=np.int64)
    vision = np.zeros((len(visual_ticks), 3, 1536), dtype=np.float16)
    vision_valid = np.zeros((len(visual_ticks), 3), dtype=np.bool_)
    video_time_60 = np.full((ticks, 3), -1, dtype=np.int64)
    acquisition = np.full((len(visual_ticks), 3), -1, dtype=np.int64)
    view_counts = np.zeros(3, dtype=np.int64)
    queued_vision: list[_VisionInput] = []
    tactile = {0: [], 1: []}
    skeleton = {0: [], 1: []}
    pose = {0: [], 1: [], 2: []}
    time_samples: list[Sample] = []
    bone_names_by_hand: list[tuple[str, ...] | None] = [None, None]
    topic_counts = {topic: 0 for topic in INPUT_TOPICS}
    decode_errors: list[dict[str, Any]] = []
    decoder_factory = DecoderFactory()
    decoder_by_channel: dict[int, Any] = {}
    validated_channels: set[int] = set()

    pipeline = _VisionPipeline(
        backbone=backbone,
        vision=vision,
        vision_valid=vision_valid,
        acquisition=acquisition,
        decode_errors=decode_errors,
        batch_size=batch_size,
        jpeg_decode_workers=jpeg_decode_workers,
        prefetch_batches=vision_prefetch_batches,
        pin_memory=vision_pin_memory,
        preprocess_workers=vision_preprocess_workers,
    )

    def flush() -> None:
        if queued_vision:
            pipeline.submit(tuple(queued_vision))
            queued_vision.clear()

    scan_started = time.perf_counter()
    try:
        with source.open("rb") as handle:
            reader = make_reader(handle)
            message_index = 0
            for schema, channel, message in reader.iter_messages(topics=INPUT_TOPICS):
                topic = channel.topic
                topic_counts[topic] += 1
                if channel.id not in validated_channels:
                    assert_formal_channel(schema_contract, schema, channel)
                    validated_channels.add(channel.id)
                if topic in VIDEO_TOPICS:
                    view = VIEW_INDEX[topic]
                    frame = int(view_counts[view])
                    view_counts[view] += 1
                    if frame < ticks:
                        video_time_60[frame, view] = int(message.log_time)
                    if frame >= ticks or frame % 2:
                        message_index += 1
                        continue
                    decoder = decoder_by_channel.get(channel.id)
                    if decoder is None:
                        decoder = decoder_factory.decoder_for(
                            channel.message_encoding, schema
                        )
                        if decoder is None:
                            raise ValueError(
                                f"no protobuf decoder for {topic} "
                                f"encoding={channel.message_encoding}"
                            )
                        decoder_by_channel[channel.id] = decoder
                    decoded = decoder(message.data)
                    image_format = str(getattr(decoded, "format", "")).lower()
                    if image_format not in {"jpeg", "jpg"}:
                        raise ValueError(
                            f"{topic} format must be JPEG, got {image_format!r}"
                        )
                    queued_vision.append(
                        _VisionInput(
                            payload=bytes(getattr(decoded, "data")),
                            topic=topic,
                            frame_index_60=frame,
                            log_time_ns=int(message.log_time),
                            slot=frame // 2,
                            view=view,
                            acquisition_ns=_acquisition_ns(decoded),
                        )
                    )
                    if len(queued_vision) >= batch_size:
                        flush()
                else:
                    if channel.message_encoding != "json":
                        raise ValueError(
                            f"sensor topic {topic} encoding is "
                            f"{channel.message_encoding!r}, expected json"
                        )
                    payload = json.loads(message.data)
                    if not isinstance(payload, Mapping):
                        raise ValueError(f"{topic} JSON payload is not an object")
                    if topic in TACTILE_TOPICS:
                        hand = HAND_INDEX[topic.rsplit("/", 1)[-1]]
                        tactile[hand].append(
                            _parse_tactile(
                                payload,
                                topic,
                                int(message.log_time),
                                message_index,
                                schema_name=(
                                    str(schema.name) if schema is not None else ""
                                ),
                                legacy_schema_names=legacy_tactile_schema_names,
                            )
                        )
                    elif topic == SKELETON_TOPIC:
                        for hand_name, hand_index in HAND_INDEX.items():
                            sample = _parse_formal_skeleton_hand(
                                payload,
                                hand_name,
                                int(message.log_time),
                                message_index,
                                skeleton_joints,
                            )
                            names = tuple(sample.metadata["bone_names"])
                            if bone_names_by_hand[hand_index] is None:
                                bone_names_by_hand[hand_index] = names
                            elif bone_names_by_hand[hand_index] != names:
                                raise ValueError(
                                    f"{SKELETON_TOPIC}.{hand_name} bone order "
                                    "changed within MCAP"
                                )
                            skeleton[hand_index].append(sample)
                    elif topic == POSE_TOPIC:
                        for pose_index, block_name in enumerate(POSE_BLOCKS):
                            pose[pose_index].append(
                                _parse_formal_pose_block(
                                    payload,
                                    block_name,
                                    int(message.log_time),
                                    message_index,
                                )
                            )
                    elif topic == TIME_TOPIC:
                        time_samples.append(
                            _parse_formal_time_sync(
                                payload, int(message.log_time), message_index
                            )
                        )
                message_index += 1
        flush()
        scan_wall_s = time.perf_counter() - scan_started
        performance = pipeline.close()
    except BaseException:
        pipeline.abort()
        raise
    performance["mcap_scan_and_submit_wall_s"] = scan_wall_s
    performance["episode_read_pipeline_wall_s"] = time.perf_counter() - scan_started
    validate_episode_topic_counts(
        topic_counts,
        ticks=ticks,
        max_endpoint_delta=max_topic_endpoint_delta,
    )
    if any(names is None or len(names) != skeleton_joints for names in bone_names_by_hand):
        raise ValueError("formal skeleton bone-name order was not established")
    return EpisodeRead(
        vision_features=vision,
        vision_valid=vision_valid,
        video_source_time_ns_60=video_time_60,
        vision_source_acquisition_time_ns=acquisition,
        tactile=tactile,
        skeleton=skeleton,
        pose=pose,
        time_samples=time_samples,
        skeleton_bone_names=(
            tuple(bone_names_by_hand[0] or ()),
            tuple(bone_names_by_hand[1] or ()),
        ),
        topic_counts=topic_counts,
        decode_errors=decode_errors,
        performance=performance,
    )


def canonical_time_axis(video_time_60: np.ndarray) -> np.ndarray:
    if video_time_60.ndim != 2 or video_time_60.shape[1] != 3:
        raise ValueError("video source times must have shape [T,3]")
    first = video_time_60[0]
    first = first[first >= 0]
    if not len(first):
        raise ValueError("no first-frame video timestamp is available")
    start = int(np.median(first))
    return start + np.rint(
        np.arange(len(video_time_60), dtype=np.float64) * (1.0e9 / 60.0)
    ).astype(np.int64)


def _nearest_visual_alignment(
    time_60: np.ndarray, vision_time: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.full((len(time_60), 3), -1, dtype=np.int32)
    aligned_valid = np.zeros((len(time_60), 3), dtype=np.bool_)
    freshness = np.full((len(time_60), 3), np.inf, dtype=np.float32)
    for view in range(3):
        available = np.flatnonzero(valid[:, view])
        if not len(available):
            continue
        times = vision_time[available]
        insertion = np.searchsorted(times, time_60, side="left")
        right = np.clip(insertion, 0, len(times) - 1)
        left = np.clip(insertion - 1, 0, len(times) - 1)
        use_right = np.abs(times[right] - time_60) < np.abs(times[left] - time_60)
        chosen = available[np.where(use_right, right, left)]
        indices[:, view] = chosen.astype(np.int32)
        aligned_valid[:, view] = True
        freshness[:, view] = (
            np.abs(vision_time[chosen] - time_60).astype(np.float64) / 1.0e6
        ).astype(np.float32)
    return indices, aligned_valid, freshness


def assemble_episode(
    row: Mapping[str, Any],
    authority: Authority,
    annotation: Mapping[str, Any],
    targets: Mapping[str, np.ndarray],
    raw: EpisodeRead,
    *,
    min_sensor_valid_fraction: float,
) -> dict[str, Any]:
    ticks = int(row["annotation_frame_count"])
    time_ns = canonical_time_axis(raw.video_source_time_ns_60)
    tactile_extra = {
        "field_code": (np.uint8, 0),
        "topic_code": (np.int8, -1),
        "payload_hand_match": (np.bool_, False),
        "payload_hand_code": (np.int8, 0),
        "device_code": (np.int8, 0),
        "protocol_code": (np.int8, 0),
        "payload_quality": (np.float32, np.nan),
    }
    tactile_streams = [
        align_samples(
            raw.tactile[hand], time_ns, (460,), extra_specs=tactile_extra
        )
        for hand in range(2)
    ]
    tactile = correct_and_map_tactile(
        tactile_streams[0],
        tactile_streams[1],
        swap_left_right=bool(row["tactile_swap_left_right"]),
        left_indices=authority.left_indices,
        right_indices=authority.right_indices,
    )
    if float(tactile["valid"].mean()) < min_sensor_valid_fraction:
        raise ValueError(
            f"tactile valid fraction {tactile['valid'].mean():.6f} is below "
            f"{min_sensor_valid_fraction}"
        )

    skeleton_streams = [
        align_samples(
            raw.skeleton[hand],
            time_ns,
            (25, 7),
        )
        for hand in range(2)
    ]
    skeleton_world = np.stack([stream.value for stream in skeleton_streams], axis=1)
    skeleton_valid = np.stack([stream.valid for stream in skeleton_streams], axis=1)
    joint_valid = np.zeros((ticks, 2, 25), dtype=np.bool_)
    # Joint validity follows the exact message selected by align_samples.
    for hand in range(2):
        by_source_index = {
            sample.source_index: sample for sample in raw.skeleton[hand]
        }
        for location, source_index in enumerate(
            skeleton_streams[hand].source_message_index
        ):
            if source_index >= 0:
                joint_valid[location, hand] = by_source_index[int(source_index)].metadata[
                    "joint_valid"
                ]
    skeleton_relative = _relative_skeleton(skeleton_world, joint_valid)

    pose_streams = [
        align_samples(raw.pose[index], time_ns, (7,)) for index in range(3)
    ]
    pose_world = np.stack([stream.value for stream in pose_streams], axis=1)
    pose_valid = np.stack([stream.valid for stream in pose_streams], axis=1)
    pose_position = pose_world[..., :3].copy()
    pose_quaternion = pose_world[..., 3:7].copy()
    kinematics, kinematics_valid, pose_quality = _build_kinematics(
        pose_position,
        pose_quaternion,
        pose_valid,
        skeleton_relative,
        joint_valid,
        time_ns,
        pose_jump_m=0.25,
    )

    vision_time = time_ns[::2]
    vision_indices, vision_aligned_valid, vision_freshness = _nearest_visual_alignment(
        time_ns, vision_time, raw.vision_valid
    )
    time_sync = align_samples(
        raw.time_samples,
        time_ns,
        (1,),
        extra_specs={"master_unix_ns": (np.int64, -1)},
    )
    return {
        "time_ns": time_ns,
        "frame_index_60": np.arange(ticks, dtype=np.int32),
        "time_sync_valid": time_sync.valid,
        "time_sync_source_time_ns": time_sync.source_time_ns,
        "time_sync_master_unix_ns": time_sync.extras["master_unix_ns"],
        "time_sync_freshness_ms": time_sync.freshness_ms,
        "targets": dict(targets),
        "tactile": tactile,
        "skeleton_world": skeleton_world,
        "skeleton_relative": skeleton_relative,
        "skeleton_valid": skeleton_valid,
        "skeleton_joint_valid": joint_valid,
        "skeleton_bone_names": raw.skeleton_bone_names,
        "skeleton_source_time_ns": np.stack(
            [stream.source_time_ns for stream in skeleton_streams], axis=1
        ),
        "skeleton_freshness_ms": np.stack(
            [stream.freshness_ms for stream in skeleton_streams], axis=1
        ),
        "pose_world": pose_world,
        "pose_valid": pose_valid,
        "pose_source_time_ns": np.stack(
            [stream.source_time_ns for stream in pose_streams], axis=1
        ),
        "pose_freshness_ms": np.stack(
            [stream.freshness_ms for stream in pose_streams], axis=1
        ),
        "kinematics": kinematics,
        "kinematics_valid": kinematics_valid,
        "pose_quality": pose_quality,
        "vision_features": raw.vision_features,
        "vision_valid": raw.vision_valid,
        "vision_time_ns": vision_time,
        "vision_frame_index_60": np.arange(0, ticks, 2, dtype=np.int32),
        "vision_source_time_ns": raw.video_source_time_ns_60[::2],
        "vision_source_acquisition_time_ns": raw.vision_source_acquisition_time_ns,
        "vision_index_60": vision_indices,
        "vision_aligned_valid_60": vision_aligned_valid,
        "vision_freshness_ms_60": vision_freshness,
        "video_source_time_ns_60": raw.video_source_time_ns_60,
    }


def _chunks(array: np.ndarray) -> tuple[int, ...] | None:
    if array.ndim == 0 or array.shape[0] == 0:
        return None
    return (min(600, array.shape[0]),) + array.shape[1:]


def _dataset(group: Any, name: str, value: Any, compression: str | None) -> None:
    array = np.asarray(value)
    use_compression = compression if array.ndim and array.size else None
    group.create_dataset(
        name,
        data=array,
        chunks=_chunks(array),
        compression=use_compression,
        shuffle=bool(use_compression and array.dtype.kind in "fiu"),
    )


def _string_dataset(group: Any, name: str, values: Iterable[Any], h5py: Any) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray([str(value or "") for value in values], dtype=object), dtype=dtype)


def cache_path_for_row(output_root: str | Path, row: Mapping[str, Any]) -> Path:
    path_hash = hashlib.sha256(str(row["mcap_path"]).encode("utf-8")).hexdigest()
    capture = str(row["capture_key"])
    return Path(output_root) / "episodes" / path_hash[:2] / f"{capture}.{path_hash[:12]}.h5"


def write_episode_hdf5(
    output_path: str | Path,
    *,
    row: Mapping[str, Any],
    annotation: Mapping[str, Any],
    arrays: Mapping[str, Any],
    authority: Authority,
    schema_contract: McapSchemaContract,
    backbone: DinoV3CLSMeanPatch,
    config_sha256: str,
    ontology_actions: Mapping[int, Mapping[str, Any]],
    ontology_sha256: str,
    compression: str | None,
    overwrite: bool,
    topic_counts: Mapping[str, int],
    decode_errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("formal500h export requires h5py") from error
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    partial = output.with_name(f".{output.name}.rank{os.environ.get('RANK', '0')}.partial")
    partial.unlink(missing_ok=True)
    annotation_hash = sha256_file(str(row["atomic_json_path"]))
    topic_count_delta = {
        topic: int(count) - int(row["annotation_frame_count"])
        for topic, count in topic_counts.items()
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": row["episode_id"],
        "capture_key": row["capture_key"],
        "task_id": row["task_id"],
        "task_name": row["task_name"],
        "source_mcap_path": row["mcap_path"],
        "source_mcap_size_bytes": int(row["mcap_size_bytes"]),
        "source_atomic_json_path": row["atomic_json_path"],
        "source_atomic_json_sha256": annotation_hash,
        "correction_manifest_sha256": authority.correction_sha256,
        "tactile_mapping_sha256": authority.mapping_sha256,
        "formal_mcap_schema_report": schema_contract.report_path,
        "formal_mcap_schema_report_sha256": schema_contract.report_sha256,
        "tactile_hand_correction": row["tactile_hand_correction"],
        "tactile_swap_left_right": bool(row["tactile_swap_left_right"]),
        "tactile_processing_unit": "whole_mcap",
        "tactile_layout": TACTILE_LAYOUT,
        "tactile_shape": "[T60,2,360]",
        "tactile_hand_order": ["left", "right"],
        "tactile_value_semantics": "raw_unscaled",
        "normalization": "none; compute from train split after correction+mapping",
        "annotation_schema": annotation["schema_version"],
        "ontology_version": annotation["ontology_version"],
        "ontology_sha256": ontology_sha256,
        "label_rate_hz": 60.0,
        "vision_rate_hz": 30.0,
        "vision_views": ["head", "left", "right"],
        "vision_backbone": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "vision_representation": "CLS token || mean patch tokens; four register tokens excluded",
        "vision_feature_dim": 1536,
        "vision_weights_sha256": backbone.weights_sha256,
        "vision_lineage": dict(
            getattr(
                backbone,
                "lineage",
                {"weights_sha256": backbone.weights_sha256},
            )
        ),
        "config_sha256": config_sha256,
        "topic_counts": dict(topic_counts),
        "topic_count_delta_from_annotation": topic_count_delta,
        "source_endpoint_policy": (
            "retain global frames in [0,T); accept only the configured per-topic "
            "endpoint count tolerance; never resize the annotation axis"
        ),
        "combined_tracking_timestamp_semantics": (
            "outer frameIndex aligns the 60 Hz aggregate frame; nested timestampNs "
            "is preserved as source_time_ns"
        ),
        "vision_decode_error_count": len(decode_errors),
        "created_unix_s": time.time(),
    }
    with h5py.File(partial, "w", libver="latest") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["complete"] = False
        handle.attrs["metadata_json"] = _canonical_json(metadata)

        timeline = handle.create_group("timeline")
        _dataset(timeline, "time_ns", arrays["time_ns"], compression)
        _dataset(timeline, "frame_index_60", arrays["frame_index_60"], compression)
        _dataset(timeline, "time_sync_valid", arrays["time_sync_valid"], compression)
        _dataset(
            timeline,
            "time_sync_source_time_ns",
            arrays["time_sync_source_time_ns"],
            compression,
        )
        _dataset(
            timeline,
            "time_sync_master_unix_ns",
            arrays["time_sync_master_unix_ns"],
            compression,
        )
        _dataset(
            timeline,
            "time_sync_freshness_ms",
            arrays["time_sync_freshness_ms"],
            compression,
        )

        targets = handle.create_group("targets")
        for name, value in arrays["targets"].items():
            _dataset(targets, name, value, compression)
        ontology = handle.create_group("annotations/ontology")
        _dataset(
            ontology,
            "class_index",
            np.arange(26, dtype=np.int16),
            None,
        )
        for name in ("label_id", "action_name", "display_name_zh", "group_id"):
            _string_dataset(
                ontology,
                name,
                (ontology_actions[index].get(name) for index in range(26)),
                h5py,
            )

        tactile = handle.create_group("sensors/tactile")
        for name, value in arrays["tactile"].items():
            _dataset(tactile, name, value, compression)
        _dataset(tactile, "left_raw_indices", authority.left_indices.astype(np.int16), None)
        _dataset(tactile, "right_raw_indices", authority.right_indices.astype(np.int16), None)
        tactile.attrs["hand_order"] = _canonical_json(["left", "right"])
        tactile.attrs["quality_names"] = _canonical_json(
            ["packet_valid", "finite_fraction", "zero_fraction", "unchanged_fraction", "jump_fraction"]
        )
        tactile.attrs["source_field_codes"] = _canonical_json(TACTILE_FIELD_CODE)
        tactile.attrs["source_topic_codes"] = _canonical_json(TACTILE_TOPIC_CODE)
        tactile.attrs["source_payload_hand_codes"] = _canonical_json(
            {"missing": 0, "left": 1, "right": 2, "other": 3}
        )
        tactile.attrs["source_device_codes"] = _canonical_json(
            {"missing": 0, "left": 1, "right": 2, "other": 3}
        )
        tactile.attrs["source_protocol_codes"] = _canonical_json(
            {"missing": 0, "moxian": 1, "other": 2}
        )

        skeleton = handle.create_group("sensors/skeleton")
        for name in (
            "skeleton_world",
            "skeleton_relative",
            "skeleton_valid",
            "skeleton_joint_valid",
            "skeleton_source_time_ns",
            "skeleton_freshness_ms",
        ):
            _dataset(skeleton, name.removeprefix("skeleton_"), arrays[name], compression)
        _string_dataset(
            skeleton,
            "bone_names_left",
            arrays["skeleton_bone_names"][0],
            h5py,
        )
        _string_dataset(
            skeleton,
            "bone_names_right",
            arrays["skeleton_bone_names"][1],
            h5py,
        )

        pose = handle.create_group("sensors/pose")
        for name in (
            "pose_world",
            "pose_valid",
            "pose_source_time_ns",
            "pose_freshness_ms",
            "pose_quality",
        ):
            _dataset(pose, name.removeprefix("pose_"), arrays[name], compression)
        pose.attrs["stream_order"] = _canonical_json(["head", "left_wrist", "right_wrist"])

        kinematics = handle.create_group("sensors/kinematics")
        _dataset(kinematics, "value", arrays["kinematics"], compression)
        _dataset(kinematics, "valid", arrays["kinematics_valid"], compression)
        kinematics.attrs["semantics"] = "offline pose/skeleton-derived; not measured IMU"

        vision = handle.create_group("vision")
        for name in (
            "vision_features",
            "vision_valid",
            "vision_time_ns",
            "vision_frame_index_60",
            "vision_source_time_ns",
            "vision_source_acquisition_time_ns",
            "vision_index_60",
            "vision_aligned_valid_60",
            "vision_freshness_ms_60",
            "video_source_time_ns_60",
        ):
            _dataset(vision, name.removeprefix("vision_"), arrays[name], compression)
        vision.attrs["view_order"] = _canonical_json(["head", "left", "right"])
        vision.attrs["feature_components"] = _canonical_json(
            {"cls": [0, 768], "mean_patch_excluding_registers": [768, 1536]}
        )

        atomic = handle.create_group("annotations/atomic_segments")
        segments = annotation["segments"]
        for name, dtype in (
            ("start_frame_60", np.int32),
            ("end_frame_60_exclusive", np.int32),
            ("class_index", np.int16),
        ):
            _dataset(atomic, name, np.asarray([row[name] for row in segments], dtype=dtype), compression)
        for name in ("segment_id", "label_id", "action_name", "group_id", "hand"):
            _string_dataset(atomic, name, (row.get(name) for row in segments), h5py)

        subtasks = handle.create_group("annotations/subtasks")
        subtask_rows = annotation["subtasks"]
        for name in ("start_frame_60", "end_frame_60_exclusive"):
            _dataset(
                subtasks,
                name,
                np.asarray([row[name] for row in subtask_rows], dtype=np.int32),
                compression,
            )
        for name in ("type", "caption"):
            _string_dataset(subtasks, name, (row.get(name) for row in subtask_rows), h5py)

        if decode_errors:
            errors = handle.create_group("qa")
            _string_dataset(errors, "vision_decode_errors_json", map(_canonical_json, decode_errors), h5py)
        handle.attrs["complete"] = True
        handle.flush()
    os.replace(partial, output)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "written",
        "mcap_path": row["mcap_path"],
        "atomic_json_path": row["atomic_json_path"],
        "episode_id": row["episode_id"],
        "cache_path": str(output),
        "cache_size_bytes": output.stat().st_size,
        "annotation_frame_count": int(row["annotation_frame_count"]),
        "tactile_hand_correction": row["tactile_hand_correction"],
        "correction_manifest_sha256": authority.correction_sha256,
        "tactile_mapping_sha256": authority.mapping_sha256,
        "formal_mcap_schema_report_sha256": schema_contract.report_sha256,
        "vision_weights_sha256": backbone.weights_sha256,
        "vision_lineage": dict(
            getattr(
                backbone,
                "lineage",
                {"weights_sha256": backbone.weights_sha256},
            )
        ),
        "source_atomic_json_sha256": annotation_hash,
        "topic_count_delta_from_annotation": topic_count_delta,
    }


def validate_existing(
    path: str | Path,
    *,
    row: Mapping[str, Any],
    authority: Authority,
    schema_contract: McapSchemaContract,
    weights_sha256: str,
    config_sha256: str,
    ontology_sha256: str,
    vision_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import h5py

    source = Path(path)
    with h5py.File(source, "r") as handle:
        if handle.attrs.get("schema_version") != SCHEMA_VERSION or not bool(
            handle.attrs.get("complete", False)
        ):
            raise ValueError(f"incomplete/stale cache: {source}")
        metadata = json.loads(handle.attrs["metadata_json"])
        expected = {
            "source_mcap_path": row["mcap_path"],
            "source_atomic_json_path": row["atomic_json_path"],
            "correction_manifest_sha256": authority.correction_sha256,
            "tactile_mapping_sha256": authority.mapping_sha256,
            "formal_mcap_schema_report_sha256": schema_contract.report_sha256,
            "vision_weights_sha256": weights_sha256,
            "config_sha256": config_sha256,
            "ontology_sha256": ontology_sha256,
            "tactile_hand_correction": row["tactile_hand_correction"],
        }
        for name, value in expected.items():
            if metadata.get(name) != value:
                raise ValueError(f"existing cache {name} is stale: {source}")
        if vision_lineage is not None and metadata.get("vision_lineage") != dict(
            vision_lineage
        ):
            raise ValueError(f"existing cache vision_lineage is stale: {source}")
        ticks = int(row["annotation_frame_count"])
        if handle["sensors/tactile/value"].shape != (ticks, 2, 360):
            raise ValueError(f"existing tactile shape is invalid: {source}")
        if handle["vision/features"].shape != ((ticks + 1) // 2, 3, 1536):
            raise ValueError(f"existing vision shape is invalid: {source}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "skipped_verified",
        "mcap_path": row["mcap_path"],
        "atomic_json_path": row["atomic_json_path"],
        "episode_id": row["episode_id"],
        "cache_path": str(source),
        "cache_size_bytes": source.stat().st_size,
        "annotation_frame_count": int(row["annotation_frame_count"]),
        "tactile_hand_correction": row["tactile_hand_correction"],
        "correction_manifest_sha256": authority.correction_sha256,
        "tactile_mapping_sha256": authority.mapping_sha256,
        "formal_mcap_schema_report_sha256": schema_contract.report_sha256,
        "vision_weights_sha256": weights_sha256,
        "topic_count_delta_from_annotation": metadata.get(
            "topic_count_delta_from_annotation", {}
        ),
    }


def _load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("export config must be a YAML mapping")
    return config, hashlib.sha256(raw).hexdigest()


def _distributed_identity(args: argparse.Namespace) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", args.rank))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
    if not 0 <= local_rank < world_size:
        raise ValueError(f"invalid local_rank/world_size: {local_rank}/{world_size}")
    return rank, local_rank, world_size


def _auto_cuda_device_index(
    *,
    local_rank: int,
    world_size: int,
    visible_devices: int,
    processes_per_gpu: int,
) -> int:
    """Map local workers to visible GPUs, with explicit multi-process opt-in."""

    if visible_devices < 1:
        raise ValueError("auto CUDA device selection found no visible GPUs")
    if processes_per_gpu < 1:
        raise ValueError("processes_per_gpu must be at least one")
    if processes_per_gpu == 1:
        if world_size > visible_devices:
            raise ValueError(
                f"world_size={world_size} exceeds {visible_devices} visible GPUs; "
                "set processes_per_gpu explicitly to opt into GPU sharing"
            )
        return local_rank
    expected_world_size = visible_devices * processes_per_gpu
    if world_size != expected_world_size:
        raise ValueError(
            f"processes_per_gpu={processes_per_gpu} with {visible_devices} visible "
            f"GPUs requires world_size={expected_world_size}, got {world_size}"
        )
    return local_rank % visible_devices


def _selected_rows(
    rows: Sequence[dict[str, Any]],
    limit: int | None,
    max_source_hours: float | None,
    selection_mode: str = "manifest_order",
    selection_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    if selection_file is not None:
        if limit is not None or max_source_hours is not None:
            raise ValueError(
                "--selection-file cannot be combined with --limit or "
                "--max-source-hours"
            )
        path = Path(selection_file)
        requested = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not requested:
            raise ValueError(f"selection file is empty: {path}")
        non_absolute = [value for value in requested if not Path(value).is_absolute()]
        if non_absolute:
            raise ValueError(
                "selection file must contain absolute mcap_path values; "
                f"first invalid value: {non_absolute[0]!r}"
            )
        seen: set[str] = set()
        duplicates: list[str] = []
        for value in requested:
            if value in seen:
                duplicates.append(value)
            seen.add(value)
        if duplicates:
            raise ValueError(
                "selection file contains duplicate mcap_path values; "
                f"first duplicate: {duplicates[0]!r}"
            )
        by_mcap_path = {str(row["mcap_path"]): row for row in rows}
        missing = [value for value in requested if value not in by_mcap_path]
        if missing:
            raise ValueError(
                "selection file contains mcap_path values absent from the frozen "
                f"authority manifest; first missing value: {missing[0]!r}"
            )
        return [by_mcap_path[value] for value in requested]

    if selection_mode == "manifest_order":
        ordered = list(rows)
    elif selection_mode == "pilot_stratified":
        # Deliberately balance keep/swap in a small engineering pilot and use a
        # stable path hash to spread samples across dates/tasks. This changes
        # only pilot order, never correction decisions or the full corpus.
        by_action = {
            False: sorted(
                (row for row in rows if not bool(row["tactile_swap_left_right"])),
                key=lambda row: hashlib.sha256(
                    str(row["mcap_path"]).encode("utf-8")
                ).hexdigest(),
            ),
            True: sorted(
                (row for row in rows if bool(row["tactile_swap_left_right"])),
                key=lambda row: hashlib.sha256(
                    str(row["mcap_path"]).encode("utf-8")
                ).hexdigest(),
            ),
        }
        ordered = []
        for index in range(max(len(by_action[False]), len(by_action[True]))):
            for action in (False, True):
                if index < len(by_action[action]):
                    ordered.append(by_action[action][index])
    else:
        raise ValueError(f"unknown selection_mode: {selection_mode!r}")
    output: list[dict[str, Any]] = []
    seconds = 0.0
    for row in ordered:
        if limit is not None and len(output) >= limit:
            break
        duration = float(row["duration_s"])
        if max_source_hours is not None and output and seconds + duration > max_source_hours * 3600:
            break
        output.append(row)
        seconds += duration
    return output


def export_command(args: argparse.Namespace) -> int:
    config, config_hash = _load_config(args.config)
    authority = load_authority(
        args.corrections,
        args.mapping,
        expected_correction_sha256=str(config["expected_correction_sha256"]),
        expected_mapping_sha256=str(config["expected_mapping_sha256"]),
        require_frozen_counts=not args.allow_nonfrozen_authority,
    )
    schema_contract = load_mcap_schema_contract(
        config["formal_mcap_schema_report"],
        str(config["expected_mcap_schema_report_sha256"]),
        expected_authority_sha256=authority.correction_sha256,
        min_inspected_files=int(config.get("mcap_schema_min_inspected_files", 32)),
    )
    rows = _selected_rows(
        authority.rows,
        args.limit,
        args.max_source_hours,
        args.selection_mode,
        args.selection_file,
    )
    ontology_actions, ontology_sha256 = load_ontology(
        config["ontology_path"], str(config["expected_ontology_sha256"])
    )
    rank, local_rank, world_size = _distributed_identity(args)
    shard = [row for index, row in enumerate(rows) if index % world_size == rank]
    run_dir = Path(args.output_root) / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stop_file = _validated_stop_file(getattr(args, "stop_file", None))
    _write_rank_state(
        run_dir,
        rank=rank,
        world_size=world_size,
        state="running",
        shard_size=len(shard),
        successful_rows=0,
        failed_rows=0,
        stop_file=stop_file,
        next_shard_index=0 if shard else None,
        next_mcap_path=str(shard[0]["mcap_path"]) if shard else None,
    )
    # This pre-model check makes a pre-existing request cheap and guarantees
    # that no episode starts when the run was already asked to pause.
    if _stop_requested(stop_file):
        state = _write_rank_state(
            run_dir,
            rank=rank,
            world_size=world_size,
            state="planned_pause",
            shard_size=len(shard),
            successful_rows=0,
            failed_rows=0,
            stop_file=stop_file,
            next_shard_index=0 if shard else None,
            next_mcap_path=str(shard[0]["mcap_path"]) if shard else None,
        )
        print(
            _canonical_json(
                {
                    "event": "rank_paused",
                    "pause_kind": "planned_episode_boundary",
                    **state,
                }
            ),
            flush=True,
        )
        return 0
    if not shard:
        state = _write_rank_state(
            run_dir,
            rank=rank,
            world_size=world_size,
            state="complete",
            shard_size=0,
            successful_rows=0,
            failed_rows=0,
            stop_file=stop_file,
        )
        print(_canonical_json({"event": "empty_rank", **state}), flush=True)
        return 0
    visible_cuda_devices: int | None = None
    assigned_cuda_device_index: int | None = None
    processes_per_gpu = int(config.get("processes_per_gpu", 1))
    if args.device == "auto":
        import torch

        visible_cuda_devices = int(torch.cuda.device_count())
        assigned_cuda_device_index = _auto_cuda_device_index(
            local_rank=local_rank,
            world_size=world_size,
            visible_devices=visible_cuda_devices,
            processes_per_gpu=processes_per_gpu,
        )
        device = f"cuda:{assigned_cuda_device_index}"
        torch.cuda.set_device(assigned_cuda_device_index)
    else:
        device = args.device
        if device.startswith("cuda"):
            import torch

            assigned_cuda_device_index = int(torch.device(device).index or 0)
            visible_cuda_devices = int(torch.cuda.device_count())
            torch.cuda.set_device(assigned_cuda_device_index)
    backbone = DinoV3CLSMeanPatch(
        config["model_dir"],
        device,
        precision=str(config.get("precision", "bf16")),
        expected_weights_sha256=str(config["expected_weights_sha256"]),
        image_processor_use_fast=bool(
            config.get("image_processor_use_fast", True)
        ),
        channels_last=bool(config.get("vision_channels_last", False)),
    )
    result_path = run_dir / f"rank{rank:02d}.jsonl"
    failures = 0
    successful_rows = 0
    paused = False
    paused_next_shard_index: int | None = None
    paused_next_mcap_path: str | None = None
    with result_path.open("w", encoding="utf-8", buffering=1) as log:
        for shard_index, row in enumerate(shard):
            # This is the only in-loop stop check.  In particular, no stop
            # request is consulted while decoding an MCAP, running DINOv3, or
            # writing the HDF5.  The previous episode is therefore complete
            # and atomically visible before this branch can be taken.
            if _stop_requested(stop_file):
                paused = True
                paused_next_shard_index = shard_index
                paused_next_mcap_path = str(row["mcap_path"])
                break
            mcap_path = str(row["mcap_path"])
            # The exact absolute path is the only authority lookup key.
            if authority.by_mcap_path.get(mcap_path) is not row:
                raise AssertionError("authority lookup detached from exact mcap_path")
            output = cache_path_for_row(args.output_root, row)
            started = time.monotonic()
            try:
                if not Path(mcap_path).is_file():
                    raise FileNotFoundError(f"MCAP does not exist: {mcap_path}")
                if not Path(str(row["atomic_json_path"])).is_file():
                    raise FileNotFoundError(
                        f"atomic JSON does not exist: {row['atomic_json_path']}"
                    )
                if Path(mcap_path).stat().st_size != int(row["mcap_size_bytes"]):
                    raise ValueError(f"MCAP size differs from frozen manifest: {mcap_path}")
                if output.exists() and args.resume:
                    validation_started = time.perf_counter()
                    record = validate_existing(
                        output,
                        row=row,
                        authority=authority,
                        schema_contract=schema_contract,
                        weights_sha256=backbone.weights_sha256,
                        config_sha256=config_hash,
                        ontology_sha256=ontology_sha256,
                        vision_lineage=backbone.lineage,
                    )
                    stage_elapsed = {
                        "resume_validation_wall_s": (
                            time.perf_counter() - validation_started
                        )
                    }
                else:
                    annotation_started = time.perf_counter()
                    annotation, targets = load_annotation(row, ontology_actions)
                    annotation_wall_s = time.perf_counter() - annotation_started
                    raw = read_episode_once(
                        mcap_path,
                        ticks=int(row["annotation_frame_count"]),
                        backbone=backbone,
                        batch_size=int(config.get("batch_size_images", 96)),
                        skeleton_joints=int(config.get("skeleton_joints", 25)),
                        schema_contract=schema_contract,
                        legacy_tactile_schema_names=frozenset(
                            str(value)
                            for value in config.get("legacy_tactile_schema_names", [])
                        ),
                        max_topic_endpoint_delta=int(
                            config.get("max_topic_endpoint_delta", 1)
                        ),
                        jpeg_decode_workers=int(
                            config.get("jpeg_decode_workers_per_rank", 4)
                        ),
                        vision_prefetch_batches=int(
                            config.get("vision_prefetch_batches", 2)
                        ),
                        vision_pin_memory=bool(
                            config.get("vision_pin_memory", True)
                        ),
                        vision_preprocess_workers=int(
                            config.get("vision_preprocess_workers_per_rank", 1)
                        ),
                    )
                    assemble_started = time.perf_counter()
                    arrays = assemble_episode(
                        row,
                        authority,
                        annotation,
                        targets,
                        raw,
                        min_sensor_valid_fraction=float(
                            config.get("min_sensor_valid_fraction", 0.95)
                        ),
                    )
                    assemble_wall_s = time.perf_counter() - assemble_started
                    write_started = time.perf_counter()
                    record = write_episode_hdf5(
                        output,
                        row=row,
                        annotation=annotation,
                        arrays=arrays,
                        authority=authority,
                        schema_contract=schema_contract,
                        backbone=backbone,
                        config_sha256=config_hash,
                        ontology_actions=ontology_actions,
                        ontology_sha256=ontology_sha256,
                        compression=config.get("hdf5_compression", "lzf"),
                        overwrite=args.overwrite,
                        topic_counts=raw.topic_counts,
                        decode_errors=raw.decode_errors,
                    )
                    hdf5_write_wall_s = time.perf_counter() - write_started
                    stage_elapsed = {
                        "annotation_load_wall_s": annotation_wall_s,
                        **raw.performance,
                        "assemble_wall_s": assemble_wall_s,
                        "hdf5_write_wall_s": hdf5_write_wall_s,
                    }
                record["stage_elapsed_s"] = stage_elapsed
                record["vision_export_runtime"] = {
                    "batch_size_images": int(config.get("batch_size_images", 96)),
                    "jpeg_decode_workers_per_rank": int(
                        config.get("jpeg_decode_workers_per_rank", 4)
                    ),
                    "vision_prefetch_batches": int(
                        config.get("vision_prefetch_batches", 2)
                    ),
                    "vision_pin_memory": bool(
                        config.get("vision_pin_memory", True)
                    ),
                    "vision_preprocess_workers_per_rank": int(
                        config.get("vision_preprocess_workers_per_rank", 1)
                    ),
                    "processes_per_gpu": processes_per_gpu,
                    "visible_cuda_devices": visible_cuda_devices,
                    "assigned_cuda_device_index": assigned_cuda_device_index,
                    "image_processor_use_fast": bool(
                        config.get("image_processor_use_fast", True)
                    ),
                    "vision_channels_last": bool(
                        config.get("vision_channels_last", False)
                    ),
                    "optimization_variant": str(
                        config.get("optimization_variant", "reference")
                    ),
                }
                record["elapsed_s"] = time.monotonic() - started
                record["rank"] = rank
                record["world_size"] = world_size
                log.write(_canonical_json(record) + "\n")
                log.flush()
                successful_rows += 1
                print(
                    _canonical_json(
                        {
                            "event": "episode_complete",
                            "rank": rank,
                            "shard_index": shard_index,
                            "shard_size": len(shard),
                            **record,
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                failures += 1
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "rank": rank,
                    "world_size": world_size,
                    "mcap_path": mcap_path,
                    "atomic_json_path": row.get("atomic_json_path"),
                    "episode_id": row.get("episode_id"),
                    "cache_path": str(output),
                    "elapsed_s": time.monotonic() - started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                log.write(_canonical_json(record) + "\n")
                log.flush()
                print(_canonical_json({"event": "episode_failed", **record}), flush=True)
                if isinstance(error, VisionPipelineFatalError):
                    # Continuing with a worker that failed to shut down could
                    # corrupt later episodes or race the same CUDA model.
                    _write_rank_state(
                        run_dir,
                        rank=rank,
                        world_size=world_size,
                        state="failed",
                        shard_size=len(shard),
                        successful_rows=successful_rows,
                        failed_rows=failures,
                        stop_file=stop_file,
                        next_shard_index=shard_index,
                        next_mcap_path=mcap_path,
                    )
                    raise
                if args.fail_fast:
                    _write_rank_state(
                        run_dir,
                        rank=rank,
                        world_size=world_size,
                        state="failed",
                        shard_size=len(shard),
                        successful_rows=successful_rows,
                        failed_rows=failures,
                        stop_file=stop_file,
                        next_shard_index=shard_index,
                        next_mcap_path=mcap_path,
                    )
                    return 1
    if paused:
        state = _write_rank_state(
            run_dir,
            rank=rank,
            world_size=world_size,
            state="planned_pause",
            shard_size=len(shard),
            successful_rows=successful_rows,
            failed_rows=failures,
            stop_file=stop_file,
            next_shard_index=paused_next_shard_index,
            next_mcap_path=paused_next_mcap_path,
        )
        print(
            _canonical_json(
                {
                    "event": "rank_paused",
                    "pause_kind": "planned_episode_boundary",
                    **state,
                }
            ),
            flush=True,
        )
        return 0
    state = _write_rank_state(
        run_dir,
        rank=rank,
        world_size=world_size,
        state="failed" if failures else "complete",
        shard_size=len(shard),
        successful_rows=successful_rows,
        failed_rows=failures,
        stop_file=stop_file,
    )
    print(
        _canonical_json(
            {
                "event": "rank_complete",
                **state,
            }
        ),
        flush=True,
    )
    # Per-episode failures are finalized only after every rank has completed.
    # Returning non-zero here would make torch elastic terminate healthy peers,
    # turning unprocessed rows into misleading "missing" failures. --fail-fast
    # retains the explicit immediate-abort behavior above.
    return 0


def finalize_command(args: argparse.Namespace) -> int:
    config, _ = _load_config(args.config)
    authority = load_authority(
        args.corrections,
        args.mapping,
        expected_correction_sha256=str(config["expected_correction_sha256"]),
        expected_mapping_sha256=str(config["expected_mapping_sha256"]),
        require_frozen_counts=not args.allow_nonfrozen_authority,
    )
    schema_contract = load_mcap_schema_contract(
        config["formal_mcap_schema_report"],
        str(config["expected_mcap_schema_report_sha256"]),
        expected_authority_sha256=authority.correction_sha256,
        min_inspected_files=int(config.get("mcap_schema_min_inspected_files", 32)),
    )
    rows = _selected_rows(
        authority.rows,
        args.limit,
        args.max_source_hours,
        args.selection_mode,
        args.selection_file,
    )
    run_dir = Path(args.output_root) / "runs" / args.run_id
    records: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for path in sorted(run_dir.glob("rank*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record["mcap_path"])
            if key in records:
                duplicates.append(key)
            records[key] = record
    rank_states: dict[int, dict[str, Any]] = {}
    rank_state_errors: list[str] = []
    for path in sorted(run_dir.glob("rank*.state.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("schema_version") != RANK_STATE_SCHEMA:
                raise ValueError("unexpected schema_version")
            rank = int(state["rank"])
            if rank in rank_states:
                raise ValueError(f"duplicate state for rank {rank}")
            rank_states[rank] = state
        except Exception as error:
            rank_state_errors.append(f"{path}: {type(error).__name__}: {error}")
    expected = {str(row["mcap_path"]) for row in rows}
    successful = {
        key
        for key, record in records.items()
        if record.get("status") in {"written", "skipped_verified"}
    }
    failed_paths = sorted(
        key
        for key, record in records.items()
        if key in expected and record.get("status") == "failed"
    )
    invalid_result_paths = sorted(
        key
        for key, record in records.items()
        if key in expected
        and record.get("status") not in {"written", "skipped_verified", "failed"}
    )
    failures = sorted(expected - successful)
    missing = sorted(expected - set(records))
    extras = sorted(set(records) - expected)
    state_world_sizes = {
        int(state.get("world_size", -1)) for state in rank_states.values()
    }
    expected_rank_ids: set[int] = set()
    if len(state_world_sizes) == 1:
        state_world_size = next(iter(state_world_sizes))
        if state_world_size > 0:
            expected_rank_ids = set(range(state_world_size))
        else:
            rank_state_errors.append(f"invalid world_size in rank states: {state_world_size}")
    elif rank_states:
        rank_state_errors.append(
            f"rank states disagree on world_size: {sorted(state_world_sizes)}"
        )
    missing_rank_states = sorted(expected_rank_ids - set(rank_states))
    extra_rank_states = sorted(set(rank_states) - expected_rank_ids) if expected_rank_ids else []
    if missing_rank_states:
        rank_state_errors.append(f"missing terminal states for ranks: {missing_rank_states}")
    if extra_rank_states:
        rank_state_errors.append(f"unexpected rank states: {extra_rank_states}")
    terminal_states = {"complete", "planned_pause", "failed"}
    nonterminal_rank_states = sorted(
        rank
        for rank, state in rank_states.items()
        if state.get("state") not in terminal_states
    )
    if nonterminal_rank_states:
        rank_state_errors.append(
            f"non-terminal rank states: {nonterminal_rank_states}"
        )
    for rank, state in sorted(rank_states.items()):
        try:
            shard_size = int(state.get("shard_size", -1))
            successful_rows = int(state.get("successful_rows", -1))
            failed_rows = int(state.get("failed_rows", -1))
        except (TypeError, ValueError):
            rank_state_errors.append(f"non-integer row counters in rank {rank} state")
            continue
        consumed_rows = successful_rows + failed_rows
        if min(shard_size, successful_rows, failed_rows) < 0 or consumed_rows > shard_size:
            rank_state_errors.append(f"invalid row counters in rank {rank} state")
        if state.get("state") == "complete" and consumed_rows != shard_size:
            rank_state_errors.append(
                f"rank {rank} claims complete after {consumed_rows}/{shard_size} rows"
            )
        if state.get("state") == "planned_pause" and shard_size > 0:
            next_shard_index = state.get("next_shard_index")
            if (
                not isinstance(next_shard_index, int)
                or isinstance(next_shard_index, bool)
                or next_shard_index != consumed_rows
            ):
                rank_state_errors.append(
                    f"rank {rank} pause boundary disagrees with consumed row count"
                )
    paused_rank_states = [
        state for state in rank_states.values() if state.get("state") == "planned_pause"
    ]
    failed_rank_states = sorted(
        rank for rank, state in rank_states.items() if state.get("state") == "failed"
    )
    stop_files = sorted(
        {
            str(state.get("stop_file"))
            for state in paused_rank_states
            if state.get("stop_file")
        }
    )
    stop_requested = (
        bool(paused_rank_states)
        and len(stop_files) == 1
        and all(state.get("stop_file") == stop_files[0] for state in paused_rank_states)
        and Path(stop_files[0]).is_file()
    )
    # Full row coverage remains the authoritative completion rule, preserving
    # compatibility with runs created before rank-state files existed.
    complete = not (failures or duplicates or extras)
    planned_pause = bool(paused_rank_states) and all(
        (
            stop_requested,
            not failed_paths,
            not invalid_result_paths,
            not duplicates,
            not extras,
            not failed_rank_states,
            not rank_state_errors,
        )
    )
    run_status = "complete" if complete else "paused" if planned_pause else "incomplete"
    audit = {
        "schema_version": "formal500h_export_run_audit_v1",
        "run_id": args.run_id,
        "run_status": run_status,
        "expected_rows": len(expected),
        "successful_rows": len(expected & successful),
        "failed_or_missing_rows": len(failures),
        "failed_rows": len(failed_paths),
        "missing_rows": len(missing),
        "invalid_result_rows": len(invalid_result_paths),
        "duplicate_result_rows": len(duplicates),
        "extra_result_rows": len(extras),
        "stop_requested": stop_requested,
        "paused_rank_count": len(paused_rank_states),
        "rank_state_count": len(rank_states),
        "rank_state_errors": rank_state_errors,
        "failed_rank_states": failed_rank_states,
        "correction_manifest_sha256": authority.correction_sha256,
        "tactile_mapping_sha256": authority.mapping_sha256,
        "formal_mcap_schema_report_sha256": schema_contract.report_sha256,
        "failures": failures,
        "failed_paths": failed_paths,
        "missing_paths": missing,
        "invalid_result_paths": invalid_result_paths,
        "duplicates": sorted(set(duplicates)),
        "extras": extras,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "audit.json"
    _atomic_write_json(audit_path, audit)
    manifest = Path(args.output_root) / f"cache_manifest.{args.run_id}.jsonl"
    pause_path = run_dir / "pause.json"
    if run_status == "paused":
        # A partial manifest is deliberately never published under the cache
        # manifest name consumed by training.  Resume discovers and verifies
        # completed HDF5 files directly from the same output root.
        manifest.unlink(missing_ok=True)
        pause = {
            "schema_version": PAUSE_SCHEMA,
            "run_id": args.run_id,
            "run_status": "paused",
            "pause_kind": "planned_episode_boundary",
            "completed_rows": len(expected & successful),
            "remaining_rows": len(failures),
            "stop_files": stop_files,
            "rank_states": [rank_states[rank] for rank in sorted(rank_states)],
            "cache_manifest_published": False,
            "audit": str(audit_path),
            "created_unix_s": time.time(),
        }
        _atomic_write_json(pause_path, pause)
        print(
            _canonical_json(
                {
                    "event": "finalize_paused",
                    "pause": str(pause_path),
                    "audit": str(audit_path),
                    **audit,
                }
            ),
            flush=True,
        )
        return 0
    if run_status == "incomplete":
        manifest.unlink(missing_ok=True)
        pause_path.unlink(missing_ok=True)
        print(_canonical_json({"event": "finalize_incomplete", **audit}), flush=True)
        return 1
    pause_path.unlink(missing_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(records[str(row["mcap_path"])]) + "\n")
    print(
        _canonical_json(
            {
                "event": "finalize_complete",
                "cache_manifest": str(manifest),
                "audit": str(audit_path),
                **audit,
            }
        ),
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "finalize"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--corrections", type=Path, required=True)
        child.add_argument("--mapping", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--run-id", required=True)
        child.add_argument("--limit", type=int)
        child.add_argument("--max-source-hours", type=float)
        child.add_argument(
            "--selection-file",
            type=Path,
            help=(
                "UTF-8 text file containing one absolute mcap_path per line. "
                "Used for exact reproducible subsets and mutually exclusive "
                "with --limit/--max-source-hours."
            ),
        )
        child.add_argument(
            "--selection-mode",
            choices=("manifest_order", "pilot_stratified"),
            default="manifest_order",
        )
        child.add_argument("--allow-nonfrozen-authority", action="store_true")
    export = subparsers.choices["export"]
    export.add_argument("--device", default="auto")
    export.add_argument("--rank", type=int, default=0)
    export.add_argument("--local-rank", type=int, default=0)
    export.add_argument("--world-size", type=int, default=1)
    export.add_argument("--resume", action="store_true")
    export.add_argument("--overwrite", action="store_true")
    export.add_argument("--fail-fast", action="store_true")
    export.add_argument(
        "--stop-file",
        type=Path,
        help=(
            "Absolute shared path used for a planned graceful pause. Each rank "
            "checks only before starting an episode, so an in-flight episode is "
            "fully committed before the rank exits successfully."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "resume", False) and getattr(args, "overwrite", False):
        parser.error("--resume and --overwrite are mutually exclusive")
    status = export_command(args) if args.command == "export" else finalize_command(args)
    raise SystemExit(status)


if __name__ == "__main__":  # pragma: no cover
    main()
