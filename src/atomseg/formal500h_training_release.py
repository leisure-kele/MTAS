from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


RELEASE_SCHEMA = "formal500h_training_universe_v1"
DEFAULT_MAXIMUM_FAIL_RATIO = 0.5
GOOD_HDF5_STATUSES = {"written", "skipped_verified"}
EXPECTED_CORRECTION_SHA256 = (
    "91271fbf49e1bdbd1902c11fd7d7d89953ab847a52ce09caa4022ef73f294829"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, path)


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
    return rows


def _latest_production_rows(run_dir: Path) -> dict[str, dict[str, Any]]:
    paths = sorted(run_dir.glob("rank*.jsonl"))
    if not paths:
        raise ValueError(f"no rank*.jsonl files in {run_dir}")
    latest: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("{"):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mcap_path = row.get("mcap_path")
                if mcap_path and row.get("status"):
                    latest[Path(str(mcap_path)).stem] = row
    return latest


def source_group_id(mcap_path: str) -> tuple[str, str]:
    """Return a conservative collection-code group without claiming participant ID."""

    directory = Path(mcap_path).parent.name
    normalized = directory.replace(" ", "_")
    match = re.search(r"(?:^|_)([ABCD]\d[A-Z]+)$", normalized)
    if match:
        return f"collection_code:{match.group(1)}", "collection_code"
    return f"capture_directory:{directory}", "capture_directory"


def _quality_rows(path: Path) -> tuple[dict[str, Mapping[str, Any]], int]:
    document = json.loads(path.read_text(encoding="utf-8"))
    episodes = document.get("episodes") if isinstance(document, Mapping) else None
    if not isinstance(episodes, list):
        raise ValueError("quality JSON must contain an episodes list")
    output: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(episodes):
        if not isinstance(row, Mapping) or not row.get("stem"):
            raise ValueError(f"invalid quality row {index}")
        stem = str(row["stem"])
        if stem in output:
            raise ValueError(f"duplicate quality stem: {stem}")
        output[stem] = row
    return output, len(episodes)


def _correction_rows(path: Path) -> tuple[dict[str, Mapping[str, Any]], int]:
    output: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(_read_jsonl(path)):
        mcap_path = row.get("mcap_path")
        if not mcap_path:
            raise ValueError(f"invalid correction row {index}: missing mcap_path")
        stem = str(row.get("capture_key") or Path(str(mcap_path)).stem)
        if stem in output:
            raise ValueError(f"duplicate correction stem: {stem}")
        output[stem] = row
    return output, len(output)


def _qc_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_contact": int(row.get("n_contact") or 0),
        "n_pass": int(row.get("n_pass") or 0),
        "n_fail": int(row.get("n_fail") or 0),
        "ratio": row.get("ratio"),
        "contact_s": float(row.get("contact_s") or 0.0),
        "pass_s": float(row.get("pass_s") or 0.0),
        "fail_s": float(row.get("fail_s") or 0.0),
        "episode_s": float(row.get("episode_s") or 0.0),
        "tactile_lr_swapped": bool(row.get("tactile_lr_swapped", False)),
        "fail_reasons": row.get("fail_reasons") or {},
        "error": row.get("error"),
    }


def _classify_quality(
    row: Mapping[str, Any], maximum_fail_ratio: float
) -> tuple[bool, str, dict[str, Any]]:
    quality = _qc_payload(row)
    if quality["error"]:
        return False, "quality_error", quality
    n_contact = quality["n_contact"]
    n_pass = quality["n_pass"]
    n_fail = quality["n_fail"]
    if min(n_contact, n_pass, n_fail) < 0 or n_pass + n_fail != n_contact:
        raise ValueError("n_pass + n_fail does not equal n_contact")
    if n_contact == 0:
        return True, "keep_no_annotation_required_contact", quality
    fail_ratio = n_fail / n_contact
    try:
        stored_ratio = float(quality["ratio"])
    except (TypeError, ValueError):
        raise ValueError("non-numeric stored ratio") from None
    if not math.isfinite(stored_ratio) or abs(stored_ratio - fail_ratio) > 1.0e-9:
        raise ValueError("stored ratio differs from n_fail / n_contact")
    if fail_ratio <= maximum_fail_ratio:
        return True, "keep_fail_ratio_at_most_threshold", quality
    return False, "exclude_fail_ratio_above_threshold", quality


def _validate_lr_correction_contract(
    correction_row: Mapping[str, Any], quality_row: Mapping[str, Any]
) -> bool:
    source_swap = correction_row.get("tactile_swap_left_right")
    quality_swap = quality_row.get("tactile_lr_swapped")
    if type(source_swap) is not bool:
        raise ValueError("correction row lacks a boolean tactile_swap_left_right")
    expected_action = "swap_left_right" if source_swap else "keep"
    if correction_row.get("tactile_hand_correction") != expected_action:
        raise ValueError("left/right correction fields disagree")
    if type(quality_swap) is not bool:
        raise ValueError("quality row lacks a boolean tactile_lr_swapped")
    return quality_swap != source_swap


def _hours(rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(row["duration_s"]) for row in rows) / 3600.0


def build_training_release(
    *,
    formal_manifest: Path,
    correction_manifest: Path,
    quality_filter: Path,
    production_run_dir: Path,
    output_dir: Path,
    maximum_fail_ratio: float = DEFAULT_MAXIMUM_FAIL_RATIO,
    expected_correction_sha256: str | None = None,
) -> dict[str, Any]:
    if not math.isfinite(maximum_fail_ratio) or not 0.0 <= maximum_fail_ratio <= 1.0:
        raise ValueError("maximum_fail_ratio must be finite and in [0,1]")
    formal = _read_jsonl(formal_manifest)
    correction_hash = sha256_file(correction_manifest)
    if (
        expected_correction_sha256
        and correction_hash != expected_correction_sha256
    ):
        raise ValueError(
            "correction manifest SHA256 mismatch: "
            f"{correction_hash} != {expected_correction_sha256}"
        )
    corrections, correction_count = _correction_rows(correction_manifest)
    quality, quality_count = _quality_rows(quality_filter)
    production = _latest_production_rows(production_run_dir)
    seen: set[str] = set()
    eligible: list[dict[str, Any]] = []
    trainable: list[dict[str, Any]] = []
    missing_hdf5: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    bad_quality: list[str] = []
    missing_quality: list[str] = []
    missing_correction: list[str] = []
    qc_lr_correction_mismatch = 0

    for source in formal:
        stem = str(source["capture_key"])
        if stem in seen:
            raise ValueError(f"duplicate formal capture key: {stem}")
        seen.add(stem)
        quality_row = quality.get(stem)
        if quality_row is None:
            missing_quality.append(stem)
            continue
        correction_row = corrections.get(stem)
        if correction_row is None:
            missing_correction.append(stem)
            continue
        try:
            if correction_row.get("mcap_path") != source.get("mcap_path"):
                raise ValueError("formal and correction MCAP paths differ")
            lr_mismatch = _validate_lr_correction_contract(
                correction_row, quality_row
            )
            keep, reason, qc = _classify_quality(quality_row, maximum_fail_ratio)
        except ValueError as error:
            bad_quality.append(f"{stem}: {error}")
            continue
        production_row = production.get(stem)
        hdf5_status = str(production_row.get("status")) if production_row else "missing"
        cache_path = str(production_row.get("cache_path")) if production_row and production_row.get("cache_path") else None
        cache_size_bytes = (
            int(production_row.get("cache_size_bytes") or 0)
            if production_row
            else 0
        )
        group_id, group_basis = source_group_id(str(source["mcap_path"]))
        record = {
            **source,
            "training_release_schema": RELEASE_SCHEMA,
            "split_group_id": group_id,
            "split_group_basis": group_basis,
            "tactile_hand_correction": correction_row["tactile_hand_correction"],
            "tactile_swap_left_right": correction_row["tactile_swap_left_right"],
            "correction_manifest_sha256": correction_hash,
            "tactile_qc_lr_differs_from_formal_correction": lr_mismatch,
            "tactile_qc_keep": keep,
            "tactile_qc_reason": reason,
            "tactile_qc": qc,
            "hdf5_status": hdf5_status,
            "cache_path": cache_path,
            "cache_size_bytes": cache_size_bytes,
        }
        qc_lr_correction_mismatch += int(lr_mismatch)
        if not keep:
            excluded.append(record)
            continue
        eligible.append(record)
        if hdf5_status in GOOD_HDF5_STATUSES and cache_path:
            trainable.append(record)
        else:
            missing_hdf5.append(record)

    if missing_quality or missing_correction or bad_quality:
        raise ValueError(
            "formal/correction/quality join is not clean: "
            f"missing_quality={len(missing_quality)}, "
            f"missing_correction={len(missing_correction)}, "
            f"invalid={len(bad_quality)}, "
            f"examples={(missing_quality + missing_correction + bad_quality)[:5]}"
        )
    if len(eligible) + len(excluded) != len(formal):
        raise AssertionError("release classification does not cover the formal universe")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "eligible_all": output_dir / "eligible_all.jsonl",
        "trainable_hdf5": output_dir / "trainable_hdf5.jsonl",
        "trainable_hdf5_paths": output_dir / "trainable_hdf5_paths.txt",
        "eligible_missing_hdf5": output_dir / "eligible_missing_hdf5.jsonl",
        "excluded_quality": output_dir / "excluded_quality.jsonl",
    }
    for name, rows in (
        ("eligible_all", eligible),
        ("trainable_hdf5", trainable),
        ("eligible_missing_hdf5", missing_hdf5),
        ("excluded_quality", excluded),
    ):
        _write_jsonl(paths[name], rows)
    path_list_temporary = paths["trainable_hdf5_paths"].with_name(
        f".{paths['trainable_hdf5_paths'].name}.{os.getpid()}.tmp"
    )
    path_list_temporary.write_text(
        "".join(f"{row['cache_path']}\n" for row in trainable), encoding="utf-8"
    )
    os.replace(path_list_temporary, paths["trainable_hdf5_paths"])
    artifact_sha256 = {name: sha256_file(path) for name, path in paths.items()}
    summary = {
        "schema_version": RELEASE_SCHEMA,
        "status": "complete",
        "created_unix_s": time.time(),
        "hostname": socket.gethostname(),
        "policy": {
            "maximum_fail_ratio": maximum_fail_ratio,
            "comparison": "n_fail / n_contact <= maximum_fail_ratio",
            "keep_exactly_equal_threshold": True,
            "keep_no_annotation_required_contact": True,
            "quality_error_policy": "release_build_fails",
            "missing_quality_policy": "release_build_fails",
            "quality_lr_swap_policy": (
                "consume the frozen external QC decision as-is; the formal "
                "correction manifest independently governs HDF5 hand order"
            ),
        },
        "sources": {
            "formal_manifest": str(formal_manifest.resolve()),
            "formal_manifest_sha256": sha256_file(formal_manifest),
            "correction_manifest": str(correction_manifest.resolve()),
            "correction_manifest_sha256": correction_hash,
            "correction_manifest_rows": correction_count,
            "quality_filter": str(quality_filter.resolve()),
            "quality_filter_sha256": sha256_file(quality_filter),
            "quality_report_episodes": quality_count,
            "production_run_dir": str(production_run_dir.resolve()),
        },
        "counts": {
            "formal": len(formal),
            "eligible_all": len(eligible),
            "trainable_hdf5": len(trainable),
            "eligible_missing_hdf5": len(missing_hdf5),
            "excluded_quality": len(excluded),
            "qc_lr_correction_mismatch": qc_lr_correction_mismatch,
            "split_groups_trainable": len({row["split_group_id"] for row in trainable}),
        },
        "hours": {
            "formal": _hours(formal),
            "eligible_all": _hours(eligible),
            "trainable_hdf5": _hours(trainable),
            "eligible_missing_hdf5": _hours(missing_hdf5),
            "excluded_quality": _hours(excluded),
        },
        "storage": {
            "trainable_hdf5_bytes_from_rank_logs": sum(
                int(row["cache_size_bytes"]) for row in trainable
            ),
            "trainable_hdf5_gib_from_rank_logs": sum(
                int(row["cache_size_bytes"]) for row in trainable
            )
            / (1024.0**3),
            "note": "zero means an older rank record did not carry cache_size_bytes",
        },
        "artifacts": {name: str(path.resolve()) for name, path in paths.items()},
        "artifact_sha256": artifact_sha256,
        "note": (
            "This directory contains manifests and paths only. HDF5 payloads remain "
            "in the immutable production cache and are not duplicated."
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze the tactile-QC formal500h training universe")
    parser.add_argument("--formal-manifest", required=True)
    parser.add_argument("--correction-manifest", required=True)
    parser.add_argument(
        "--expected-correction-sha256",
        default=EXPECTED_CORRECTION_SHA256,
    )
    parser.add_argument("--quality-filter", required=True)
    parser.add_argument("--production-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maximum-fail-ratio", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    build_training_release(
        formal_manifest=Path(arguments.formal_manifest),
        correction_manifest=Path(arguments.correction_manifest),
        quality_filter=Path(arguments.quality_filter),
        production_run_dir=Path(arguments.production_run_dir),
        output_dir=Path(arguments.output_dir),
        maximum_fail_ratio=arguments.maximum_fail_ratio,
        expected_correction_sha256=arguments.expected_correction_sha256,
    )


if __name__ == "__main__":
    main()
