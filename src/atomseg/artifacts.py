from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping


HASH_CHUNK_BYTES = 8 * 1024 * 1024


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable JSON representation used by all protocol hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_stream(handle: BinaryIO, chunk_bytes: int = HASH_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(chunk_bytes)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = HASH_CHUNK_BYTES) -> str:
    with Path(path).open("rb") as handle:
        return sha256_stream(handle, chunk_bytes=chunk_bytes)


def quick_file_fingerprint(path: str | Path, sample_bytes: int = 1024 * 1024) -> str:
    """Hash size plus the first/last bytes for discovery-only audits.

    This is deliberately named a fingerprint rather than a checksum.  Formal
    releases must use :func:`sha256_file`.
    """

    file_path = Path(path)
    size = file_path.stat().st_size
    digest = hashlib.sha256()
    digest.update(f"size:{size}\n".encode("ascii"))
    with file_path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def hash_paths(paths: Iterable[str | Path]) -> str:
    rows = []
    for value in sorted((Path(path) for path in paths), key=lambda path: str(path)):
        rows.append({"path": str(value), "sha256": sha256_file(value)})
    return sha256_json(rows)


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, value: Any, *, indent: int = 2) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        allow_nan=False,
    ).encode("utf-8")
    atomic_write_bytes(path, payload + b"\n")


def artifact_record(
    *,
    artifact_type: str,
    schema_version: str,
    payload_sha256: str,
    config_sha256: str,
    inputs: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a uniform, self-hashing artifact metadata record."""

    record: dict[str, Any] = {
        "artifact_type": artifact_type,
        "schema_version": schema_version,
        "payload_sha256": payload_sha256,
        "config_sha256": config_sha256,
        "inputs": dict(sorted((inputs or {}).items())),
    }
    if extra:
        record.update(extra)
    record["artifact_metadata_sha256"] = sha256_json(record)
    return record


def verify_artifact_record(record: Mapping[str, Any]) -> None:
    expected = record.get("artifact_metadata_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Artifact metadata is missing artifact_metadata_sha256")
    unhashed = dict(record)
    unhashed.pop("artifact_metadata_sha256", None)
    actual = sha256_json(unhashed)
    if actual != expected:
        raise ValueError(
            f"Artifact metadata hash mismatch: expected {expected}, computed {actual}"
        )


def require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be a non-null 64-character SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field_name} is not hexadecimal SHA256") from error
    return value.lower()
