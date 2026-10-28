"""Read-only contract helpers for the Formal500h H0 experiments.

H0 is the canonical *no hand-loss* training condition.  This module is
deliberately independent of the historical runners: importing it does not
touch global model state, environment variables, files, or the old protocol
implementation.  Runners may use :func:`assert_h0_config` at their boundary
while old/legacy runners continue to exist unchanged.

The validator accepts the common flat ``resolved_config.json`` layout as well
as the nested ``data``/``model``/``training`` layouts used by the experiment
tools.  It never edits the supplied mapping.  ``validate_h0_config`` returns
a JSON-serialisable report; use ``assert_h0_config`` when a fail-closed
exception is desired.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any


H0_SCHEMA_VERSION = "formal500h_h0_contract_v1"
H0_NUM_CLASSES = 26
H0_OUTPUT_RATE_HZ = 30
H0_HAND_WEIGHT = 0.0
H0_HAND_LOSS_ENABLED = False
H0_BATCH_SIZE_PER_GPU = 16
H0_WORKERS_PER_JOB = 20
H0_EPOCHS = 3
H0_LEARNING_RATE = 3.0e-4
H0_TEST_MANIFEST_READ = False
H0_TEST_SEALED = True

# Short aliases mirror the names used by the existing runners while keeping
# the explicit ``_PER_GPU``/``_PER_JOB`` names in the serialized contract.
H0_BATCH_SIZE = H0_BATCH_SIZE_PER_GPU
H0_NUM_WORKERS = H0_WORKERS_PER_JOB
H0_NUM_ACTIONS = H0_NUM_CLASSES
H0_OUTPUT_RATE = H0_OUTPUT_RATE_HZ
H0_BATCH = H0_BATCH_SIZE_PER_GPU
H0_WORKERS = H0_WORKERS_PER_JOB


# A shallow immutable mapping is sufficient because all values in the public
# contract are scalars.  ``h0_contract()`` below returns a normal dict for
# JSON writers, so callers cannot mutate this canonical object by accident.
H0_CONTRACT = MappingProxyType(
    {
        "schema_version": H0_SCHEMA_VERSION,
        "num_classes": H0_NUM_CLASSES,
        "output_rate_hz": H0_OUTPUT_RATE_HZ,
        "hand_weight": H0_HAND_WEIGHT,
        "hand_loss_enabled": H0_HAND_LOSS_ENABLED,
        "batch_size_per_gpu": H0_BATCH_SIZE_PER_GPU,
        "workers_per_job": H0_WORKERS_PER_JOB,
        "epochs": H0_EPOCHS,
        "learning_rate": H0_LEARNING_RATE,
        "test_manifest_read": H0_TEST_MANIFEST_READ,
        "test_sealed": H0_TEST_SEALED,
    }
)


def h0_contract() -> dict[str, Any]:
    """Return a mutable copy suitable for embedding in a run manifest."""

    return dict(H0_CONTRACT)


# Each field can occur in one or more established config layouts.  If more
# than one path is present, all values must agree; this catches a stale flat
# override silently disagreeing with a nested training section.
_FIELD_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "num_classes": (
        ("num_classes",),
        ("num_actions",),
        ("model", "num_classes"),
        ("model", "num_actions"),
        ("data", "num_classes"),
    ),
    "output_rate_hz": (
        ("output_rate_hz",),
        ("model", "output_rate_hz"),
        ("data", "output_rate_hz"),
        ("evaluation", "output_rate_hz"),
    ),
    "hand_weight": (
        ("hand_weight",),
        ("loss", "hand_weight"),
        ("model", "hand_weight"),
        ("training", "hand_weight"),
    ),
    "hand_loss_enabled": (
        ("hand_loss_enabled",),
        ("loss", "hand_loss_enabled"),
        ("model", "hand_loss_enabled"),
        ("training", "hand_loss_enabled"),
    ),
    "batch_size_per_gpu": (
        ("batch_size_per_gpu",),
        ("batch_size",),
        ("training", "batch_size_per_gpu"),
        ("training", "batch_size"),
    ),
    "workers_per_job": (
        ("workers_per_job",),
        ("num_workers",),
        ("training", "workers_per_job"),
        ("training", "num_workers"),
    ),
    "epochs": (("epochs",), ("training", "epochs")),
    "learning_rate": (
        ("learning_rate",),
        ("training", "learning_rate"),
        ("optimizer", "learning_rate"),
    ),
    "test_manifest_read": (
        ("test_manifest_read",),
        ("evaluation", "test_manifest_read"),
        ("training", "test_manifest_read"),
    ),
    "test_sealed": (
        ("test_sealed",),
        ("evaluation", "test_sealed"),
        ("protocol", "test_sealed"),
    ),
}


def _lookup(config: Mapping[str, Any], path: Sequence[str]) -> tuple[bool, Any]:
    """Look up a path without mutating or invoking mapping defaults."""

    current: Any = config
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _observed(config: Mapping[str, Any], field: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for path in _FIELD_PATHS[field]:
        present, value = _lookup(config, path)
        if present:
            values[".".join(path)] = value
    return values


def _scalar_equal(field: str, value: Any, expected: Any) -> bool:
    if field in {"hand_loss_enabled", "test_manifest_read", "test_sealed"}:
        return type(value) is bool and value is expected
    if field in {
        "num_classes",
        "output_rate_hz",
        "batch_size_per_gpu",
        "workers_per_job",
        "epochs",
    }:
        return (
            isinstance(value, (Integral, Real))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value).is_integer()
            and int(value) == expected
        )
    if field in {"hand_weight", "learning_rate"}:
        return (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and math.isclose(
                float(value),
                float(expected),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        )
    return value == expected


def validate_h0_config(
    config: Mapping[str, Any], *, raise_on_error: bool = False
) -> dict[str, Any]:
    """Validate a resolved config against the frozen H0 scalar contract.

    Parameters
    ----------
    config:
        A mapping, normally the parsed ``resolved_config.json`` or a protocol
        dictionary.  It is read only.
    raise_on_error:
        If true, raise :class:`ValueError` when any check fails.

    Returns
    -------
    dict
        A JSON-serialisable report with ``passed``, per-field ``checks``,
        observed paths, and human-readable ``errors``.
    """

    if not isinstance(config, Mapping):
        report = {
            "schema_version": H0_SCHEMA_VERSION,
            "passed": False,
            "checks": {},
            "observed": {},
            "errors": ["config must be a mapping"],
        }
        if raise_on_error:
            raise ValueError("H0 contract failed: config must be a mapping")
        return report

    checks: dict[str, bool] = {}
    observed: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    # ``schema_version`` is optional for compatibility with resolved configs
    # emitted before this helper existed; when present, it must identify H0.
    schema_present = "schema_version" in config
    checks["schema_version"] = (
        not schema_present or config.get("schema_version") == H0_SCHEMA_VERSION
    )
    if not checks["schema_version"]:
        errors.append(
            f"schema_version={config.get('schema_version')!r} != {H0_SCHEMA_VERSION!r}"
        )

    expected_fields = {
        "num_classes": H0_NUM_CLASSES,
        "output_rate_hz": H0_OUTPUT_RATE_HZ,
        "hand_weight": H0_HAND_WEIGHT,
        "hand_loss_enabled": H0_HAND_LOSS_ENABLED,
        "batch_size_per_gpu": H0_BATCH_SIZE_PER_GPU,
        "workers_per_job": H0_WORKERS_PER_JOB,
        "epochs": H0_EPOCHS,
        "learning_rate": H0_LEARNING_RATE,
    }
    for field, expected in expected_fields.items():
        values = _observed(config, field)
        observed[field] = values
        checks[field] = bool(values) and all(
            _scalar_equal(field, value, expected) for value in values.values()
        )
        if not values:
            errors.append(f"missing required H0 field: {field}")
        elif not checks[field]:
            details = ", ".join(f"{path}={value!r}" for path, value in values.items())
            errors.append(f"{field} must be {expected!r}; observed {details}")

    read_values = _observed(config, "test_manifest_read")
    sealed_values = _observed(config, "test_sealed")
    observed["test_manifest_read"] = read_values
    observed["test_sealed"] = sealed_values
    checks["test_manifest_read"] = bool(read_values) and all(
        _scalar_equal("test_manifest_read", value, H0_TEST_MANIFEST_READ)
        for value in read_values.values()
    )
    checks["test_sealed"] = bool(sealed_values) and all(
        _scalar_equal("test_sealed", value, H0_TEST_SEALED)
        for value in sealed_values.values()
    )
    # A run may use the older, explicit ``test_manifest_read=false`` spelling
    # without a separate ``test_sealed`` key.  That is still fail-closed with
    # respect to test access, while a contradictory explicit key fails.
    checks["test_sealed"] = checks["test_sealed"] or (
        checks["test_manifest_read"] and not sealed_values
    )
    if not read_values:
        errors.append("missing required H0 field: test_manifest_read=false")
    elif not checks["test_manifest_read"]:
        errors.append("test_manifest_read must be the boolean false")
    if sealed_values and not all(
        _scalar_equal("test_sealed", value, H0_TEST_SEALED)
        for value in sealed_values.values()
    ):
        errors.append("test_sealed must be the boolean true")
    elif not sealed_values and not checks["test_manifest_read"]:
        errors.append("test_sealed is not established")

    report = {
        "schema_version": H0_SCHEMA_VERSION,
        "passed": not errors,
        "checks": checks,
        "observed": observed,
        "errors": errors,
    }
    if raise_on_error and errors:
        raise ValueError("H0 contract failed: " + "; ".join(errors))
    return report


def is_h0_config(config: Mapping[str, Any]) -> bool:
    """Return true exactly when :func:`validate_h0_config` passes."""

    return bool(validate_h0_config(config)["passed"])


def assert_h0_config(config: Mapping[str, Any]) -> None:
    """Fail closed if *config* is not the canonical H0 recipe."""

    validate_h0_config(config, raise_on_error=True)


# Friendly aliases for callers that use “contract” rather than “config” in
# their naming.  Keeping these aliases here avoids duplicating validation
# logic in each runner.
validate_h0_contract = validate_h0_config
assert_h0_contract = assert_h0_config


__all__ = [
    "H0_SCHEMA_VERSION",
    "H0_NUM_CLASSES",
    "H0_OUTPUT_RATE_HZ",
    "H0_HAND_WEIGHT",
    "H0_HAND_LOSS_ENABLED",
    "H0_BATCH_SIZE_PER_GPU",
    "H0_WORKERS_PER_JOB",
    "H0_BATCH_SIZE",
    "H0_NUM_WORKERS",
    "H0_NUM_ACTIONS",
    "H0_OUTPUT_RATE",
    "H0_BATCH",
    "H0_WORKERS",
    "H0_EPOCHS",
    "H0_LEARNING_RATE",
    "H0_TEST_MANIFEST_READ",
    "H0_TEST_SEALED",
    "H0_CONTRACT",
    "h0_contract",
    "validate_h0_config",
    "is_h0_config",
    "assert_h0_config",
    "validate_h0_contract",
    "assert_h0_contract",
]
