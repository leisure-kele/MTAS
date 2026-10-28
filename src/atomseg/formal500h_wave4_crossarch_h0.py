"""H0 release wrapper for the Formal500h cross-architecture matrix.

The historical ``formal500h_wave4_crossarch`` module is intentionally kept
unchanged because it produced a legacy H1 (``hand_weight=0.1``) result.  This
module provides a fail-closed H0 entry point around that implementation:

* the hand objective is forced to zero in a process-local context;
* the resolved model config records ``hand_loss_enabled=false``;
* every run, gate, and aggregate is stamped with the H0 contract and wrapper
  digest; and
* an existing H1/partial artifact is never silently reused.

The temporal heads and M2R2-style representation remain clean-room
adaptations; this is not an official reproduction of any external method.
The formal test manifest is never opened.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.nn import functional as F

from . import formal500h_h0_contract as h0_contract_module
from . import formal500h_wave4_crossarch as legacy
from .formal500h_h0_contract import h0_contract, validate_h0_config


H0_RELEASE_SCHEMA = "formal500h_wave4_crossarch_h0_release_v1"
H0_MODEL_GATE_SCHEMA = legacy.MODEL_GATE_SCHEMA
H0_HAND_WEIGHT = 0.0
H0_HAND_LOSS_ENABLED = False
# The cross-architecture runner deliberately keeps its own training-schema
# identifier in ``resolved_config.json``.  The H0 contract is an orthogonal
# recipe contract (hand-loss, data, optimizer, and test sealing), so rejecting
# the runner's schema here would fail after an otherwise-complete training
# job.  Only this one, known schema is accepted as a compatibility alias; all
# scalar H0 checks below remain strict.
H0_RESOLVED_SCHEMA_ALIASES = frozenset(
    {
        h0_contract_module.H0_SCHEMA_VERSION,
        legacy.TRAINING_SCHEMA,
    }
)
HEADS = legacy.HEADS
REPRESENTATIONS = legacy.REPRESENTATIONS
SEEDS = legacy.SEEDS

# The H0 wrapper delegates most of the actual work to the historical
# cross-architecture implementation.  A checksum of the wrapper alone is
# therefore insufficient: changing an imported decoder, evaluator, or data
# reader could alter a supposedly frozen result without changing the launch
# manifest.  Keep an explicit, reviewable source closure here.  The shell
# release manifest also lists these paths, so a source drift fails before any
# training process is started.
def _source_tree() -> dict[str, Path]:
    """Return every repository file that can affect a crossarch H0 run."""

    module_dir = Path(__file__).resolve().parent
    return {
        # Package initialization can affect which symbols are available while
        # importing the legacy module, so it is part of the executable source
        # closure rather than an implicit/unhashed packaging detail.
        "package_init": module_dir / "__init__.py",
        "crossarch_h0": Path(__file__).resolve(),
        "crossarch_legacy": Path(legacy.__file__).resolve(),
        "h0_contract": Path(h0_contract_module.__file__).resolve(),
        "wave3b_fixture": module_dir / "formal500h_wave3b_vt.py",
        "wave1_trainer": module_dir / "formal500h_wave1_train.py",
        "wave1_data": module_dir / "formal500h_wave1_data.py",
        # The canonical 60 Hz -> 30 Hz conversion is an executable part of
        # both data preparation and target construction, not incidental
        # utility code.  Bind it into every H0 source closure.
        "clock": module_dir / "formal500h_clock.py",
        "runtime": module_dir / "formal500h_runtime.py",
        "evaluator": module_dir / "formal500h_evaluation.py",
        "infra_runtime": module_dir / "formal500h_infra_runtime.py",
        "protocol": module_dir / "formal500h_protocol.py",
        # ``formal500h_protocol`` imports RELEASE_SCHEMA from this module;
        # changing that constant changes the accepted data contract.
        "training_release": module_dir / "formal500h_training_release.py",
        "model": module_dir / "model.py",
        "baselines": module_dir / "baselines.py",
        "losses": module_dir / "losses.py",
        "inference": module_dir / "inference.py",
    }


def _source_records() -> dict[str, dict[str, str]]:
    """Hash the source closure and fail closed if a dependency is missing."""

    paths = _source_tree()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        details = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise ValueError(f"crossarch H0 source tree is incomplete: {details}")
    return {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in sorted(paths.items())
    }


def _source_tree_sha256(records: Mapping[str, Mapping[str, str]]) -> str:
    """Hash names and digests, independent of host path spelling."""

    canonical = {
        name: str(record["sha256"])
        for name, record in sorted(records.items())
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _h0_context() -> Iterator[None]:
    """Apply the H0 override only within the current Python process."""

    old_weight = legacy.FROZEN_HAND_WEIGHT
    old_model_config = legacy._model_config
    old_loss_terms = legacy.crossarch_loss_terms

    def h0_model_config(
        spec: legacy.CellSpec,
        *,
        d_model: int = 128,
        dropout: float = 0.15,
    ) -> dict[str, Any]:
        value = old_model_config(spec, d_model=d_model, dropout=dropout)
        value = dict(value)
        value["hand_weight"] = H0_HAND_WEIGHT
        value["hand_loss_enabled"] = H0_HAND_LOSS_ENABLED
        value["action_order_negative_policy"] = "unique_episode_only"
        return value

    legacy.FROZEN_HAND_WEIGHT = H0_HAND_WEIGHT
    legacy._model_config = h0_model_config
    # The function is resolved at context-entry time (after module import),
    # so every H0 training call uses the unique-episode negative policy below.
    legacy.crossarch_loss_terms = _h0_crossarch_loss_terms
    try:
        yield
    finally:
        legacy.FROZEN_HAND_WEIGHT = old_weight
        legacy._model_config = old_model_config
        legacy.crossarch_loss_terms = old_loss_terms


def _h0_fields() -> dict[str, Any]:
    source_records = _source_records()
    return {
        "h0_contract": h0_contract(),
        "hand_weight": H0_HAND_WEIGHT,
        "hand_loss_enabled": H0_HAND_LOSS_ENABLED,
        "action_order_negative_policy": "unique_episode_only",
        "h0_wrapper_sha256": _sha256(Path(__file__).resolve()),
        "h0_source_tree_sha256": _source_tree_sha256(source_records),
        "test_manifest_read": False,
    }


def _assert_h0_resolved(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    # ``formal500h_h0_contract_v1`` is the standalone recipe schema, whereas
    # a cross-architecture run correctly records
    # ``formal500h_wave4_crossarch_training_v1`` as its resolved-config
    # schema.  Validate the latter through a copy with the contract schema
    # substituted; never rewrite the producer's resolved config here.
    schema = value.get("schema_version")
    if schema == legacy.TRAINING_SCHEMA:
        contract_value = dict(value)
        contract_value["schema_version"] = h0_contract_module.H0_SCHEMA_VERSION
    else:
        contract_value = value
    report = validate_h0_config(contract_value)
    if not report["passed"]:
        raise ValueError(f"not an H0 resolved config: {path}: {report['errors']}")
    return value


def _stamp_run(output_dir: Path) -> dict[str, Any]:
    """Stamp a legacy-produced run without changing model/checkpoint bytes."""

    summary_path = output_dir / "summary.json"
    complete_path = output_dir / "COMPLETE.json"
    summary = _read_json(summary_path)
    resolved_path = Path(str(summary.get("resolved_config", "")))
    if not resolved_path.is_file():
        raise ValueError(f"missing resolved config: {output_dir}")
    _assert_h0_resolved(resolved_path)
    summary.update(_h0_fields())
    _atomic_json(summary_path, summary)
    complete = _read_json(complete_path)
    complete.update(_h0_fields())
    complete["summary_sha256"] = _sha256(summary_path)
    _atomic_json(complete_path, complete)
    return summary


def build_model_gate(*, output: Path) -> dict[str, Any]:
    """Build a model gate and bind it to H0 without overwriting stale files."""

    output = output.resolve()
    current_source_tree_sha256 = _source_tree_sha256(_source_records())
    if output.exists():
        existing = _read_json(output)
        checks = {
            "h0_schema": existing.get("h0_release_schema") == H0_RELEASE_SCHEMA,
            "h0_contract": existing.get("hand_weight") == H0_HAND_WEIGHT
            and existing.get("hand_loss_enabled") is H0_HAND_LOSS_ENABLED,
            "test_sealed": existing.get("test_manifest_read") is False,
            "wrapper_hash": existing.get("h0_wrapper_sha256")
            == _sha256(Path(__file__).resolve()),
            "source_tree_hash": existing.get("h0_source_tree_sha256")
            == current_source_tree_sha256,
        }
        if all(checks.values()):
            return existing
        raise ValueError(f"stale/incompatible H0 model gate: {checks}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.legacy.tmp")
    try:
        with _h0_context():
            value = legacy.build_model_gate(output=temporary)
        value = dict(value)
        value.update(_h0_fields())
        value["h0_release_schema"] = H0_RELEASE_SCHEMA
        value["purpose"] = "formal_wave4_cross_architecture_h0_model_gate"
        checks = dict(value.get("checks") or {})
        checks["h0_contract"] = True
        checks["test_sealed"] = value.get("test_manifest_read") is False
        checks["source_tree_hash"] = (
            value.get("h0_source_tree_sha256") == current_source_tree_sha256
        )
        value["checks"] = checks
        value["status"] = "pass" if all(checks.values()) else "fail"
        _atomic_json(output, value)
    finally:
        temporary.unlink(missing_ok=True)
    if value["status"] != "pass":
        raise RuntimeError(f"H0 model gate failed: {value['checks']}")
    return value


def audit_release(*, data_dir: Path, output: Path) -> dict[str, Any]:
    """Create a metadata-only H0 release record for the frozen data."""

    data_summary_path = (data_dir / "summary.json").resolve()
    data = _read_json(data_summary_path)
    source_records = _source_records()
    checks = {
        "data_complete": data.get("status") == "complete",
        "data_test_sealed": data.get("test_manifest_read") is False,
        "class_count_26": data.get("class_count") == 26,
        "test_manifest_not_present": "test_manifest" not in data,
        "source_protocol_bound": bool(data.get("source_protocol_sha256")),
        "source_audit_bound": bool(data.get("source_audit_sha256")),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid H0 data release: {checks}")
    result = {
        "schema_version": H0_RELEASE_SCHEMA,
        "status": "complete",
        "purpose": "formal_wave4_cross_architecture_h0_validation_only",
        "data": {
            "summary": str(data_summary_path),
            "summary_sha256": _sha256(data_summary_path),
            "source_protocol_sha256": data.get("source_protocol_sha256"),
            "source_audit_sha256": data.get("source_audit_sha256"),
        },
        "heads": list(HEADS),
        "representations": list(REPRESENTATIONS),
        "seeds": list(SEEDS),
        "jobs": len(HEADS) * len(REPRESENTATIONS) * len(SEEDS),
        "recipe": {
            "precision": legacy.FROZEN_PRECISION,
            "batch_size_per_gpu": legacy.FROZEN_BATCH_SIZE,
            "num_workers": legacy.FROZEN_NUM_WORKERS,
            "epochs": legacy.FROZEN_EPOCHS,
            "learning_rate": legacy.FROZEN_LEARNING_RATE,
            "hand_weight": H0_HAND_WEIGHT,
            "hand_loss_enabled": H0_HAND_LOSS_ENABLED,
            "test_manifest_read": False,
        },
        "sources": {
            name: dict(record) for name, record in source_records.items()
        },
        "source_tree_sha256": _source_tree_sha256(source_records),
        "checks": checks,
        "test_manifest_read": False,
        "test_selection_deferred": True,
    }
    result.update(_h0_fields())
    result["h0_release_schema"] = H0_RELEASE_SCHEMA
    _atomic_json(output, result)
    return result


def validate_release(
    *, release_path: Path, data_dir: Path
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Validate an existing H0 release against current data and source code.

    A release file is an immutable claim about the source tree and frozen
    data.  Checking only its schema allows a stale file from another run to be
    reused, so every recorded digest is compared before training starts.
    Paths are intentionally not compared byte-for-byte: a project may be
    mounted at a different absolute prefix while retaining identical hashes.
    """

    release = _read_json(release_path)
    data_summary_path = (data_dir / "summary.json").resolve()
    data = _read_json(data_summary_path)
    current_sources = _source_records()
    recorded_sources = release.get("sources")
    if not isinstance(recorded_sources, Mapping):
        recorded_sources = {}
    release_data = release.get("data")
    if not isinstance(release_data, Mapping):
        release_data = {}

    checks: dict[str, bool] = {
        "schema": release.get("schema_version") == H0_RELEASE_SCHEMA,
        "status": release.get("status") == "complete",
        "h0_schema": release.get("h0_release_schema") == H0_RELEASE_SCHEMA,
        "h0_contract": release.get("h0_contract") == h0_contract(),
        "h0_weight": release.get("hand_weight") == H0_HAND_WEIGHT,
        "h0_loss_disabled": release.get("hand_loss_enabled") is H0_HAND_LOSS_ENABLED,
        "test_sealed": release.get("test_manifest_read") is False,
        "test_deferred": release.get("test_selection_deferred") is True,
        "data_complete": data.get("status") == "complete",
        "data_test_sealed": data.get("test_manifest_read") is False,
        "data_class_count": data.get("class_count") == 26,
        "data_summary_hash": (
            release_data.get("summary_sha256")
            == _sha256(data_summary_path)
        ),
        "data_protocol_hash": (
            release_data.get("source_protocol_sha256")
            == data.get("source_protocol_sha256")
        ),
        "data_audit_hash": (
            release_data.get("source_audit_sha256")
            == data.get("source_audit_sha256")
        ),
        "source_tree_hash": (
            release.get("source_tree_sha256")
            == _source_tree_sha256(current_sources)
            == release.get("h0_source_tree_sha256")
        ),
        "source_keys_exact": set(recorded_sources) == set(current_sources),
        "heads": release.get("heads") == list(HEADS),
        "representations": release.get("representations") == list(REPRESENTATIONS),
        "seeds": release.get("seeds") == list(SEEDS),
        "jobs": release.get("jobs") == len(HEADS) * len(REPRESENTATIONS) * len(SEEDS),
    }
    recipe = release.get("recipe")
    if not isinstance(recipe, Mapping):
        recipe = {}
    checks.update(
        {
            "recipe_precision": recipe.get("precision") == legacy.FROZEN_PRECISION,
            "recipe_batch": recipe.get("batch_size_per_gpu") == legacy.FROZEN_BATCH_SIZE,
            "recipe_workers": recipe.get("num_workers") == legacy.FROZEN_NUM_WORKERS,
            "recipe_epochs": recipe.get("epochs") == legacy.FROZEN_EPOCHS,
            "recipe_learning_rate": recipe.get("learning_rate") == legacy.FROZEN_LEARNING_RATE,
            "recipe_hand_weight": recipe.get("hand_weight") == H0_HAND_WEIGHT,
            "recipe_hand_loss_disabled": recipe.get("hand_loss_enabled") is H0_HAND_LOSS_ENABLED,
            "recipe_test_sealed": recipe.get("test_manifest_read") is False,
        }
    )

    for name, current in current_sources.items():
        recorded = recorded_sources.get(name)
        checks[f"source:{name}"] = (
            isinstance(recorded, Mapping)
            and recorded.get("sha256") == current.get("sha256")
        )

    if not all(checks.values()):
        failed = {name: value for name, value in checks.items() if not value}
        raise ValueError(f"stale or incompatible crossarch H0 release: {failed}")
    return release, checks


def _unique_episode_action_order_loss(
    representation: torch.Tensor,
    order: torch.Tensor,
    episode_ids: Sequence[Any],
) -> torch.Tensor:
    """Symmetric InfoNCE with same-episode windows masked as negatives.

    The M2R2-style auxiliary target is supervised in this project: ``order``
    is built from the ground-truth action sequence.  Overlapping windows from
    one episode therefore should not be forced apart.  The diagonal positive
    remains available even when a batch contains only one unique episode; in
    that degenerate case the term is zero rather than manufacturing a false
    negative.
    """

    if representation.ndim != 2 or order.ndim != 2:
        raise ValueError("action-order embeddings must have shape [B,D]")
    if representation.shape != order.shape:
        raise ValueError("action-order embeddings must share shape [B,D]")
    if len(episode_ids) != representation.shape[0]:
        raise ValueError("episode_id count does not match action-order batch")
    logits = representation @ order.transpose(0, 1)
    logits = logits / legacy.ACTION_ORDER_TEMPERATURE
    batch_size = logits.shape[0]
    same_episode = torch.tensor(
        [
            str(left) == str(right)
            for left in episode_ids
            for right in episode_ids
        ],
        dtype=torch.bool,
        device=logits.device,
    ).reshape(batch_size, batch_size)
    off_diagonal = ~torch.eye(
        batch_size, dtype=torch.bool, device=logits.device
    )
    logits = logits.masked_fill(
        same_episode & off_diagonal,
        torch.finfo(logits.dtype).min,
    )
    labels = torch.arange(batch_size, device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


def _h0_crossarch_loss_terms(
    criterion: torch.nn.Module,
    output: dict[str, Any],
    batch: dict[str, Any],
    hand_weight: float,
) -> dict[str, torch.Tensor]:
    """H0 loss kernel, preserving legacy terms except for safe negatives."""

    terms = legacy.base_formal500h_loss_terms(
        criterion, output, batch, hand_weight
    )
    zero = terms["loss"].new_zeros(())
    order_loss = zero
    soft_boundary_mse = zero
    if "representation_embedding" in output or "order_embedding" in output:
        if "representation_embedding" not in output or "order_embedding" not in output:
            raise ValueError("incomplete M2R2-style action-order outputs")
        episode_ids = batch.get("episode_id")
        if not isinstance(episode_ids, Sequence) or isinstance(
            episode_ids, (str, bytes, bytearray)
        ):
            raise ValueError("M2R2-style H0 training requires episode_id metadata")
        order_loss = _unique_episode_action_order_loss(
            output["representation_embedding"],
            output["order_embedding"],
            episode_ids,
        )
        if "soft_boundary_logits" not in output:
            raise ValueError("M2R2-style soft-boundary prediction is missing")
        target = criterion.boundary_target(batch)
        mask = batch["boundary_loss_mask"]
        squared = (
            torch.sigmoid(output["soft_boundary_logits"]) - target
        ).square()
        soft_boundary_mse = (
            squared * mask.to(squared.dtype)
        ).sum() / mask.sum().clamp_min(1)
        terms["loss"] = (
            terms["loss"]
            + legacy.ACTION_ORDER_CONTRASTIVE_WEIGHT * order_loss
            + legacy.SOFT_BOUNDARY_MSE_WEIGHT * soft_boundary_mse
        )
    terms["action_order_contrastive"] = order_loss
    terms["soft_boundary_mse"] = soft_boundary_mse
    return terms


def train_cell(**kwargs: Any) -> dict[str, Any]:
    """Train one cross-architecture cell under the H0 override."""

    for name, expected in (
        ("precision", legacy.FROZEN_PRECISION),
        ("batch_size", legacy.FROZEN_BATCH_SIZE),
        ("num_workers", legacy.FROZEN_NUM_WORKERS),
        ("epochs", legacy.FROZEN_EPOCHS),
        ("calibration_workers", legacy.FROZEN_CALIBRATION_WORKERS),
    ):
        if kwargs.get(name, expected) != expected:
            raise ValueError(f"H0 cross-architecture {name} is frozen at {expected}")
    learning_rate = float(kwargs.get("learning_rate", legacy.FROZEN_LEARNING_RATE))
    if learning_rate != legacy.FROZEN_LEARNING_RATE:
        raise ValueError("H0 cross-architecture learning rate is frozen at 3e-4")

    output_dir = Path(kwargs["output_dir"]).resolve()
    with _h0_context():
        result = legacy.train_cell(**kwargs)
    # A completed legacy result at this path is accepted only if its resolved
    # config independently proves H0.  H1 results therefore fail closed.
    return _stamp_run(output_dir)


def validate_run(
    output_dir: Path, *, head: str, representation: str, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    with _h0_context():
        summary, resolved = legacy.validate_run(
            output_dir, head=head, representation=representation, seed=seed
        )
    _assert_h0_resolved(Path(str(summary["resolved_config"])))
    current_source_tree_sha256 = _source_tree_sha256(_source_records())
    expected_wrapper_sha256 = _sha256(Path(__file__).resolve())
    complete = _read_json(output_dir / "COMPLETE.json")
    h0_checks = {
        "summary_h0_contract": summary.get("h0_contract") == h0_contract(),
        "summary_hand_weight": summary.get("hand_weight") == H0_HAND_WEIGHT,
        "summary_hand_loss_disabled": (
            summary.get("hand_loss_enabled") is H0_HAND_LOSS_ENABLED
        ),
        "summary_wrapper_hash": (
            summary.get("h0_wrapper_sha256") == expected_wrapper_sha256
        ),
        "summary_source_tree_hash": (
            summary.get("h0_source_tree_sha256") == current_source_tree_sha256
        ),
        "complete_h0_contract": complete.get("h0_contract") == h0_contract(),
        "complete_hand_weight": complete.get("hand_weight") == H0_HAND_WEIGHT,
        "complete_hand_loss_disabled": (
            complete.get("hand_loss_enabled") is H0_HAND_LOSS_ENABLED
        ),
        "complete_wrapper_hash": (
            complete.get("h0_wrapper_sha256") == expected_wrapper_sha256
        ),
        "complete_source_tree_hash": (
            complete.get("h0_source_tree_sha256") == current_source_tree_sha256
        ),
        "test_sealed": (
            summary.get("test_manifest_read") is False
            and complete.get("test_manifest_read") is False
        ),
    }
    if not all(h0_checks.values()):
        raise ValueError(f"run is stale or not H0: {output_dir}: {h0_checks}")
    return summary, resolved


def aggregate_crossarch(*, runs_root: Path, model_gate: Path, output: Path) -> dict[str, Any]:
    """Aggregate only runs whose resolved configs prove the H0 contract."""

    runs_root = runs_root.resolve()
    gate = _read_json(model_gate)
    current_source_tree_sha256 = _source_tree_sha256(_source_records())
    gate_checks = {
        "h0_schema": gate.get("h0_release_schema") == H0_RELEASE_SCHEMA,
        "status": gate.get("status") == "pass",
        "test_sealed": gate.get("test_manifest_read") is False,
        "jobs": gate.get("jobs") == len(HEADS) * len(REPRESENTATIONS) * len(SEEDS),
        "legacy_hash": gate.get("trainer_sha256") == _sha256(Path(legacy.__file__).resolve()),
        "wrapper_hash": gate.get("h0_wrapper_sha256") == _sha256(Path(__file__).resolve()),
        "source_tree_hash": gate.get("h0_source_tree_sha256") == current_source_tree_sha256,
    }
    if not all(gate_checks.values()):
        raise ValueError(f"invalid or stale H0 model gate: {gate_checks}")
    for head in HEADS:
        for representation in REPRESENTATIONS:
            for seed in SEEDS:
                run = runs_root / head / representation / f"seed_{seed}"
                validate_run(
                    run, head=head, representation=representation, seed=seed
                )
    with _h0_context():
        result = legacy.aggregate_crossarch(
            runs_root=runs_root, model_gate=model_gate, output=output
        )
    value = _read_json(output)
    value.update(_h0_fields())
    value["h0_release_schema"] = H0_RELEASE_SCHEMA
    value["purpose"] = "formal_wave4_cross_architecture_h0_validation_only"
    controls = dict(value.get("controls") or {})
    controls["hand_loss_weight_zero"] = True
    controls["h0_wrapper_sha256"] = _sha256(Path(__file__).resolve())
    value["controls"] = controls
    _atomic_json(output, value)
    complete_path = output.parent / "WAVE4_CROSSARCH_COMPLETE.json"
    complete = _read_json(complete_path)
    complete.update(_h0_fields())
    complete["h0_release_schema"] = H0_RELEASE_SCHEMA
    complete["aggregate_sha256"] = _sha256(output)
    _atomic_json(complete_path, complete)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    release = commands.add_parser("audit-release")
    release.add_argument("--data-dir", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    validate_release_parser = commands.add_parser("validate-release")
    validate_release_parser.add_argument("--data-dir", type=Path, required=True)
    validate_release_parser.add_argument("--release", type=Path, required=True)
    gate = commands.add_parser("model-gate")
    gate.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--head", choices=HEADS, required=True)
    train.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    train.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--precision", default=legacy.FROZEN_PRECISION)
    train.add_argument("--batch-size", type=int, default=legacy.FROZEN_BATCH_SIZE)
    train.add_argument("--num-workers", type=int, default=legacy.FROZEN_NUM_WORKERS)
    train.add_argument("--epochs", type=int, default=legacy.FROZEN_EPOCHS)
    train.add_argument("--learning-rate", type=float, default=legacy.FROZEN_LEARNING_RATE)
    train.add_argument("--calibration-workers", type=int, default=legacy.FROZEN_CALIBRATION_WORKERS)
    validate = commands.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--head", choices=HEADS, required=True)
    validate.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    validate.add_argument("--seed", type=int, choices=SEEDS, required=True)
    repair = commands.add_parser(
        "repair",
        help=(
            "stamp an otherwise complete legacy-produced run with the current "
            "H0 metadata without touching checkpoints"
        ),
    )
    repair.add_argument("--output-dir", type=Path, required=True)
    repair.add_argument("--head", choices=HEADS, required=True)
    repair.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    repair.add_argument("--seed", type=int, choices=SEEDS, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--runs-root", type=Path, required=True)
    aggregate.add_argument("--model-gate", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "audit-release":
        print(json.dumps(audit_release(data_dir=args.data_dir, output=args.output), ensure_ascii=False, indent=2))
    elif args.command == "validate-release":
        _, checks = validate_release(release_path=args.release, data_dir=args.data_dir)
        print(json.dumps({"event": "crossarch_h0_release_valid", "checks": checks, "test_manifest_read": False}, ensure_ascii=False, indent=2))
    elif args.command == "model-gate":
        print(json.dumps(build_model_gate(output=args.output), ensure_ascii=False, indent=2))
    elif args.command == "train":
        train_cell(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            head=args.head,
            representation=args.representation,
            seed=args.seed,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            calibration_workers=args.calibration_workers,
        )
    elif args.command == "validate":
        summary, _ = validate_run(
            args.output_dir, head=args.head, representation=args.representation, seed=args.seed
        )
        print(json.dumps({"event": "wave4_crossarch_h0_run_valid", "variant": f"{args.head}_{args.representation}", "seed": args.seed, "test_manifest_read": False}))
    elif args.command == "repair":
        output_dir = args.output_dir.resolve()
        # Repair is metadata-only and fail-closed: both legacy completion
        # files must already exist, and validation runs immediately after the
        # H0 stamp.  Model/checkpoint bytes are never rewritten.
        if not (output_dir / "summary.json").is_file() or not (
            output_dir / "COMPLETE.json"
        ).is_file():
            raise ValueError(
                f"repair requires an existing complete run (summary + COMPLETE): {output_dir}"
            )
        _stamp_run(output_dir)
        validate_run(
            output_dir,
            head=args.head,
            representation=args.representation,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "event": "wave4_crossarch_h0_run_repaired",
                    "variant": f"{args.head}_{args.representation}",
                    "seed": args.seed,
                    "output_dir": str(output_dir),
                    "metadata_only": True,
                    "test_manifest_read": False,
                }
            )
        )
    elif args.command == "aggregate":
        aggregate_crossarch(runs_root=args.runs_root, model_gate=args.model_gate, output=args.output)


if __name__ == "__main__":
    main()
