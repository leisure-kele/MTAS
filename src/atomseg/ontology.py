from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .artifacts import sha256_file, sha256_json


FORBIDDEN_20H_PATH_TOKENS = (
    "atomic_v0.9",
    "atomic_v0_9",
    "future500h",
    "migration_legacy20h",
)


@dataclass(frozen=True)
class FrozenOntology:
    path: Path
    sha256: str
    version: str
    actions: tuple[str, ...]
    ignore_index: int
    raw: Mapping[str, Any]
    semantic_sha256: str

    @property
    def index_by_name(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.actions)}


def ontology_semantic_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return the corpus-independent identity of the legacy20h ontology.

    The historical YAML also records observations from one delivery (paths,
    segment/duration counts and boundary counts).  Those values are useful
    provenance, but they must not make a new annotation release look like a new
    21-class semantic ontology.  Keep only fields that define label meaning,
    normalization, output heads and boundary semantics.
    """

    source_library = raw.get("source_library", {})
    output_head = raw.get("output_head", {})
    boundaries = raw.get("boundaries", {})
    action_rows = raw.get("actions", [])
    return {
        "schema_version": raw.get("schema_version"),
        "ontology_version": raw.get("ontology_version"),
        "separation_policy": raw.get("separation_policy"),
        "source_library": {
            key: source_library.get(key)
            for key in (
                "declared_version",
                "button_count",
                "hand_labels_populated",
                "label_field_populated",
                "supervision_field",
            )
        },
        "output_head": {
            key: output_head.get(key)
            for key in (
                "num_actions",
                "ignore_index",
                "idle_is_action",
                "hand_head_enabled",
            )
        },
        "actions": [
            {"index": row.get("index"), "name": row.get("name")}
            for row in action_rows
            if isinstance(row, Mapping)
        ],
        "unobserved_source_buttons": raw.get("unobserved_source_buttons"),
        "source_key_normalization": raw.get("source_key_normalization"),
        "reporting": raw.get("reporting"),
        "boundaries": {
            key: boundaries.get(key)
            for key in (
                "preserve_adjacent_same_label_instances",
                "boundary_types",
                "primary_evaluation_view",
                "literature_evaluation_view",
                "exclude_episode_edges",
                "exclude_tail_extension",
            )
        },
    }


def ontology_semantic_sha256(raw: Mapping[str, Any]) -> str:
    return sha256_json(ontology_semantic_payload(raw))


def _reject_forbidden_path(path: Path) -> None:
    lowered = str(path).lower()
    matches = [token for token in FORBIDDEN_20H_PATH_TOKENS if token in lowered]
    if matches:
        raise ValueError(
            "The legacy20h pipeline refuses future/migration ontology paths: "
            f"{path} (matched {matches})"
        )


def load_frozen_ontology(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> FrozenOntology:
    ontology_path = Path(path).expanduser().resolve()
    _reject_forbidden_path(ontology_path)
    with ontology_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Ontology must be a mapping: {ontology_path}")
    version = raw.get("ontology_version")
    if version != "legacy20h_v1":
        raise ValueError(
            f"Only legacy20h_v1 is valid for the 20h pipeline, got {version!r}"
        )
    separation = raw.get("separation_policy", {})
    if separation.get("migrate_to_atomic_v0_9") is not False:
        raise ValueError("legacy20h ontology must explicitly disable atomic_v0.9 migration")
    if separation.get("mix_with_later_500h") is not False:
        raise ValueError("legacy20h ontology must explicitly disable later-500h mixing")

    action_rows = raw.get("actions")
    if not isinstance(action_rows, list) or not action_rows:
        raise ValueError("Ontology actions must be a non-empty list")
    indices = [row.get("index") for row in action_rows]
    expected_indices = list(range(len(action_rows)))
    if indices != expected_indices:
        raise ValueError(
            f"Ontology action indices must be contiguous and ordered: {indices}"
        )
    actions = tuple(str(row["name"]) for row in action_rows)
    if len(actions) != 21 or len(set(actions)) != len(actions):
        raise ValueError("legacy20h_v1 must contain exactly 21 unique active actions")
    output_head = raw.get("output_head", {})
    if int(output_head.get("num_actions", -1)) != len(actions):
        raise ValueError("output_head.num_actions does not match actions")
    if bool(output_head.get("hand_head_enabled", True)):
        raise ValueError("legacy20h_v1 must not enable a hand action head")
    boundaries = raw.get("boundaries", {})
    if not bool(boundaries.get("preserve_adjacent_same_label_instances", False)):
        raise ValueError("legacy20h_v1 must preserve adjacent same-label instances")

    digest = sha256_file(ontology_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"Ontology SHA256 mismatch: expected {expected_sha256}, computed {digest}"
        )
    return FrozenOntology(
        path=ontology_path,
        sha256=digest,
        version=str(version),
        actions=actions,
        ignore_index=int(output_head.get("ignore_index", -100)),
        raw=raw,
        semantic_sha256=ontology_semantic_sha256(raw),
    )
