from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from atomseg.formal500h_protocol import (
    _repair_duration_assignment,
    freeze_formal_protocol,
)
from atomseg.formal500h_training_release import RELEASE_SCHEMA


def _write_episode(
    path: Path,
    *,
    index: int,
    mcap_path: str,
    correction: str,
    labels: list[int],
) -> int:
    segment_ticks = 60
    ticks = segment_ticks * len(labels)
    starts = np.arange(len(labels), dtype=np.int32) * segment_ticks
    ends = starts + segment_ticks
    action = np.repeat(np.asarray(labels, dtype=np.int16), segment_ticks)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "formal500h_training_episode_v3"
        handle.attrs["complete"] = True
        handle.attrs["metadata_json"] = json.dumps(
            {
                "episode_id": f"episode_{index:03d}",
                "task_id": f"T{index % 4:03d}",
                "source_mcap_path": mcap_path,
                "tactile_hand_correction": correction,
                "tactile_swap_left_right": correction == "swap_left_right",
                "tactile_hand_order": ["left", "right"],
            }
        )
        handle.create_dataset("targets/action_class", data=action)
        group = handle.require_group("annotations/atomic_segments")
        group.create_dataset("start_frame_60", data=starts)
        group.create_dataset("end_frame_60_exclusive", data=ends)
        group.create_dataset("class_index", data=np.asarray(labels, dtype=np.int16))
    return ticks


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class Formal500hProtocolTest(unittest.TestCase):
    def test_duration_repair_preserves_hard_constraints_and_original_tolerance(
        self,
    ) -> None:
        assignment = np.asarray([0] * 5 + [1] * 3 + [2] * 2, dtype=np.int8)
        duration = np.ones(10, dtype=np.float64)
        fractions = np.asarray([0.70, 0.15, 0.15], dtype=np.float64)

        def hard_valid(value: np.ndarray) -> bool:
            return bool(np.all(np.bincount(value, minlength=3) >= 1))

        def score(value: np.ndarray) -> float:
            achieved = np.bincount(
                value, weights=duration, minlength=3
            ) / duration.sum()
            return float(np.square(achieved - fractions).sum())

        repaired, report = _repair_duration_assignment(
            assignment,
            duration=duration,
            fractions=fractions,
            tolerance=0.11,
            score=score,
            hard_valid=hard_valid,
        )
        self.assertFalse(report["before"]["passes"])
        self.assertTrue(report["after"]["passes"])
        self.assertTrue(hard_valid(repaired))
        self.assertEqual(report["after"]["tolerance"], 0.11)
        self.assertGreaterEqual(report["accepted_moves"], 1)

    def test_freezes_disjoint_complete_reproducible_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "trainable.jsonl"
            rows = []
            for group_index in range(20):
                for episode_offset in range(2):
                    index = 2 * group_index + episode_offset
                    capture = f"episode_{index:03d}"
                    mcap = f"/data/TASK_T001_A{group_index:02d}ABC/{capture}.mcap"
                    cache = root / f"{capture}.h5"
                    correction = "swap_left_right" if index % 3 == 0 else "keep"
                    labels = [0, 1 + group_index % 3, 4, 0]
                    if group_index == 0:
                        # A singleton-group class must remain in train while group
                        # durations stay equal for the ratio assertion.
                        labels[-1] = 5
                    ticks = _write_episode(
                        cache,
                        index=index,
                        mcap_path=mcap,
                        correction=correction,
                        labels=labels,
                    )
                    rows.append(
                        {
                            "capture_key": capture,
                            "episode_id": capture,
                            "task_id": f"T{index % 4:03d}",
                            "mcap_path": mcap,
                            "cache_path": str(cache),
                            "duration_s": ticks / 60.0,
                            "annotation_frame_count": ticks,
                            "split_group_id": f"collection_code:G{group_index:02d}",
                            "split_group_basis": "collection_code",
                            "training_release_schema": RELEASE_SCHEMA,
                            "tactile_qc_keep": True,
                            "hdf5_status": "written",
                            "tactile_swap_left_right": correction == "swap_left_right",
                            "tactile_hand_correction": correction,
                        }
                    )
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            first = root / "first"
            second = root / "second"
            freeze_formal_protocol(
                training_manifest=manifest,
                output_dir=first,
                workers=1,
                restarts=4,
                iterations=2000,
                minimum_groups_per_split=2,
            )
            freeze_formal_protocol(
                training_manifest=manifest,
                output_dir=second,
                workers=1,
                restarts=4,
                iterations=2000,
                minimum_groups_per_split=2,
            )
            assignment_first = {
                row["split_group_id"]: row["split"]
                for row in _read_jsonl(first / "group_assignments.jsonl")
            }
            assignment_second = {
                row["split_group_id"]: row["split"]
                for row in _read_jsonl(second / "group_assignments.jsonl")
            }
            self.assertEqual(assignment_first, assignment_second)
            self.assertEqual(len(assignment_first), 20)
            split_groups = {
                name: {
                    row["formal_split_group_id"]
                    for row in _read_jsonl(first / f"{name}.jsonl")
                }
                for name in ("train", "val", "test")
            }
            self.assertFalse(split_groups["train"] & split_groups["val"])
            self.assertFalse(split_groups["train"] & split_groups["test"])
            self.assertFalse(split_groups["val"] & split_groups["test"])
            self.assertEqual(
                sum(len(_read_jsonl(first / f"{name}.jsonl")) for name in split_groups),
                len(rows),
            )
            self.assertIn("collection_code:G00", split_groups["train"])
            audit = json.loads((first / "audit.json").read_text())
            self.assertEqual(audit["status"], "pass")
            self.assertTrue(all(audit["checks"].values()))
            self.assertIn(5, audit["class_policy"]["not_three_way_coverable_classes"])
            self.assertTrue(audit["checks"]["all_source_tasks_in_train"])
            self.assertEqual(
                audit["evaluation_policy"]["wave0_diagnostic"]["output_rate_hz"],
                30,
            )
            self.assertEqual(
                audit["evaluation_policy"]["formal_reportable"][
                    "primary_boundary_tolerance_ms"
                ],
                250,
            )
            protocol = json.loads((first / "protocol.json").read_text())
            self.assertIn("full_episode", protocol["evaluation_policy"]["formal_reportable"]["evaluation_unit"])

    def test_refuses_to_overwrite_frozen_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "protocol"
            output.mkdir()
            (output / "protocol.json").write_text("{}")
            manifest = root / "empty.jsonl"
            manifest.write_text("")
            with self.assertRaises(ValueError):
                freeze_formal_protocol(
                    training_manifest=manifest,
                    output_dir=output,
                    workers=1,
                    restarts=1,
                    iterations=1,
                )


if __name__ == "__main__":
    unittest.main()
