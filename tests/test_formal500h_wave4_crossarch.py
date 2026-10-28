from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from atomseg import formal500h_wave1_train as wave1
from atomseg.formal500h_protocol import sha256_file
from atomseg.losses import ContactTASLoss
from atomseg import formal500h_wave4_crossarch as crossarch


class Formal500hWave4CrossArchTest(unittest.TestCase):
    def test_frozen_three_by_three_by_three_contract(self) -> None:
        self.assertEqual(crossarch.HEADS, ("MSTCNPP", "ASRF", "DIFFACT"))
        self.assertEqual(crossarch.REPRESENTATIONS, ("V", "VT", "M2R2"))
        self.assertEqual(
            crossarch.SEEDS, (20260824, 20260825, 20260826)
        )
        self.assertEqual(len(crossarch.CELL_SPECS), 9)
        self.assertEqual(len(crossarch.CELL_SPECS) * len(crossarch.SEEDS), 27)
        fingerprints = {
            crossarch.architecture_fingerprint(spec)
            for spec in crossarch.CELL_SPECS
        }
        self.assertEqual(len(fingerprints), 9)

    def test_representation_and_boundary_contracts_are_explicit(self) -> None:
        for head in crossarch.HEADS:
            v = crossarch.CELL_BY_ID[f"{head}_V"]
            vt = crossarch.CELL_BY_ID[f"{head}_VT"]
            m2r2 = crossarch.CELL_BY_ID[f"{head}_M2R2"]
            self.assertEqual(v.modalities, ("vision",))
            self.assertEqual(vt.modalities, ("vision", "tactile"))
            self.assertEqual(vt.as_dict()["reporting_label"], "D30B")
            contract = crossarch._architecture_contract(vt)
            self.assertEqual(
                contract["representation"]["tactile_recipe"],
                {
                    "pre_registered_id": "D30B",
                    "representation": "raw_delta_abs",
                    "event_rate_hz": 30,
                },
            )
            m2r2_contract = crossarch._architecture_contract(m2r2)
            self.assertFalse(m2r2_contract["official_reproduction"])
            self.assertEqual(
                m2r2_contract["representation"]["independent_modality_encoders"],
                ["vision", "tactile"],
            )
            self.assertEqual(
                set(m2r2_contract["representation"]["auxiliary_supervision"]),
                {
                    "action_order_symmetric_info_nce",
                    "gaussian_soft_boundary_mse",
                },
            )
            expected_boundary = head == "ASRF"
            self.assertIs(v.use_boundary_head, expected_boundary)
            self.assertIs(vt.use_boundary_head, expected_boundary)

    def test_all_models_forward_and_diffact_eval_does_not_read_labels(self) -> None:
        batch = crossarch._fixture_batch()
        for spec in crossarch.CELL_SPECS:
            wave1._seed_everything(123)
            model = crossarch.CrossArchFormal500hModel(
                crossarch._model_config(spec)
            ).eval()
            with torch.no_grad():
                output = model(crossarch._clone_batch(batch))
            self.assertEqual(output["atomic_logits"].shape, (2, 6, 26))
            self.assertEqual(output["boundary_logits"].shape, (2, 6))
            self.assertEqual(output["hand_logits"].shape, (2, 6, 3))
            self.assertTrue(torch.isfinite(output["atomic_logits"]).all())

        for representation in crossarch.REPRESENTATIONS:
            spec = crossarch.CELL_BY_ID[f"DIFFACT_{representation}"]
            model = crossarch.CrossArchFormal500hModel(
                crossarch._model_config(spec)
            ).eval()
            first = crossarch._fixture_batch(with_targets=True)
            second = crossarch._clone_batch(first)
            second["atomic_target"] = second["atomic_target"].flip(1)
            with torch.no_grad():
                first_output = model(first)
                second_output = model(second)
            self.assertEqual(
                crossarch._max_output_difference(first_output, second_output),
                0.0,
            )

    def test_m2r2_auxiliary_objectives_are_live(self) -> None:
        spec = crossarch.CELL_BY_ID["ASRF_M2R2"]
        model = crossarch.CrossArchFormal500hModel(
            crossarch._model_config(spec)
        ).train()
        batch = crossarch._fixture_batch(with_targets=True)
        output = model(batch)
        criterion = ContactTASLoss(
            {
                "action_weight": 1.0,
                "smooth_weight": 0.1,
                "boundary_weight": 0.5,
                "subtask_weight": 0.0,
                "boundary_positive_weight": 8.0,
            }
        )
        terms = crossarch.crossarch_loss_terms(
            criterion, output, batch, hand_weight=0.1
        )
        self.assertIn("order_embedding", output)
        self.assertIn("soft_boundary_logits", output)
        self.assertTrue(torch.isfinite(terms["loss"]))
        self.assertGreaterEqual(float(terms["action_order_contrastive"]), 0.0)
        self.assertGreater(float(terms["soft_boundary_mse"]), 0.0)
        terms["loss"].backward()
        self.assertIsNotNone(
            model.backbone.representation.order_projection.weight.grad
        )
        self.assertIsNotNone(
            model.backbone.representation.soft_boundary_head.weight.grad
        )

    def test_model_gate_passes_and_reuses_only_current_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gate.json"
            result = crossarch.build_model_gate(output=output)
            self.assertEqual(result["status"], "pass")
            self.assertTrue(all(result["checks"].values()))
            self.assertEqual(result["jobs"], 27)
            self.assertEqual(
                len(set(result["architecture_fingerprints"].values())), 9
            )
            self.assertEqual(crossarch.build_model_gate(output=output), result)

    def test_training_wrapper_applies_and_restores_frozen_contract(self) -> None:
        captured: dict = {}
        upstream = {
            "release_manifest": "/tmp/upstream",
            "release_manifest_sha256": "upstream",
        }
        base_loss = {
            "action_weight": 1.0,
            "smooth_weight": 0.1,
            "boundary_weight": 0.5,
            "subtask_weight": 0.0,
        }

        def fake_train(**kwargs):
            captured.update(kwargs)
            self.assertEqual(
                wave1.VARIANT_MODALITIES["DIFFACT_M2R2"],
                ("vision", "tactile"),
            )
            config = wave1._model_config(
                "DIFFACT_M2R2", d_model=128, dropout=0.15
            )
            self.assertEqual(config["head"], "DIFFACT")
            self.assertEqual(config["representation"], "M2R2")
            loss = wave1._loss_config(Path("statistics.npz"))
            self.assertEqual(loss["boundary_weight"], 0.0)
            self.assertEqual(
                loss["action_order_contrastive_weight"],
                crossarch.ACTION_ORDER_CONTRASTIVE_WEIGHT,
            )
            self.assertIs(wave1.Formal500hModel, crossarch.CrossArchFormal500hModel)
            self.assertIs(wave1.formal500h_loss_terms, crossarch.crossarch_loss_terms)
            self.assertIs(wave1._evaluate_predictions, crossarch._evaluate_predictions)
            self.assertEqual(wave1.TRAINING_SCHEMA, crossarch.TRAINING_SCHEMA)
            return {"status": "complete"}

        with mock.patch.object(
            wave1, "train_formal_variant", side_effect=fake_train
        ), mock.patch.object(
            wave1, "_implementation_release", return_value=upstream
        ) as release, mock.patch.object(
            wave1, "_loss_config", return_value=base_loss
        ) as loss:
            originals = (
                wave1.VARIANT_MODALITIES,
                wave1._dataset,
                wave1.collate_formal500h,
                wave1._implementation_release,
                wave1._model_config,
                wave1._loss_config,
                wave1.formal500h_loss_terms,
                wave1.Formal500hModel,
                wave1._evaluate_predictions,
                wave1.TRAINING_SCHEMA,
            )
            result = crossarch.train_cell(
                data_dir=Path("data"),
                output_dir=Path("output"),
                head="DIFFACT",
                representation="M2R2",
                seed=crossarch.SEEDS[0],
            )
            self.assertEqual(result, {"status": "complete"})
            self.assertEqual(
                (
                    wave1.VARIANT_MODALITIES,
                    wave1._dataset,
                    wave1.collate_formal500h,
                    wave1._implementation_release,
                    wave1._model_config,
                    wave1._loss_config,
                    wave1.formal500h_loss_terms,
                    wave1.Formal500hModel,
                    wave1._evaluate_predictions,
                    wave1.TRAINING_SCHEMA,
                ),
                originals,
            )
            self.assertIs(wave1._implementation_release, release)
            self.assertIs(wave1._loss_config, loss)
        self.assertEqual(captured["batch_size"], 16)
        self.assertEqual(captured["num_workers"], 20)
        self.assertEqual(captured["epochs"], 3)

    def test_common_evaluator_switches_only_head_boundary_mechanism(self) -> None:
        fake = mock.Mock()
        fake.metrics = {"primary": {"checkpoint_composite": 1.0}}
        for head, expected in (("ASRF", True), ("MSTCNPP", False)):
            spec = crossarch.CELL_BY_ID[f"{head}_VT"]
            with mock.patch.object(
                crossarch, "evaluate_dataset", return_value=fake
            ) as evaluate, mock.patch.object(crossarch, "_ACTIVE_SPEC", spec):
                metrics = crossarch._evaluate_predictions(
                    {}, {},
                    boundary_threshold=0.5,
                    boundary_min_distance_frames=6,
                    min_segment_frames=9,
                )
            self.assertIs(evaluate.call_args.kwargs["use_boundary_head"], expected)
            self.assertIs(evaluate.call_args.kwargs["include_action_changes"], True)
            self.assertEqual(
                evaluate.call_args.kwargs["boundary_tolerances_ms"],
                (50, 100, 250, 500, 1000),
            )
            self.assertFalse(
                metrics["wave4_common_metric_contract"][
                    "m2r2_style_adapted_dr_official_reproduction"
                ]
            )

    @staticmethod
    def _metric_summary(value: float, *, parameter_count: int, initial: str) -> dict:
        boundary = {
            tolerance: {"f1": value}
            for tolerance in ("250ms", "500ms", "1000ms")
        }
        return {
            "updates": 100,
            "parameter_count": parameter_count,
            "initial_model_sha256": initial,
            "data_summary_sha256": "data",
            "source_protocol_sha256": "protocol",
            "source_audit_sha256": "audit",
            "release_manifest_sha256": sha256_file(Path(crossarch.__file__)),
            "calibrated_validation": {
                "selected": {
                    "boundary_threshold": 0.5,
                    "boundary_min_distance_frames": 6,
                    "min_segment_frames": 9,
                },
                "metrics": {
                    "primary": {"checkpoint_composite": value},
                    "collapsed": {
                        "f1@10": value,
                        "f1@25": value,
                        "f1@50": value,
                        "edit": value,
                    },
                    "boundary": {
                        "class_change": boundary,
                        "all": {"1000ms": {"f1": value}},
                    },
                    "frame_accuracy": value,
                    "tick_macro_f1": value,
                },
                "hand": {"macro_f1": value},
            },
        }

    def test_aggregate_uses_preregistered_contrast_directions(self) -> None:
        summaries: dict[tuple[str, str, int], tuple[dict, dict]] = {}
        counts = {"MSTCNPP": 100, "ASRF": 90, "DIFFACT": 80}
        for head_index, head in enumerate(crossarch.HEADS):
            for representation_index, representation in enumerate(
                crossarch.REPRESENTATIONS
            ):
                spec = crossarch.CELL_BY_ID[f"{head}_{representation}"]
                count = counts[head] + (20 if representation == "M2R2" else 0)
                for seed_index, seed in enumerate(crossarch.SEEDS):
                    value = float(head_index * 10 + representation_index + seed_index)
                    initial = (
                        f"{head}-{seed}" if representation != "M2R2"
                        else f"{head}-m2r2-{seed}"
                    )
                    summaries[(head, representation, seed)] = (
                        self._metric_summary(
                            value, parameter_count=count, initial=initial
                        ),
                        {
                            "model": {
                                "architecture_fingerprint_sha256": (
                                    crossarch.architecture_fingerprint(spec)
                                )
                            }
                        },
                    )

        def fake_validate(_root, *, head, representation, seed):
            return summaries[(head, representation, seed)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate_path = root / "model_gate.json"
            gate_path.write_text(
                json.dumps(
                    {
                        "schema_version": crossarch.MODEL_GATE_SCHEMA,
                        "status": "pass",
                        "test_manifest_read": False,
                        "jobs": 27,
                        "trainer_sha256": sha256_file(Path(crossarch.__file__)),
                        "architecture_fingerprints": {
                            spec.cell: crossarch.architecture_fingerprint(spec)
                            for spec in crossarch.CELL_SPECS
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "aggregate.json"
            with mock.patch.object(
                crossarch, "validate_run", side_effect=fake_validate
            ):
                result = crossarch.aggregate_crossarch(
                    runs_root=root / "runs",
                    model_gate=gate_path,
                    output=output,
                )
            for head in crossarch.HEADS:
                self.assertIn(f"{head}_VT_minus_V", result["paired"])
                self.assertIn(f"{head}_VT_minus_M2R2", result["paired"])
                self.assertNotIn(f"{head}_M2R2_minus_VT", result["paired"])
            self.assertEqual(
                result["representation_reporting_labels"]["VT"], "D30B"
            )
            self.assertEqual(result["jobs"], 27)
            self.assertFalse(result["test_manifest_read"])
            self.assertTrue((root / "WAVE4_CROSSARCH_COMPLETE.json").is_file())


if __name__ == "__main__":
    unittest.main()
