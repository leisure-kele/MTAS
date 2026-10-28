from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from atomseg import formal500h_asformer as asformer
from atomseg import formal500h_wave1_train as wave1


class Formal500hASFormerTest(unittest.TestCase):
    def test_padding_is_excluded_from_local_attention(self) -> None:
        block = asformer.LocalTransformerBlock(
            dimension=8,
            heads=2,
            local_radius=3,
            dropout=0.0,
            ffn_multiplier=2,
        ).eval()
        torch.manual_seed(123)
        valid = torch.randn(1, 4, 8)
        first = torch.cat([valid, torch.zeros(1, 2, 8)], dim=1)
        second = torch.cat([valid, torch.full((1, 2, 8), 1000.0)], dim=1)
        mask = torch.tensor([[True, True, True, True, False, False]])
        with torch.no_grad():
            first_out = block(first, mask)
            second_out = block(second, mask)
        torch.testing.assert_close(first_out[:, :4], second_out[:, :4])
        self.assertTrue(torch.equal(first_out[:, 4:], torch.zeros_like(first_out[:, 4:])))
        self.assertTrue(torch.equal(second_out[:, 4:], torch.zeros_like(second_out[:, 4:])))

    def test_frozen_matrix_and_recipe(self) -> None:
        self.assertEqual(asformer.VARIANTS, ("ASFORMER_V", "ASFORMER_VT"))
        self.assertEqual(asformer.SEEDS, (20260824, 20260825, 20260826))
        self.assertEqual(len(asformer.VARIANTS) * len(asformer.SEEDS), 6)
        config = asformer._model_config("ASFORMER_VT", d_model=128, dropout=0.15)
        self.assertEqual(config["head"], "ASFORMER")
        self.assertEqual(config["representation"], "VT")
        self.assertFalse(config["use_boundary_head"])
        self.assertFalse(config["hand_loss_enabled"])
        self.assertEqual(config["num_actions"], 26)

    def test_decoder_is_multistage_and_eval_does_not_need_targets(self) -> None:
        from atomseg.formal500h_wave4_crossarch import _clone_batch, _fixture_batch

        model = asformer.ASFormerFormal500hModel(
            asformer._model_config("ASFORMER_V", d_model=128, dropout=0.15)
        ).eval()
        with torch.no_grad():
            output = model(_clone_batch(_fixture_batch()))
        self.assertEqual(output["atomic_logits"].shape, (2, 6, 26))
        self.assertEqual(output["boundary_logits"].shape, (2, 6))
        self.assertEqual(output["hand_logits"].shape, (2, 6, 3))
        self.assertEqual(len(output["atomic_stages"]), 2)
        self.assertTrue(torch.isfinite(output["atomic_logits"]).all())
        self.assertTrue(torch.equal(output["boundary_logits"], torch.full_like(output["boundary_logits"], -20.0)))

    def test_v_and_vt_are_parameter_matched_but_tactile_sensitive(self) -> None:
        from atomseg.formal500h_wave4_crossarch import _clone_batch, _fixture_batch

        batch = _fixture_batch()
        for variant in asformer.VARIANTS:
            wave1._seed_everything(20260831)
            model = asformer.ASFormerFormal500hModel(
                asformer._model_config(variant, d_model=128, dropout=0.15)
            ).eval()
            with torch.no_grad():
                first = model(_clone_batch(batch))["atomic_logits"]
                changed = _clone_batch(batch)
                changed["tactile"] = changed["tactile"] + 7.0
                second = model(changed)["atomic_logits"]
            difference = float((first.float() - second.float()).abs().max())
            if variant.endswith("_V"):
                self.assertEqual(difference, 0.0)
            else:
                self.assertGreater(difference, 0.0)

    def test_model_gate_reuses_current_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model_gate.json"
            result = asformer.build_model_gate(output=output)
            self.assertEqual(result["status"], "pass")
            self.assertTrue(all(result["checks"].values()))
            self.assertEqual(asformer.build_model_gate(output=output), result)

    def test_train_wrapper_restores_wave1_globals(self) -> None:
        captured: dict[str, object] = {}

        def fake_train(**kwargs):
            captured.update(kwargs)
            self.assertIs(wave1.Formal500hModel, asformer.ASFormerFormal500hModel)
            self.assertEqual(wave1.VARIANT_MODALITIES["ASFORMER_VT"], ("vision", "tactile"))
            self.assertEqual(wave1._model_config("ASFORMER_VT", d_model=128, dropout=0.15)["head"], "ASFORMER")
            self.assertIs(wave1._loss_config, asformer._loss_config)
            return {"status": "complete"}

        originals = (
            wave1.VARIANT_MODALITIES,
            wave1._dataset,
            wave1.collate_formal500h,
            wave1._implementation_release,
            wave1._model_config,
            wave1._loss_config,
            wave1.Formal500hModel,
            wave1._ground_truth,
            wave1._evaluate_predictions,
            wave1.TRAINING_SCHEMA,
        )
        with mock.patch.object(wave1, "train_formal_variant", side_effect=fake_train), mock.patch.object(
            asformer, "_validated_release", return_value={"status": "complete"}
        ):
            result = asformer.train_cell(
                data_dir=Path("data"),
                release=Path("release.json"),
                output_dir=Path("output"),
                variant="ASFORMER_VT",
                seed=asformer.SEEDS[0],
            )
        self.assertEqual(result, {"status": "complete"})
        self.assertEqual((wave1.VARIANT_MODALITIES, wave1._dataset, wave1.collate_formal500h, wave1._implementation_release, wave1._model_config, wave1._loss_config, wave1.Formal500hModel, wave1._ground_truth, wave1._evaluate_predictions, wave1.TRAINING_SCHEMA), originals)
        self.assertEqual(captured["batch_size"], 16)
        self.assertEqual(captured["hand_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
