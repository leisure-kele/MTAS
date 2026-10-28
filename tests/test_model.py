from __future__ import annotations

import unittest

import torch

from atomseg.losses import ContactTASLoss
from atomseg.model import ContactTAS, ReliabilityFusion, SharedHandTactileEncoder


def make_batch(batch_size: int = 2, steps: int = 32, vision_steps: int = 8):
    index = torch.arange(steps).div(4, rounding_mode="floor").clamp_max(vision_steps - 1)
    index = index[None, :, None].repeat(batch_size, 1, 3)
    labels = torch.arange(steps).div(7, rounding_mode="floor").clamp_max(4)
    labels = labels[None].repeat(batch_size, 1)
    hard = torch.zeros(batch_size, steps, dtype=torch.bool)
    hard[:, 1:] = labels[:, 1:] != labels[:, :-1]
    return {
        "vision_features": torch.randn(batch_size, vision_steps, 3, 16),
        "vision_cache_valid": torch.ones(batch_size, vision_steps, 3, dtype=torch.bool),
        "vision_index_60": index,
        "vision_valid_60": torch.ones(batch_size, steps, 3, dtype=torch.bool),
        "vision_freshness_ms": torch.zeros(batch_size, steps, 3),
        "tactile": torch.randn(batch_size, steps, 2, 12),
        "tactile_valid": torch.ones(batch_size, steps, 2, dtype=torch.bool),
        "skeleton": torch.randn(batch_size, steps, 2, 5, 3),
        "skeleton_valid": torch.ones(batch_size, steps, 2, dtype=torch.bool),
        "kinematics": torch.randn(batch_size, steps, 8),
        "kinematics_valid": torch.ones(batch_size, steps, dtype=torch.bool),
        "atomic_target": labels,
        "subtask_target": (labels >= 2).long(),
        "boundary_target": hard.float(),
        "boundary_hard": hard,
        "loss_mask": torch.ones(batch_size, steps, dtype=torch.bool),
        "boundary_loss_mask": torch.ones(batch_size, steps, dtype=torch.bool),
        "sequence_mask": torch.ones(batch_size, steps, dtype=torch.bool),
    }


class ContactTASTest(unittest.TestCase):
    def setUp(self):
        self.model_config = {
            "vision_dim": 16,
            "tactile_dim": 12,
            "skeleton_joints": 5,
            "skeleton_joint_dim": 3,
            "kinematic_dim": 8,
            "num_actions": 5,
            "num_subtasks": 2,
            "d_model": 32,
            "stream_layers": 2,
            "decoder_channels": 24,
            "decoder_layers": 2,
            "decoder_stages": 2,
            "reliability_hidden": 16,
            "dropout": 0.0,
            "modality_dropout": 0.2,
        }

    def test_forward_loss_backward(self):
        model = ContactTAS(self.model_config)
        criterion = ContactTASLoss({"robust_weight": 0.1})
        batch = make_batch()
        output = model(batch)
        self.assertEqual(tuple(output["atomic_logits"].shape), (2, 32, 5))
        self.assertEqual(tuple(output["boundary_logits"].shape), (2, 32))
        losses = criterion.supervised(output, batch)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_all_modalities_missing_is_finite(self):
        model = ContactTAS(self.model_config).eval()
        batch = make_batch(batch_size=1)
        batch["vision_valid_60"].zero_()
        batch["tactile_valid"].zero_()
        batch["skeleton_valid"].zero_()
        batch["kinematics_valid"].zero_()
        with torch.no_grad():
            output = model(batch)
        self.assertTrue(torch.isfinite(output["atomic_logits"]).all())
        self.assertTrue(torch.isfinite(output["boundary_logits"]).all())

    def test_boundary_head_ablation_removes_branch_from_action_refinement(self):
        config = dict(self.model_config)
        config["use_boundary_head"] = False
        model = ContactTAS(config).eval()
        self.assertFalse(
            any("boundary_head" in name for name, _ in model.named_parameters())
        )
        batch = make_batch(batch_size=1)
        with torch.no_grad():
            output = model(batch)
        self.assertEqual(tuple(output["atomic_logits"].shape), (1, 32, 5))
        torch.testing.assert_close(
            output["boundary_logits"],
            torch.full_like(output["boundary_logits"], -20.0),
        )

    def test_tactile_encoder_consumes_raw_signed_and_absolute_delta(self):
        encoder = SharedHandTactileEncoder(
            tactile_dim=12, output_dim=8, dropout=0.0
        )
        first_layer = encoder.hand_encoder[0]
        self.assertEqual(first_layer.in_features, 36)

    def test_tactile_raw_control_is_parameter_matched_and_zeros_delta_blocks(self):
        derived = SharedHandTactileEncoder(
            tactile_dim=1,
            output_dim=3,
            dropout=0.0,
            representation="raw_delta_abs",
        )
        raw = SharedHandTactileEncoder(
            tactile_dim=1,
            output_dim=3,
            dropout=0.0,
            representation="raw",
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in derived.parameters()),
            sum(parameter.numel() for parameter in raw.parameters()),
        )
        derived.hand_encoder = torch.nn.Identity()
        raw.hand_encoder = torch.nn.Identity()
        with torch.no_grad():
            derived.hand_embedding.zero_()
            raw.hand_embedding.zero_()
        tactile = torch.tensor([[[[1.0], [2.0]], [[4.0], [8.0]]]])
        valid = torch.ones(1, 2, 2, dtype=torch.bool)
        derived_value = derived(tactile, valid).reshape(1, 2, 2, 3)
        raw_value = raw(tactile, valid).reshape(1, 2, 2, 3)
        torch.testing.assert_close(derived_value[0, 1, 0], torch.tensor([4.0, 3.0, 3.0]))
        torch.testing.assert_close(raw_value[0, 1, 0], torch.tensor([4.0, 0.0, 0.0]))

    def test_unknown_tactile_representation_fails(self):
        with self.assertRaisesRegex(ValueError, "tactile_representation"):
            SharedHandTactileEncoder(1, 3, 0.0, representation="topology")

    def test_tactile_delta_does_not_bridge_invalid_packets(self):
        encoder = SharedHandTactileEncoder(
            tactile_dim=1, output_dim=3, dropout=0.0
        )
        encoder.hand_encoder = torch.nn.Identity()
        with torch.no_grad():
            encoder.hand_embedding.zero_()
        tactile = torch.tensor(
            [[[[1.0], [1.0]], [[9.0], [2.0]], [[3.0], [4.0]]]]
        )
        valid = torch.tensor([[[True, True], [False, True], [True, True]]])
        encoded = encoder(tactile, valid).reshape(1, 3, 2, 3)
        torch.testing.assert_close(encoded[0, 2, 0], torch.tensor([3.0, 0.0, 0.0]))
        torch.testing.assert_close(encoded[0, 2, 1], torch.tensor([4.0, 2.0, 2.0]))

    def test_skeleton_delta_does_not_bridge_invalid_frames(self):
        sequence = torch.tensor([[[1.0, 2.0], [99.0, 99.0], [4.0, 8.0], [7.0, 9.0]]])
        quality = torch.tensor([[1.0, 0.0, 1.0, 0.5]])
        difference, pair_quality = ContactTAS._masked_temporal_difference(
            sequence, quality
        )
        torch.testing.assert_close(difference[0, 1], torch.zeros(2))
        torch.testing.assert_close(difference[0, 2], torch.zeros(2))
        torch.testing.assert_close(difference[0, 3], torch.tensor([1.5, 0.5]))
        torch.testing.assert_close(pair_quality, torch.tensor([[0.0, 0.0, 0.0, 0.5]]))

    def test_observable_quality_fields_drive_reliability_inputs(self):
        model = ContactTAS(self.model_config).eval()
        batch = make_batch(batch_size=1)
        steps = batch["sequence_mask"].shape[1]
        batch["tactile_quality"] = torch.zeros(1, steps, 2, 5)
        batch["tactile_quality"][..., 0] = 1.0
        batch["skeleton_joint_valid"] = torch.ones(
            1, steps, 2, 5, dtype=torch.bool
        )
        batch["pose_quality"] = torch.zeros(1, steps, 6)
        batch["pose_quality"][..., :4] = 1.0

        with torch.no_grad():
            good = model._encode_modalities(batch, apply_modality_dropout=False)
        torch.testing.assert_close(good["tactile_quality"], torch.ones(1, steps))
        torch.testing.assert_close(good["skeleton_quality"], torch.ones(1, steps))
        torch.testing.assert_close(good["kinematic_quality"], torch.ones(1, steps))

        # A zero reading is legal tactile data; zero_fraction alone is not a
        # missingness or failure signal.
        batch["tactile_quality"][..., 1] = 1.0
        with torch.no_grad():
            zero_heavy = model._encode_modalities(
                batch, apply_modality_dropout=False
            )
        torch.testing.assert_close(
            zero_heavy["tactile_quality"], torch.ones(1, steps)
        )

        batch["tactile_quality"][..., 1:] = 1.0
        batch["skeleton_joint_valid"][..., 2:] = False
        batch["pose_quality"][..., 4:] = 1.0
        with torch.no_grad():
            degraded = model._encode_modalities(batch, apply_modality_dropout=False)
        torch.testing.assert_close(
            degraded["tactile_quality"], torch.zeros(1, steps)
        )
        torch.testing.assert_close(
            degraded["skeleton_quality"], torch.full((1, steps), 0.4)
        )
        torch.testing.assert_close(
            degraded["kinematic_quality"], torch.zeros(1, steps)
        )

    def test_disabled_modalities_stay_disabled_during_modality_dropout(self):
        config = dict(self.model_config)
        config["enabled_modalities"] = ["vision"]
        model = ContactTAS(config).train()
        batch = make_batch(batch_size=4)

        masks = model._drop_modalities(batch, enabled=True)

        self.assertFalse(masks["tactile_valid"].any())
        self.assertFalse(masks["skeleton_valid"].any())
        self.assertFalse(masks["kinematics_valid"].any())

    def test_zero_quality_stream_cannot_leak_through_reliability_residual(self):
        torch.manual_seed(11)
        fusion = ReliabilityFusion(dimension=8, hidden=4).eval()
        event = torch.randn(2, 5, 8)
        semantic_a = torch.randn(2, 5, 8)
        semantic_b = semantic_a + 1000.0 * torch.randn(2, 5, 8)
        semantic_quality = torch.zeros(2, 5)
        event_quality = torch.ones(2, 5)

        with torch.no_grad():
            output_a, gate_a = fusion(
                semantic_a, event, semantic_quality, event_quality
            )
            output_b, gate_b = fusion(
                semantic_b, event, semantic_quality, event_quality
            )

        self.assertTrue(torch.equal(gate_a[..., 0], torch.zeros_like(gate_a[..., 0])))
        self.assertTrue(torch.equal(gate_b[..., 0], torch.zeros_like(gate_b[..., 0])))
        torch.testing.assert_close(gate_a, gate_b)
        torch.testing.assert_close(output_a, output_b)

        semantic = torch.randn(2, 5, 8)
        event_a = torch.randn(2, 5, 8)
        event_b = event_a + 1000.0 * torch.randn(2, 5, 8)
        semantic_quality = torch.ones(2, 5)
        event_quality = torch.zeros(2, 5)

        with torch.no_grad():
            output_a, gate_a = fusion(
                semantic, event_a, semantic_quality, event_quality
            )
            output_b, gate_b = fusion(
                semantic, event_b, semantic_quality, event_quality
            )

        self.assertTrue(torch.equal(gate_a[..., 1], torch.zeros_like(gate_a[..., 1])))
        self.assertTrue(torch.equal(gate_b[..., 1], torch.zeros_like(gate_b[..., 1])))
        torch.testing.assert_close(gate_a, gate_b)
        torch.testing.assert_close(output_a, output_b)


if __name__ == "__main__":
    unittest.main()
