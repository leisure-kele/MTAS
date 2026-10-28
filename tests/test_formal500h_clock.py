from __future__ import annotations

import unittest

import numpy as np

from atomseg.formal500h_clock import (
    CLOCK_POLICY,
    clock_metadata,
    derive_change_mask,
    downsample_event_mask,
    downsample_labels,
    downsample_targets_60_to_30,
)


class Formal500hClockTest(unittest.TestCase):
    def test_labels_use_first_sample_of_each_pair(self) -> None:
        values = np.arange(7)
        np.testing.assert_array_equal(downsample_labels(values), [0, 2, 4, 6])

    def test_even_event_maps_to_containing_output_tick(self) -> None:
        events = np.zeros(6, dtype=bool)
        events[2] = True
        np.testing.assert_array_equal(downsample_event_mask(events), [False, True, False])

    def test_odd_event_maps_to_following_tick(self) -> None:
        events = np.zeros(6, dtype=bool)
        events[1] = True
        np.testing.assert_array_equal(downsample_event_mask(events), [False, True, False])

    def test_multiple_events_in_one_bin_are_or_ed(self) -> None:
        events = np.zeros(6, dtype=bool)
        # Under ceiling mapping raw indices 1 and 2 both land in output bin 1.
        events[[1, 2]] = True
        np.testing.assert_array_equal(downsample_event_mask(events), [False, True, False])

    def test_final_odd_event_is_clipped_to_last_output_tick(self) -> None:
        """An event on the final source frame must never index past output."""

        events = np.zeros(6, dtype=bool)
        events[5] = True
        np.testing.assert_array_equal(downsample_event_mask(events), [False, False, True])

    def test_final_event_for_odd_source_length_is_in_last_tick(self) -> None:
        events = np.zeros(5, dtype=bool)
        events[4] = True
        np.testing.assert_array_equal(downsample_event_mask(events), [False, False, True])

    def test_derived_boundary_matches_sampled_instances(self) -> None:
        instances = np.array([10, 10, 20, 20, 30, 30])
        np.testing.assert_array_equal(
            derive_change_mask(downsample_labels(instances)), [False, True, True]
        )

    def test_target_bundle_is_consistent_for_odd_transition(self) -> None:
        actions = np.array([0, 0, 1, 1, 1, 2])
        instances = np.array([5, 5, 6, 6, 6, 7])
        result = downsample_targets_60_to_30(
            action_class_60=actions,
            action_instance_60=instances,
        )
        np.testing.assert_array_equal(result["action_class"], [0, 1, 1])
        np.testing.assert_array_equal(result["action_instance"], [5, 6, 6])
        np.testing.assert_array_equal(result["boundary_instance"], [False, True, False])
        np.testing.assert_array_equal(result["boundary_class_change"], [False, True, False])

    def test_metadata_is_explicit(self) -> None:
        self.assertEqual(clock_metadata()["policy"], CLOCK_POLICY)


if __name__ == "__main__":
    unittest.main()
