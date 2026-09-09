#!/usr/bin/env python3
"""Regression tests for the data-side HT motivation probe."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from scripts.probe_data_ht_motivation import events, phase_ranges


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/paper/data_ht_motivation"


class DataHTMotivationTest(unittest.TestCase):
    def test_event_and_phase_windows_are_ordered(self):
        command = -np.ones(750)
        command[70:380] = 1
        command[500:670] = 1
        self.assertEqual(events(command), (70, 380, 500, 670))
        ranges = phase_ranges(command)
        self.assertEqual(ranges, ((80, 335), (340, 372), (510, 615), (620, 662)))
        self.assertTrue(all(ranges[i][1] <= ranges[i+1][0] for i in range(3)))

    def test_saved_probe_invariants(self):
        summary = json.loads((OUT / "summary.json").read_text())
        data = np.load(OUT / "probe.npz")
        self.assertTrue(summary["gripper_excluded"])
        self.assertEqual(summary["continuous_action_channels"], list(range(6)))
        self.assertEqual(data["residual"].shape, (summary["query_n"], 48))
        self.assertTrue(np.isfinite(data["residual"]).all())
        self.assertTrue((data["local_scale"] > 0).all())

    def test_phase_and_tail_signals(self):
        summary = json.loads((OUT / "summary.json").read_text())
        for comparison in summary["paired_phase_summary"]:
            self.assertLess(comparison["median_paired_ratio"], .9)
            self.assertLess(comparison["bootstrap_95_ci"][1], 1)
        mixture = summary["phase_gaussian_mixture_fit"]
        self.assertTrue(all(f["precision_over_transit_sigma"] < .8 for f in mixture))
        for calibration in summary["local_scale_calibration"]:
            rms = [g["heldout_residual_rms"] for g in calibration["groups"]]
            self.assertTrue(all(a < b for a, b in zip(rms, rms[1:])))
            self.assertGreater(calibration["high_over_low_rms"], 1.8)
            self.assertGreater(calibration["high_over_low_bootstrap_95_ci"][0], 1.7)
            self.assertLess(calibration["within_phase_permutation_p"], .001)
            self.assertGreater(calibration["high_over_low_rms"],
                               calibration["within_phase_null_ratio_q95_q99"][1])
        controlled = summary["coordinate_controlled_student_fit"]
        self.assertTrue(all(f["heldout_student_nll_gain"] > 0 for f in controlled))
        unclipped = summary["unclipped_coordinate_controlled_student_fit"]
        self.assertTrue(all(f["heldout_student_nll_gain"] > .02 for f in unclipped))
        self.assertTrue(all(f["retained_fraction_test"] > .98 for f in unclipped))
        tail3 = summary["tail_plot"]["threshold_survival"]["3"]
        self.assertGreater(tail3["empirical_over_gaussian"], 3)

    def test_neighbor_count_sensitivity_preserves_ordering(self):
        for suffix in ("k16", "k64"):
            path = ROOT / f"analysis/paper/data_ht_motivation_{suffix}/summary.json"
            summary = json.loads(path.read_text())
            rms = [row["residual_rms"] for row in summary["phase_summary"]]
            self.assertLess(rms[1], rms[0])
            self.assertLess(rms[3], rms[2])


if __name__ == "__main__":
    unittest.main()
