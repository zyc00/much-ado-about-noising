#!/usr/bin/env python3
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.plot_crossdataset_stage_scale import load_mse_residuals


ROOT = Path("analysis/paper/data_ht_motivation/cross_dataset")


class CrossDatasetStageScaleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((ROOT / "cross_dataset_scale_summary.json").read_text())

    def test_expected_scope(self):
        datasets = self.summary["datasets"]
        self.assertEqual(len(datasets), 5)
        self.assertEqual(sum(item["tasks"] for item in datasets), 149)
        self.assertTrue(self.summary["gripper_excluded"])
        self.assertEqual(self.summary["horizon"], 8)

    def test_reference_radius_predicts_heldout_spread(self):
        for dataset in self.summary["datasets"]:
            self.assertGreater(dataset["q5_over_q1_heldout_rms"], 1.0)
            self.assertGreater(dataset["task_bootstrap_95_ci"][0], 1.0)
            self.assertLessEqual(dataset["within_task_stage_permutation_p"], 1 / 4001)

    def test_lerobot_folds_are_task_balanced(self):
        for file_name in ("bridge_stage_probe.npz", "fractal_stage_probe.npz"):
            archive = np.load(ROOT / file_name)
            self.assertEqual(archive["gt"].shape[1:], (8, 6))
            for task in np.unique(archive["task"]):
                counts = [
                    len(np.unique(archive["episode"][(archive["task"] == task) & (archive["split"] == fold)]))
                    for fold in (0, 1)
                ]
                self.assertEqual(counts, [12, 12])

    def test_gr1_state_level_span_matches_sigma_audit(self):
        mse = load_mse_residuals(Path("analysis/paper/mse_scale/gr1.npz"))
        self.assertAlmostEqual(mse["central_ratio"], 4.62, places=2)
        self.assertAlmostEqual(mse["full_ratio"], 50.79, places=2)
        ht = np.load("analysis/paper/mse_scale/ht_gr1_60k_raw.npz")
        sigma = ht["sigma"].astype(float)
        self.assertAlmostEqual(float(sigma.max() / sigma.min()), 55.25, places=2)
        self.assertAlmostEqual(
            float(np.quantile(sigma, 0.9) / np.quantile(sigma, 0.1)), 9.35, places=2
        )


if __name__ == "__main__":
    unittest.main()
