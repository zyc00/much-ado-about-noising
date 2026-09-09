"""Tests for the figure's normalization and reconstructed HT measurements."""
from pathlib import Path
import unittest
import numpy as np
from plot_multiscale_rms import task_normalized_rms,weighted_quantiles,load_ht


class MultiscaleFigureTest(unittest.TestCase):
    def test_task_wide_scale_is_removed(self):
        r,w=task_normalized_rms([1,2,4,10,20,40],["a"]*3+["b"]*3)
        np.testing.assert_allclose(r[:3],r[3:])
        self.assertAlmostEqual(w.sum(),1.)

    def test_equal_task_weights_ignore_duplicate_observations(self):
        r,w=task_normalized_rms([1,2,4,10,20,40],["a"]*3+["b"]*3)
        rr,ww=task_normalized_rms([1,2,4,10,10,20,20,40,40],["a"]*3+["b"]*6)
        np.testing.assert_allclose(weighted_quantiles(r,w,[.1,.5,.9]),
                                   weighted_quantiles(rr,ww,[.1,.5,.9]))

    def test_nonpositive_values_fail(self):
        with self.assertRaises(AssertionError):
            task_normalized_rms([0,1],["a"]*2)

    def test_saved_ht_reconstruction_and_group_coverage(self):
        path=Path(__file__).resolve().parents[1]/"analysis/paper/mse_scale/ht_gr1_60k_raw.npz"
        if not path.exists():self.skipTest("Cluster probe archive not present")
        h=load_ht(path)  # Also reconstructs RMS and sigma from full stored tensors.
        self.assertEqual(sum(g["n"] for g in h["binned"]),2400)
        self.assertAlmostEqual(h["summary"]["sigma_rms_spearman"],.9077124726530822)


if __name__=="__main__":unittest.main()
