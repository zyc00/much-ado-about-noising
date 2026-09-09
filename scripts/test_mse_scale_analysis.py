"""Small deterministic checks; run with the repository's Python environment."""
import json
import unittest
import numpy as np
from probe_mse_stage_scale import sample_steps
from analyze_mse_stage_scale import metric_arrays,task_matrices,replication,available_mean,json_safe


class ScaleAnalysisTest(unittest.TestCase):
    def test_missing_terminal_stage_is_not_imputed(self):
        values=np.array([[1.,np.nan],[3.,np.nan]])
        mean=available_mean(values,axis=0)
        self.assertEqual(mean[0],2.)
        self.assertTrue(np.isnan(mean[1]))
        self.assertEqual(json.dumps(json_safe(mean.tolist()),allow_nan=False),'[2.0, null]')

    def test_sampling_keeps_complete_execution(self):
        rng=np.random.default_rng(1)
        for length in (50,80,120,500):
            for step,stage in sample_steps(length,1,length-8,2,rng):
                self.assertTrue(1<=step<=length-8)
                self.assertEqual(int(step*10/length),stage)

    def test_channel_exclusion_and_time_alignment(self):
        pred=np.ones((3,16,10));pred[:,:,9]=1000;pred[:,0,:]=1000
        z=dict(metadata=np.array(json.dumps(dict(executed_start=1,executed_horizon=8,
            continuous_channels=list(range(9))))),gt=np.zeros_like(pred),pred_final=pred,
            valid_time=np.ones((3,16),bool))
        r=metric_arrays(z,'pred_final')
        self.assertEqual(r.shape,(3,72));np.testing.assert_array_equal(r,1)
        z['valid_time'][0,8]=False
        with self.assertRaises(AssertionError):metric_arrays(z,'pred_final')

    def test_replication_null_and_positive(self):
        rng=np.random.default_rng(1);n=4000
        stage=np.tile(np.arange(10),n//10);episode=np.repeat(np.arange(n//10),10)
        energy=np.mean(rng.normal(size=(n,48))**2,axis=1)
        z=dict(stage=stage,episode=episode,task=np.array(['task']*n),split=episode%2)
        null=replication(task_matrices(z,energy))
        self.assertLess(abs(np.mean([v['high_low_ratio'] for v in null])-1),.05)
        positive=replication(task_matrices(z,energy*(1+stage/3)**2))
        self.assertGreater(min(v['high_low_ratio'] for v in positive),2)


if __name__=='__main__':unittest.main()
