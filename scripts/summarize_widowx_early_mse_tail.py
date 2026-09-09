"""Aggregate measured early-MSE checkpoints with existing zero/20k probes."""
import json
from pathlib import Path

import numpy as np

from tail_stats_single import load


def main():
    root = Path('analysis/paper/longtail_motivation')
    baseline = json.loads((root/'zero_model_tail_summary.json').read_text())
    reference = load('analysis/paper/widowx_heterogeneous_scale/raw/mse_rank*.npz')
    cases = {'zero_model': baseline['zero_model']}
    for step in (50, 100, 200, 250, 500):
        folder = root/'mse_early_checkpoints'/str(step) if step != 500 else root/'mse500'
        measured = load(str(folder/'early_mse_rank0.npz'))
        for field in ('episode', 'task_id', 'step', 'target'):
            np.testing.assert_allclose(measured[field], reference[field], atol=1e-7, rtol=0)
        stats = json.loads((folder/'early_checkpoint_tail_summary.json').read_text())
        assert stats['objective'] == 'mse' and stats['training_steps'] == step
        cases[f'mse_{step}'] = stats['early_mse']
    cases['mse_20000'] = baseline['trained_mse']
    report = {
        'protocol': 'Identical 1728 inputs; six continuous channels; other-fold coordinate RMS; pooled scalar fits',
        'training': 'Fresh replay of original MSE recipe, original 20k LR schedule and 1000-step warmup',
        'cases': cases,
    }
    (root/'early_mse_checkpoint_summary.json').write_text(json.dumps(report, indent=2))
    print('case,tailfit_nu,mle_nu,p_gt3_percent,tailfit_upper_boundary')
    for name, s in cases.items():
        print(f"{name},{s['tailfit_nu']:.6f},{s['mle_nu']:.6f},{100*s['p_gt3']:.6f},{s['tailfit_hits_upper_nu_bound']}")


if __name__ == '__main__':
    main()
