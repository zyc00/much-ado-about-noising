"""Frozen early HT checkpoint on the exact zero-model/20k-MSE probe inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from probe_widowx_general_scale import load_model, probe, select_episodes
from probe_zero_model_tail import analyze
from tail_stats_single import load, make_folds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--objective', choices=['hetero_t', 'mse'], default='hetero_t')
    p.add_argument('--training-steps', type=int, default=2000)
    p.add_argument('--data', type=Path, default=Path('/mnt/pfs/yuchen/groot/bridge_orig_lerobot'))
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--draw-batch', type=int, default=1)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    key = 'early_ht' if args.objective == 'hetero_t' else 'early_mse'
    output = args.output / f'{key}_rank0.npz'
    torch.set_num_threads(4)
    if not output.exists():
        torch.cuda.set_device(0)
        episodes, tasks = select_episodes(args.data, 48)
        model, processor = load_model(args.checkpoint, args.objective)
        # mode=mse requests one deterministic prediction; it does NOT change
        # the checkpoint's hetero_t model configuration or inference path.
        result = probe(model, processor, episodes, args, 'mse')
        result.pop('state', None)
        result.pop('embedding', None)
        meta = dict(checkpoint=str(args.checkpoint), objective=args.objective,
                    training_steps=args.training_steps, task_names=tasks,
                    residual='prediction minus label; no own-sigma division',
                    gripper_excluded=True, policy_split='training-demonstration diagnostic')
        np.savez_compressed(output, **result, metadata=json.dumps(meta))
    early = load(str(output))
    reference = load(str(args.reference / 'mse_rank*.npz'))
    for field in ('episode', 'task_id', 'step'):
        np.testing.assert_array_equal(early[field], reference[field])
    np.testing.assert_allclose(early['target'], reference['target'], atol=1e-7, rtol=0)
    fold = make_folds(early['task_id'], early['episode'])
    stats = analyze(early['residual'].astype(np.float64), fold)
    summary = dict(checkpoint=str(args.checkpoint), objective=args.objective,
                   training_steps=args.training_steps,
                   states=len(fold), episodes=len(np.unique(early['episode'])),
                   input_and_target_match='passed against 20k MSE probe',
                   protocol='Same five-fold fixed-coordinate RMS standardization as zero model; no own-sigma division',
                   **{key: stats})
    (args.output / 'early_checkpoint_tail_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
