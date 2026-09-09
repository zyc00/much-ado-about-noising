"""Replay three existing HT trajectories and measure the checkpoint's own sigma."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from gr00t.eval.sim.SimplerEnv.simpler_env import GoogleFractalEnv
from gr00t.policy.server_client import PolicyClient
from probe_fractal_handoff import snapshot, verify, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = GoogleFractalEnv('google_robot_close_drawer', image_size=(256, 320))
    client = PolicyClient(host='127.0.0.1', port=5567, timeout_ms=120000)
    static_steps = [0, 30, 60, 120, 210, 297]
    for i, seed in enumerate([1235, 1236, 1240]):
        with (args.source / 'collect' / f'{seed}.pkl').open('rb') as f:
            original = pickle.load(f)
        obs, _ = env.reset(seed=seed)
        client.reset({'seed': seed})
        frames, max_delta, checks = [], 0., []
        for step, action in enumerate(original['actions']):
            current = snapshot(env, obs)
            check = verify(original['states'][step], current)
            assert check['valid'], (seed, step, check)
            checks.append(check)
            if step in static_steps:
                frames.append(obs['video.image'].copy())
            prediction = predict(client, obs)  # Triggers native sigma dump, no label used.
            delta = max(float(np.max(np.abs(prediction[k]-action[k]))) for k in action)
            max_delta = max(max_delta, delta)
            assert delta < 1e-5, (seed, step, delta)
            obs, *_ = env.step(action)  # Exact recorded action, never drift to a new rollout.
            if (step+1) % 50 == 0:
                print(f'SIGMA_PROGRESS seed={seed} step={step+1} max_action_diff={max_delta}', flush=True)
        rows = [json.loads(line) for line in (args.output/'sigma_calls.jsonl').read_text().splitlines()]
        assert len(rows) == (i+1)*300, len(rows)
        selected = rows[i*300:(i+1)*300]
        assert [r['call'] for r in selected] == list(range(i*300,(i+1)*300))
        states = original['states']
        xyz = np.stack([s['xyz'] for s in states])
        command = np.stack([np.concatenate([a['action.'+k] for k in ('x','y','z')]) for a in original['actions']])
        np.savez_compressed(args.output/f'{seed}.npz', step=np.arange(300),
                            sigma=[r['sigma'] for r in selected],
                            sigma_per_dim=[r['per_dim'] for r in selected],
                            sigma_per_step=[r['per_step'] for r in selected],
                            drawer=[s['drawer'] for s in states[:300]],
                            eef_displacement=np.linalg.norm(np.diff(xyz,axis=0),axis=1),
                            command_norm=np.linalg.norm(command,axis=1),
                            static_frames=np.stack(frames), static_steps=static_steps)
        sigma = np.array([r['sigma'] for r in selected])
        meta = dict(seed=seed, instruction=original['instruction'], success=any(original['successes']),
                    checkpoint='/mnt/pfs/yuchen/groot/ft_fr_nu224/checkpoint-20000',
                    definition='mean softplus(s_raw + ht_sbias) over 8 steps x 7 channels + 0.001',
                    units='checkpoint-normalized action space; includes gripper; one shared scalar',
                    alignment='sigma[t] is predicted from the observation BEFORE executing action[t]',
                    all_300_replay_checks_valid=True, max_action_difference=max_delta,
                    sigma_min=float(sigma.min()), sigma_median=float(np.median(sigma)),
                    sigma_max=float(sigma.max()), sigma_last60_median=float(np.median(sigma[-60:])))
        (args.output/f'{seed}.json').write_text(json.dumps(meta,indent=2))
        print(json.dumps(meta), flush=True)
    client.close()
    env.close()
    print('ROLLOUT_SIGMA_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
