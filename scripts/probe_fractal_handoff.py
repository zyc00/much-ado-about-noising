"""Close-drawer paired policy handoff; simulator Python, no training changes.

Reset + replay the recorded action prefix, including the real gripper wrapper.
Reject branches whose physical state, controller state, or input image differs.
The selected cohort is HT failures, not a benchmark success-rate estimate.
"""
import argparse
import copy
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from gr00t.eval.sim.SimplerEnv.simpler_env import GoogleFractalEnv
from gr00t.policy.server_client import PolicyClient


def proprio(obs):
    return np.concatenate([np.asarray(obs['state.' + k]) for k in ('x', 'y', 'z')])


def snapshot(env, obs):
    base = env.env.unwrapped
    return dict(physical=np.asarray(base.get_state()).copy(),
                controller=copy.deepcopy(base.agent.controller.get_state()),
                image=hashlib.sha256(obs['video.image'].tobytes()).hexdigest(),
                latch=[env.sticky_action_is_on, env.sticky_gripper_action,
                       env.gripper_action_repeat],
                xyz=proprio(obs).copy(),
                drawer=float(base.art_obj.get_qpos()[base.joint_idx]))


def flat_numeric(obj):
    if isinstance(obj, dict):
        parts = [flat_numeric(obj[k]) for k in sorted(obj)]
        return np.concatenate(parts) if parts else np.empty(0)
    if obj is None:
        return np.empty(0)
    return np.asarray(obj, dtype=float).ravel()


def verify(a, b):
    result = dict(physical_max_abs=float(np.max(np.abs(a['physical']-b['physical']))),
                  image_exact=a['image'] == b['image'], latch_exact=a['latch'] == b['latch'])
    ca, cb = flat_numeric(a['controller']), flat_numeric(b['controller'])
    result['controller_max_abs'] = float(np.max(np.abs(ca-cb))) if ca.size else 0.
    result['valid'] = (result['physical_max_abs'] < 1e-6 and result['image_exact']
                       and result['latch_exact'] and result['controller_max_abs'] < 1e-6)
    return result


def predict(client, obs):
    batch = {k: ((v,) if isinstance(v, str) else
                 np.asarray(v, dtype=np.uint8 if k.startswith('video.') else np.float32)[None, None])
             for k, v in obs.items()}
    actions, _ = client.get_action(batch)
    return {k: v[0, 0].copy() for k, v in actions.items()}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', choices=['collect', 'ht', 'flow', 'flow_zero'], required=True)
    p.add_argument('--episodes', type=int, default=10)
    args = p.parse_args()
    out = args.root / args.phase
    out.mkdir(parents=True, exist_ok=True)
    client = PolicyClient(host='127.0.0.1', port=5567, timeout_ms=120000)
    configs = client.get_modality_config()
    for name in ('video', 'state'):
        assert configs[name].delta_indices == [0], configs[name]
    env = GoogleFractalEnv('google_robot_close_drawer', image_size=(256, 320))
    for index in range(args.episodes):
        seed = 1234 + index  # Predefined new 10-scene pilot, not the old 5-env seed chain.
        source_path = args.root / 'collect' / f'{seed}.pkl'
        if args.phase == 'collect':
            obs, _ = env.reset(seed=seed)
            client.reset({'seed': seed})
            actions, states, frames, successes = [], [], [], []
            for step in range(300):
                states.append(snapshot(env, obs))
                if step % 3 == 0:
                    frames.append(obs['video.image'].copy())
                action = predict(client, obs)
                obs, _, _, _, info = env.step(action)
                actions.append(action)
                successes.append(bool(info['success']))
            states.append(snapshot(env, obs))
            # Predeclared stall: after step 60, last 20 steps average EEF
            # displacement <1 mm/step AND drawer displacement <1 mm.
            xyz = np.stack([s['xyz'] for s in states])
            drawer = np.array([s['drawer'] for s in states])
            candidates = [t for t in range(60, 221)
                          if np.linalg.norm(np.diff(xyz[t-20:t+1], axis=0), axis=1).mean() < .001
                          and abs(drawer[t]-drawer[t-20]) < .001]
            anchor = candidates[0] if candidates else 100
            branches = sorted(set([max(0, anchor-20), anchor])) if not any(successes) else []
            record = dict(seed=seed, actions=actions, states=states, successes=successes,
                          branches=branches, stall_detected=bool(candidates),
                          instruction=obs['annotation.human.action.task_description'])
            with source_path.open('wb') as f:
                pickle.dump(record, f)
            imageio.mimsave(out / f'{seed}.mp4', frames, fps=10)
            summary = {k: record[k] for k in ('seed', 'branches', 'stall_detected', 'instruction')}
            summary.update(success=any(successes), final_drawer=float(drawer[-1]))
            # Immediate prefix replay smoke check, before further expensive runs.
            obs, _ = env.reset(seed=seed)
            replay_step = branches[-1] if branches else 100
            for action in actions[:replay_step]:
                obs, *_ = env.step(action)
            summary['replay_check'] = verify(states[replay_step], snapshot(env, obs))
            save_json(out / f'{seed}.json', summary)
            print(json.dumps(summary), flush=True)
            if not summary['replay_check']['valid']:
                raise RuntimeError('Prefix replay is not exact; do not interpret handoff results.')
        else:
            with source_path.open('rb') as f:
                record = pickle.load(f)
            for branch in record['branches']:
                obs, _ = env.reset(seed=seed)
                for action in record['actions'][:branch]:
                    obs, *_ = env.step(action)
                check = verify(record['states'][branch], snapshot(env, obs))
                if not check['valid']:
                    save_json(out / f'{seed}_{branch}_invalid.json', check)
                    raise RuntimeError(f'Invalid paired state: {check}')
                client.reset({'seed': seed * 1000 + branch})
                states, actions, frames, successes = [], [], [], []
                for step in range(branch, 300):
                    states.append(snapshot(env, obs))
                    if step % 3 == 0:
                        frames.append(obs['video.image'].copy())
                    action = predict(client, obs)
                    obs, _, _, _, info = env.step(action)
                    actions.append(action)
                    successes.append(bool(info['success']))
                    if (step + 1) % 50 == 0:
                        print(f'PROGRESS {args.phase} seed={seed} branch={branch} step={step+1}', flush=True)
                states.append(snapshot(env, obs))
                action_difference = max(float(np.max(np.abs(a[k]-b[k])))
                                        for a, b in zip(actions, record['actions'][branch:]) for k in a)
                summary = dict(seed=seed, branch=branch, method=args.phase,
                               success=any(successes), replay_check=check,
                               final_drawer=states[-1]['drawer'],
                               max_action_difference_from_ht=action_difference)
                save_json(out / f'{seed}_{branch}.json', summary)
                with (out / f'{seed}_{branch}.pkl').open('wb') as f:
                    pickle.dump(dict(summary=summary, states=states, actions=actions, successes=successes), f)
                imageio.mimsave(out / f'{seed}_{branch}.mp4', frames, fps=10)
                print(json.dumps(summary), flush=True)
    env.close()
    client.close()
    if args.phase == 'flow_zero':
        subprocess.run([sys.executable, str(Path(__file__).with_name('summarize_fractal_handoff.py')),
                        '--root', str(args.root)], check=True)
    print(f'PHASE_DONE {args.phase}', flush=True)


if __name__ == '__main__':
    main()
