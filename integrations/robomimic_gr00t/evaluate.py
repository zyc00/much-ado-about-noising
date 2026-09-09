"""Native GR00T policy-client rollouts and expert-action replay on RoboMimic."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import h5py
import numpy as np
from common import (TASKS, absolute_env_metadata, decode_actions, encode_actions,
                    policy_observation, state_keys, unpack_policy_actions)


def make_env(metadata, task, height, width):
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.envs.env_robosuite import EnvRobosuite
    ObsUtils.initialize_obs_modality_mapping_from_dict({
        'low_dim': state_keys(task),
        'rgb': [f'{c}_image' for c in TASKS[task]['cameras']],
    })
    metadata = absolute_env_metadata(metadata, task)
    kwargs = metadata['env_kwargs']
    # EnvRobosuite supplies these explicit arguments itself.
    for key in ('has_renderer', 'has_offscreen_renderer', 'use_camera_obs'):
        kwargs.pop(key, None)
    kwargs.update(camera_heights=height, camera_widths=width)
    env = EnvRobosuite(env_name=metadata['env_name'], render=False,
                      render_offscreen=True, use_image_obs=True,
                      postprocess_visual_obs=False, **kwargs)
    return env


def replay(source, dataset, output, demo_ids):
    """Replay converted actions, not just raw HDF5 actions; validate pixels/state too."""
    import pyarrow.parquet as pq
    provenance = json.loads((dataset / 'meta/provenance.json').read_text())
    task = provenance['task']
    selected = {x['source_demo']: x['episode_index'] for x in provenance['episodes']}
    results = []
    with h5py.File(source) as f:
        meta = json.loads(f['data'].attrs['env_args'])
        camera = TASKS[task]['cameras'][0]
        height, width = f[f'data/{demo_ids[0]}/obs/{camera}_image'].shape[1:3]
        env = make_env(meta, task, height, width)
        try:
            for name in demo_ids:
                d = f[f'data/{name}']
                index = selected[name]
                table = pq.read_table(dataset / f'data/chunk-000/episode_{index:06d}.parquet').to_pydict()
                encoded = np.asarray(table['action'])
                raw = decode_actions(encoded, task)
                # Compare rotations in matrix representation (rotvec has equivalent branches).
                roundtrip_error = float(np.abs(encode_actions(raw, task)-encoded).max())
                def reset_recorded(t):
                    # Reset controller internals and nullspace reference as well
                    # as MuJoCo qpos/qvel. State-only reset is not a controlled
                    # one-step comparison after a preceding controller action.
                    np.random.seed(42)
                    return env.reset_to({'states': d['states'][t], 'model': d.attrs['model_file']})

                obs = reset_recorded(0)
                from common import pack_state
                state_error = float(np.abs(pack_state(obs, task)-np.asarray(table['observation.state'][0])).max())
                pixels = {}
                for camera in TASKS[task]['cameras']:
                    live = obs[f'{camera}_image'].astype(float)
                    recorded = d['obs'][f'{camera}_image'][0].astype(float)
                    pixels[camera] = {'same_mae': float(np.abs(live-recorded).mean()),
                                      'vertical_flip_mae': float(np.abs(live[::-1]-recorded).mean()),
                                      'rotate180_mae': float(np.abs(live[::-1, ::-1]-recorded).mean())}
                # Exercise the same wire format and action grouping used at deployment.
                wire = policy_observation(obs, task)
                assert all(v.shape[:2] == (1, 1) for k,v in wire.items() if k.startswith(('state.', 'video.')))
                success = False
                trajectory_errors = []
                for t, action in enumerate(raw):
                    obs, reward, done, info = env.step(action)
                    success = success or bool(env.is_success()['task'])
                    if t+1 < len(d['states']):
                        trajectory_errors.append(float(np.linalg.norm(env.get_state()['states']-d['states'][t+1])))
                # Original absolute-action datasets themselves may not replay with
                # 100% success. Compare against their original float64 actions.
                reset_recorded(0)
                baseline_success = False
                for action in d['actions'][:]:
                    env.step(action)
                    baseline_success = baseline_success or bool(env.is_success()['task'])
                reset_recorded(0)
                float32_success = False
                for action in d['actions'][:].astype(np.float32):
                    env.step(action)
                    float32_success = float32_success or bool(env.is_success()['task'])
                one_step_errors = []
                for t in np.linspace(0, len(raw)-1, 5, dtype=int):
                    reset_recorded(t)
                    env.step(d['actions'][t])
                    expected = env.get_state()['states'].copy()
                    reset_recorded(t)
                    env.step(raw[t])
                    one_step_errors.append(float(np.abs(env.get_state()['states']-expected).max()))
                result = {'demo': name, 'steps': len(raw), 'success': success,
                          'raw_action_replay_success': baseline_success,
                          'raw_float32_replay_success': float32_success,
                          'one_step_raw_vs_converted_max_error': max(one_step_errors),
                          'action_roundtrip_max_error': roundtrip_error,
                          'initial_proprioception_max_error': state_error,
                          'image_orientation': pixels,
                          'mean_state_trajectory_l2_error': float(np.mean(trajectory_errors))}
                results.append(result)
                print(json.dumps(result), flush=True)
        finally:
            env.env.close()
    summary = {'mode': 'expert_replay', 'task': task, 'source': str(source),
               'dataset': str(dataset), 'results': results,
               'full_replay_success_agreement': all(x['success'] == x['raw_action_replay_success'] for x in results),
               'success_rate': float(np.mean([x['success'] for x in results]))}
    output.write_text(json.dumps(summary, indent=2)+'\n')
    if any(x['action_roundtrip_max_error'] > 1e-5 or x['initial_proprioception_max_error'] > 1e-3 for x in results):
        raise AssertionError(f'Mapping check failed; see {output}')
    if any(x['one_step_raw_vs_converted_max_error'] > 1e-3 for x in results):
        raise AssertionError(f'Controller transition mismatch: {output}')
    if not summary['full_replay_success_agreement']:
        print(f'WARNING: complete contact-rich replays differ; see raw/float32/Rot6D controls in {output}', flush=True)


def rollout(dataset, output, host, port, episodes, seed_start, execute_steps, max_steps=None):
    from client import PolicyClient
    provenance = json.loads((dataset / 'meta/provenance.json').read_text())
    task = provenance['task']
    meta = json.loads((dataset / 'meta/robomimic_env.json').read_text())
    data_info = json.loads((dataset / 'meta/info.json').read_text())
    feature = data_info['features'][f'observation.images.{TASKS[task]["cameras"][0]}']
    height, width = feature['shape'][:2]
    env = make_env(meta, task, height, width)
    client = PolicyClient(host=host, port=port, timeout_ms=120000)
    results = []
    horizon = TASKS[task]['horizon'] if max_steps is None else max_steps
    try:
        for seed in range(seed_start, seed_start+episodes):
            np.random.seed(seed)
            obs = env.reset()
            client.reset()
            success, steps, queries = False, 0, 0
            began = time.monotonic()
            while steps < horizon and not success:
                actions, _ = client.get_action(policy_observation(obs, task))
                raw = unpack_policy_actions(actions, task)
                if not 1 <= execute_steps <= len(raw):
                    raise ValueError('Execution horizon exceeds predicted chunk')
                queries += 1
                for action in raw[:execute_steps]:
                    # Absolute XYZ/rotation must NOT be clipped to [-1,1].
                    action[6::7] = np.clip(action[6::7], -1, 1)
                    obs, _, _, _ = env.step(action)
                    steps += 1
                    success = bool(env.is_success()['task'])
                    if success or steps >= horizon:
                        break
            results.append({'seed': seed, 'success': success, 'steps': steps,
                            'policy_queries': queries, 'wall_seconds': time.monotonic()-began})
            print(json.dumps(results[-1]), flush=True)
            output.write_text(json.dumps({'task': task, 'episodes': results,
                'success_rate': float(np.mean([r['success'] for r in results])),
                'execute_steps': execute_steps, 'max_episode_steps': horizon,
                'smoke_only': max_steps is not None and max_steps != TASKS[task]['horizon']}, indent=2)+'\n')
    finally:
        client.close()
        env.env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['replay', 'policy'], required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--demo-ids', nargs='+', default=['demo_0'])
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5555)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--seed-start', type=int, default=10000)
    p.add_argument('--execute-steps', type=int, default=4)
    p.add_argument('--max-steps', type=int, help='Only for short interface smoke tests; not benchmark evaluation')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == 'replay':
        if a.source is None:
            p.error('--source is required for replay')
        replay(a.source, a.dataset, a.output, a.demo_ids)
    else:
        rollout(a.dataset, a.output, a.host, a.port, a.episodes, a.seed_start, a.execute_steps, a.max_steps)
