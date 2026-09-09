"""Reference unseeded SIMPLER evaluation; seeded results are retained separately."""
import ast
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

ROOT = Path('/mnt/pfs/yuchen/groot/Isaac-GR00T')
TASKS = {
    'widowx': ['widowx_spoon_on_towel', 'widowx_carrot_on_plate',
               'widowx_stack_cube', 'widowx_put_eggplant_in_basket',
               'widowx_put_eggplant_in_sink', 'widowx_open_drawer', 'widowx_close_drawer'],
    'fractal': ['google_robot_pick_coke_can', 'google_robot_pick_object',
                'google_robot_move_near', 'google_robot_open_drawer',
                'google_robot_close_drawer', 'google_robot_place_in_closed_drawer'],
}


def evaluate_task(dataset, checkpoint, output, gpu, task):
    is_wx = dataset == 'widowx'
    episodes, nas = (50, 4) if is_wx else (100, 1)
    tag = 'SIMPLER_ENV_WIDOWX' if is_wx else 'SIMPLER_ENV_GOOGLE'
    prefix = 'simpler_env_widowx' if is_wx else 'simpler_env_google'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL='egl',
               PYOPENGL_PLATFORM='egl')
    env.pop('GR00T_EVAL_SEED', None)
    port = str(5655 + gpu)
    server_log = output / f'{task}.server.log'
    client_log = output / f'{task}.rollout.log'
    with server_log.open('w') as slog, client_log.open('w') as clog:
        server = subprocess.Popen([
            str(ROOT / '.venv/bin/python'), str(Path(__file__).with_name('nu_hold7k_server.py')),
            '--checkpoint', str(checkpoint), '--embodiment-tag', tag,
            '--port', port,
        ], cwd=ROOT, env=env, stdout=slog, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited: {server_log}')
                try:
                    with socket.create_connection(('127.0.0.1', int(port)), timeout=2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(str(server_log))
                    time.sleep(10)
            time.sleep(5)
            subprocess.run([
                str(ROOT / 'gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python'),
                'gr00t/eval/rollout_policy.py', '--n-episodes', str(episodes),
                '--policy-client-host', '127.0.0.1', '--policy-client-port', port,
                '--max-episode-steps', '300', '--env-name', f'{prefix}/{task}',
                '--n-action-steps', str(nas), '--n-envs', '5',
            ], cwd=ROOT, env=env, stdout=clog, stderr=subprocess.STDOUT,
                check=True, timeout=14400)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    matches = re.findall(r'success rate:\s*([0-9.]+)', client_log.read_text(), re.I)
    if not matches:
        raise ValueError(f'No success rate: {client_log}')
    success = float(matches[-1])
    assert 0 <= success <= 1, (task, success)
    raw = next(line.split(':', 1)[1].strip() for line in client_log.read_text().splitlines()
               if line.startswith('results:'))
    outcomes = ast.literal_eval(raw)[1]
    assert abs(sum(outcomes) / len(outcomes) - success) < 1e-9
    result = {'task': task, 'success_rate': success, 'episodes': episodes,
              'actual_episodes': len(outcomes), 'successes': sum(outcomes), 'seed': None}
    (output / f'{task}.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=TASKS, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--step', type=int, default=20000)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    args = parser.parse_args()
    for step in (args.step,):
        checkpoint = args.run_dir / f'checkpoint-{step}'
        assert json.loads((checkpoint / 'trainer_state.json').read_text())['global_step'] == step
        nu = json.loads((checkpoint / 'config.json').read_text())['ht_df']
        output = args.run_dir / f'eval-{step}-unseeded'
        output.mkdir()
        tasks = TASKS[args.dataset]
        gpus = [int(g) for g in args.gpus.split(',')]
        assert gpus and len(gpus) == len(set(gpus))
        def worker(gpu, assigned):
            return [evaluate_task(args.dataset, checkpoint, output, gpu, task)
                    for task in assigned]
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu, tasks[i::len(gpus)])
                       for i, gpu in enumerate(gpus)]
            results = [r for future in futures for r in future.result()]
        results.sort(key=lambda r: tasks.index(r['task']))
        summary = {'dataset': args.dataset, 'step': step, 'nu': nu,
                   'seed_protocol': 'unseeded, five vector environments, matching 10k', 'tasks': results,
                   'macro_success_rate': sum(r['success_rate'] for r in results) / len(results)}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
