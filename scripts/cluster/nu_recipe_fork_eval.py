"""Separate unseeded references, legacy seeded reproductions, and fixed scenes."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

REPO=Path('/mnt/pfs/yuchen/groot/Isaac-GR00T')
TASKS=['google_robot_pick_coke_can','google_robot_pick_object','google_robot_move_near',
       'google_robot_open_drawer','google_robot_close_drawer','google_robot_place_in_closed_drawer']


def evaluate(checkpoint,out,gpu,task,seed,fixed=False):
    dest=out/f'{task}.json'
    if dest.exists():
        result=json.loads(dest.read_text())
        assert result['checkpoint']==str(checkpoint) and result['seed']==seed and result['episodes']==100
        if fixed:assert result['protocol']=='fixed_episode_seeds_v1'
        return result
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    env.pop('GR00T_EVAL_SEED',None)
    scene_manifest=out/f'{task}.scenes.json'
    if fixed:env['NU_SCENE_MANIFEST_PATH']=str(scene_manifest)
    port=str(5755+gpu)
    attempt=0
    while (out/f'{task}.attempt{attempt}.server.log').exists():attempt+=1
    server_log=out/f'{task}.attempt{attempt}.server.log'
    client_log=out/f'{task}.attempt{attempt}.rollout.log'
    with server_log.open('x') as slog,client_log.open('x') as clog:
        server=subprocess.Popen([str(REPO/'.venv/bin/python'),str(Path(__file__).with_name('nu_hold7k_server.py')),
            '--checkpoint',str(checkpoint),'--embodiment-tag','SIMPLER_ENV_GOOGLE','--port',port],
            cwd=REPO,env=env,stdout=slog,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+1800
            while True:
                if server.poll() is not None:raise RuntimeError(str(server_log))
                try:
                    with socket.create_connection(('127.0.0.1',int(port)),timeout=2):break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError(str(server_log))
                    time.sleep(5)
            collector=str(Path(__file__).with_name('nu_fixed_scenes_rollout.py')) if fixed else 'gr00t/eval/rollout_policy.py'
            command=[str(REPO/'gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python'),
                collector,'--n-episodes','100','--policy-client-host','127.0.0.1',
                '--policy-client-port',port,'--max-episode-steps','300','--env-name',f'simpler_env_google/{task}',
                '--n-action-steps','1','--n-envs','5']
            if seed is not None:command+=['--seed',str(seed)]
            subprocess.run(command,cwd=REPO,env=env,stdout=clog,stderr=subprocess.STDOUT,check=True,timeout=14400)
        finally:
            server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait()
    raw=client_log.read_text()
    success=float(re.findall(r'success rate:\s*([0-9.]+)',raw,re.I)[-1])
    line=next(s.split(':',1)[1].strip() for s in raw.splitlines() if s.startswith('results:'))
    outcomes=ast.literal_eval(line)[1]
    assert len(outcomes)==100 and abs(sum(outcomes)/len(outcomes)-success)<1e-9
    result=dict(task=task,checkpoint=str(checkpoint),seed=seed,episodes=100,
                successes=int(sum(outcomes)),success_rate=success,outcomes=[bool(v) for v in outcomes],
                protocol='reference_unseeded' if seed is None else 'fixed_episode_seeds_v1' if fixed else 'same_seed_reproduction',
                log=str(client_log),n_envs=5,n_action_steps=1,max_steps=300)
    if fixed:
        scenes=json.loads(scene_manifest.read_text())
        assert scenes['completed']==100
        assert [r['seed'] for r in scenes['episodes']]==list(range(seed,seed+100))
        assert [r['success'] for r in scenes['episodes']]==result['outcomes']
        result['scene_manifest']=str(scene_manifest)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print('EVAL_DONE',json.dumps({k:v for k,v in result.items() if k!='outcomes'}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    # Keep the old paired CLI spelling for already-submitted reproduction jobs;
    # it does NOT imply scene identity. New "both" uses truly fixed scene seeds.
    p.add_argument('--protocol',choices=['both','paired','fixed','unseeded'],default='both')
    p.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    args=p.parse_args()
    ck=args.run_dir/f'checkpoint-{args.step}'
    assert json.loads((ck/'trainer_state.json').read_text())['global_step']==args.step
    schedules=([(None,False),(1234,True)] if args.protocol=='both' else
               [(1234,True)] if args.protocol=='fixed' else
               [(1234,False)] if args.protocol=='paired' else [(None,False)])
    gpus=list(map(int,args.gpus.split(',')))
    for seed,fixed in schedules:
        suffix=f'fixed{seed}' if fixed else str(seed) if seed is not None else 'unseeded'
        out=args.run_dir/f'eval-{args.step}-nudebug-{suffix}'
        out.mkdir(exist_ok=True)
        def worker(gpu,tasks):return [evaluate(ck,out,gpu,t,seed,fixed) for t in tasks]
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(worker,gpu,args.tasks[i::len(gpus)]) for i,gpu in enumerate(gpus)]
            rows=[r for f in futures for r in f.result()]
        rows.sort(key=lambda r:TASKS.index(r['task']))
        (out/'summary.json').write_text(json.dumps(dict(checkpoint=str(ck),seed=seed,tasks=rows,
            macro_success_rate=sum(r['success_rate'] for r in rows)/len(rows)),indent=2)+'\n')


if __name__=='__main__':main()
