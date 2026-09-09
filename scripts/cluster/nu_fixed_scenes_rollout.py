"""Diagnostic-only collector: predetermined episode seeds, not first-N finishes.

Leaves the shared rollout implementation untouched. Every batch runs each
initial scene exactly once; auto-reset episodes from finished slots are ignored.
Unseeded reference evaluations continue to use the original collector.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def fingerprint(observations, index, n_envs):
    digest=hashlib.sha256()
    def visit(value,path):
        if isinstance(value,dict):
            for key in sorted(value):visit(value[key],path+'/'+str(key))
            return
        value=np.asarray(value)
        if value.ndim and value.shape[0]==n_envs:value=value[index]
        value=np.asarray(value)
        digest.update(path.encode())
        digest.update(str((value.shape,str(value.dtype))).encode())
        if value.dtype.hasobject or value.dtype.kind in 'US':
            digest.update(json.dumps(value.tolist(),sort_keys=True).encode())
        else:digest.update(np.ascontiguousarray(value).tobytes())
    visit(observations,'obs')
    return digest.hexdigest()


def collect(env,policy,n_episodes,n_envs,seed,step_counter=None):
    assert seed is not None and n_episodes%n_envs==0
    if step_counter is None:
        from gr00t.eval.rollout_policy import _macro_step_env_steps
        step_counter=_macro_step_env_steps
    rows=[]
    for start in range(0,n_episodes,n_envs):
        seeds=[int(seed)+start+i for i in range(n_envs)]
        observations,_=env.reset(seed=seeds)
        hashes=[fingerprint(observations,i,n_envs) for i in range(n_envs)]
        policy.reset()
        active=np.ones(n_envs,dtype=bool)
        successes=np.zeros(n_envs,dtype=bool)
        lengths=np.zeros(n_envs,dtype=int)
        rewards_sum=np.zeros(n_envs,dtype=float)
        iterations=0
        while active.any():
            actions,_=policy.get_action(observations)
            observations,rewards,terminated,truncated,infos=env.step(actions)
            iterations+=1
            assert iterations<=1000, 'An episode failed to terminate under the diagnostic step budget.'
            for i in np.flatnonzero(active):
                if 'success' in infos:successes[i]|=bool(np.any(infos['success'][i]))
                final=infos.get('final_info')
                if final is not None and final[i] is not None:
                    successes[i]|=bool(np.any(final[i].get('success',False)))
                lengths[i]+=step_counter(infos,i)
                rewards_sum[i]+=float(rewards[i])
                if terminated[i] or truncated[i]:active[i]=False
        for i in range(n_envs):
            assert lengths[i]>0
            rows.append(dict(episode_id=start+i,seed=seeds[i],initial_observation_sha256=hashes[i],
                             success=bool(successes[i]),length=int(lengths[i]),reward=float(rewards_sum[i])))
        if os.environ.get('NU_SCENE_MANIFEST_PATH'):
            dest=Path(os.environ['NU_SCENE_MANIFEST_PATH'])
            dest.write_text(json.dumps(dict(protocol='fixed_episode_seeds_v1',requested=n_episodes,
                                           completed=len(rows),episodes=rows),indent=2)+'\n')
        print('FIXED_SCENES_COMPLETED',len(rows),'/',n_episodes,flush=True)
    return ([r['success'] for r in rows],[r['length'] for r in rows],[r['reward'] for r in rows],
            {'scene_seed':[r['seed'] for r in rows],
             'initial_observation_sha256':[r['initial_observation_sha256'] for r in rows]})


def main():
    import tyro
    from gr00t.eval import rollout_policy as rp
    args=tyro.cli(rp.RolloutConfig)
    assert args.seed is not None and args.n_action_steps==1 and args.max_episode_steps==300
    assert not args.model_path and args.policy_client_host and args.policy_client_port is not None
    rp._collect_rollout_episodes=collect
    results=rp.run_gr00t_sim_policy(**asdict(args))
    print('results: ',results)
    print('success rate: ',np.mean(results[1]))


if __name__=='__main__':main()
