"""Ensure policy-dependent completion order does not select different scenes."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).parent/'cluster'))
from nu_fixed_scenes_rollout import collect,fingerprint


class Env:
    def reset(self,seed):
        self.seeds=np.array(seed);self.steps=np.zeros(len(seed),int)
        return self.obs(),{}
    def obs(self):return {'state':self.seeds[:,None].astype(float),'image':np.repeat(self.seeds[:,None],3,axis=1)}
    def step(self,actions):
        self.steps+=1
        budgets=2+self.seeds%5+actions.astype(int)
        done=self.steps>=budgets
        success=done&(self.seeds%2==0)
        self.steps[done]=0
        self.seeds[done]+=1000  # auto-reset scenes must NOT enter the result
        return self.obs(),success.astype(float),done,np.zeros(len(done),bool),{'success':success}


class Policy:
    def __init__(self,extra):self.extra=extra
    def reset(self):pass
    def get_action(self,obs):return np.full(len(obs['state']),self.extra),{}


out=[]
for extra in (0,5):
    out.append(collect(Env(),Policy(extra),10,2,1234,step_counter=lambda infos,i:1))
for success,length,reward,info in out:
    assert info['scene_seed']==list(range(1234,1244))
    assert success==[s%2==0 for s in range(1234,1244)]
    assert len(length)==len(reward)==10
assert out[0][3]['initial_observation_sha256']==out[1][3]['initial_observation_sha256']
assert out[0][1]!=out[1][1]
assert fingerprint({'a':np.array([[1],[2]]),'b':np.array(['x','y'])},0,2)==fingerprint(
    {'b':np.array(['x','y']),'a':np.array([[1],[2]])},0,2)
print('Fixed-scene collector checks passed, including asynchronous auto-reset exclusion.')
