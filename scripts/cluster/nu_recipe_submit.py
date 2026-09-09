"""Submit a bounded three-arm continuation study, or the paired baseline audit."""
import argparse
import json
import subprocess


def pod(name,command,gpus,cpu,memory,app):
    return dict(apiVersion='v1',kind='Pod',metadata=dict(name=name,labels=dict(owner='yuchen',app=app)),
        spec=dict(restartPolicy='Never',affinity=dict(nodeAffinity=dict(requiredDuringSchedulingIgnoredDuringExecution=dict(
            nodeSelectorTerms=[dict(matchExpressions=[dict(key='aihc.baidu.com/dedicated-pool',operator='DoesNotExist')])]))),
            containers=[dict(name='main',image='zyc00/easypod:latest',command=['bash','-lc'],args=[command],
                env=[dict(name='NVIDIA_DRIVER_CAPABILITIES',value='all'),
                     dict(name='HF_HOME',value='/mnt/pfs/yuchen/hf_home'),
                     dict(name='HF_HUB_OFFLINE',value='1'),dict(name='TRANSFORMERS_OFFLINE',value='1'),
                     dict(name='PYTHONUNBUFFERED',value='1'),dict(name='NO_ALBUMENTATIONS_UPDATE',value='1')],
                resources=dict(requests={'cpu':str(cpu),'memory':memory,'nvidia.com/gpu':str(gpus)},
                               limits={'cpu':str(cpu),'memory':memory,'nvidia.com/gpu':str(gpus)}),
                volumeMounts=[dict(name='pfs',mountPath='/mnt/pfs'),dict(name='dshm',mountPath='/dev/shm')])],
            volumes=[dict(name='pfs',persistentVolumeClaim=dict(claimName='main-pfs-pvc')),
                     dict(name='dshm',emptyDir=dict(medium='Memory'))]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--which',choices=['train','paired'],required=True)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--arms',nargs='+',choices=['stair14','smooth14','smooth7'],default=['stair14','smooth14','smooth7'])
    p.add_argument('--attempt',type=int,default=1)
    args=p.parse_args()
    code='/mnt/pfs/yuchen/nu_recipe_debug_20260908/code'
    if args.which=='train':
        assert args.attempt>=1
        suffix=f'-r{args.attempt}' if args.attempt>1 else ''
        run_suffix=f'_r{args.attempt}' if args.attempt>1 else ''
        items=[pod(f'yuchen-fr-nudebug-{arm}-0908{suffix}',
                   f'NU_DEBUG_ATTEMPT_SUFFIX={run_suffix} bash {code}/nu_recipe_fork_train.sh {arm}',
                   8,96,'768Gi','nu-recipe-training') for arm in args.arms]
    else:
        command=f'''set -euo pipefail
cd /mnt/pfs/yuchen/groot/Isaac-GR00T
export PYTHONPATH="$PWD:{code}"
root=/mnt/pfs/yuchen/groot
.venv/bin/python {code}/nu_recipe_fork_eval.py --run-dir "$root/ft_fr_nu1024_hold7k_to14_20260908" --step 18000 --protocol paired --gpus 0,1,2 --tasks google_robot_pick_coke_can google_robot_move_near google_robot_close_drawer
.venv/bin/python {code}/nu_recipe_fork_eval.py --run-dir "$root/ft_fr_nu224" --step 20000 --protocol paired --gpus 0,1,2 --tasks google_robot_pick_coke_can google_robot_move_near google_robot_close_drawer
'''
        items=[pod('yuchen-fr-nudebug-paired-0908',command,3,24,'96Gi','nu-recipe-paired-eval')]
    manifest=json.dumps(dict(apiVersion='v1',kind='List',items=items))
    if args.apply:
        # create, not apply: an existing job must not be silently reconfigured.
        subprocess.run(['kubectl','create','-f','-'],input=manifest,text=True,check=True)
    else:
        print(manifest)


if __name__=='__main__':main()
