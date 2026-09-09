"""Direct authenticated Hugging Face dataset download on a PARCC compute node."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from huggingface_hub import snapshot_download, whoami

ROOT = Path('/vast/projects/jiayuanm/mao-lab/yuchen')
SPECS = {
    'widowx': dict(
        repo='IPEC-COMMUNITY/bridge_orig_lerobot',
        revision='0e9d76d07e9df3ea3eba257b2520d4913833fad2',
        local_dir=ROOT/'data/widowx', allow=None,
        task_roots=1, episodes=53192, parquet=53192, videos=212768),
    'fractal': dict(
        repo='IPEC-COMMUNITY/fractal20220817_data_lerobot',
        revision='91bf7d7f7ce50770a1ba5c6db14b8d1c0815122e',
        local_dir=ROOT/'data/fractal', allow=None,
        task_roots=1, episodes=87212, parquet=87212, videos=87212),
    'gr1': dict(
        repo='nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim',
        revision='09c6de8af50168090e7e9cc01e1ec3bce788de24',
        local_dir=ROOT/'data/gr1', allow=['LeRobot/gr1_unified.*/*'],
        task_roots=24, episodes=24000, parquet=24000, videos=24000),
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def validate(name, spec):
    local=spec['local_dir']
    if name=='gr1':
        roots=sorted((local/'LeRobot').glob('gr1_unified.*'))
    else:
        roots=[local]
    assert len(roots)==spec['task_roots'], (len(roots),spec['task_roots'])
    infos=[]
    for root in roots:
        info_path=root/'meta/info.json'
        assert info_path.is_file(), info_path
        info=json.loads(info_path.read_text())
        infos.append(dict(root=str(root),episodes=int(info['total_episodes']),
                          videos=int(info['total_videos']),frames=int(info['total_frames']),
                          robot_type=info.get('robot_type')))
    parquet=sum(1 for root in roots for _ in root.glob('data/**/*.parquet'))
    videos=sum(1 for root in roots for _ in root.glob('videos/**/*.mp4'))
    episodes=sum(x['episodes'] for x in infos)
    assert episodes==spec['episodes'], (episodes,spec['episodes'])
    assert parquet==spec['parquet'], (parquet,spec['parquet'])
    assert videos==spec['videos'], (videos,spec['videos'])
    return dict(task_roots=len(roots),episodes=episodes,parquet_files=parquet,
                mp4_files=videos,metadata=infos)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset',choices=SPECS)
    args=parser.parse_args()
    name=args.dataset
    spec=SPECS[name]
    audit=ROOT/'data/_hf_download_20260908'
    audit.mkdir(parents=True,exist_ok=True)
    status=audit/f'{name}.status.json'
    complete=audit/f'{name}.complete.json'
    if complete.exists():
        validation=validate(name,spec)
        print(json.dumps(dict(dataset=name,status='already_complete',validation=validation)),flush=True)
        return
    identity=whoami()
    assert identity.get('name')=='yuchen0187' and identity.get('isPro') is True
    spec['local_dir'].mkdir(parents=True,exist_ok=True)
    public={k:(str(v) if isinstance(v,Path) else v) for k,v in spec.items()}
    atomic_json(status,dict(dataset=name,phase='downloading',started_utc=timestamp(),
                            account=identity['name'],is_pro=True,spec=public,
                            slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    kwargs=dict(repo_id=spec['repo'],repo_type='dataset',revision=spec['revision'],
                local_dir=spec['local_dir'],max_workers=8)
    if spec['allow']:
        kwargs['allow_patterns']=spec['allow']
    kwargs['ignore_patterns']=['*.jpgpack','*.part']
    last=None
    for attempt in range(1,11):
        try:
            print(json.dumps(dict(dataset=name,event='snapshot_download',attempt=attempt,
                                  utc=timestamp(),repo=spec['repo'],revision=spec['revision'])),flush=True)
            snapshot_download(**kwargs)
            last=None
            break
        except Exception as exc:
            last=f'{type(exc).__name__}: {exc}'
            delay=min(600,30*attempt)
            atomic_json(status,dict(dataset=name,phase='retrying',attempt=attempt,
                                    utc=timestamp(),error=last,retry_seconds=delay,
                                    account=identity['name'],spec=public,
                                    slurm_job_id=os.environ.get('SLURM_JOB_ID')))
            print(json.dumps(dict(dataset=name,event='retry',attempt=attempt,
                                  error=last,seconds=delay)),flush=True)
            time.sleep(delay)
    if last is not None:
        raise RuntimeError(f'Exhausted download retries: {last}')
    atomic_json(status,dict(dataset=name,phase='validating',utc=timestamp(),spec=public,
                            slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    validation=validate(name,spec)
    result=dict(dataset=name,phase='complete',completed_utc=timestamp(),
                source='Hugging Face direct from PARCC compute node',
                account=identity['name'],is_pro=True,spec=public,validation=validation,
                slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    atomic_json(complete,result)
    atomic_json(status,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
