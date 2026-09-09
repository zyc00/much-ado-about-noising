"""Re-evaluate both old 20k checkpoints unseeded, then start Fractal warm-hold training."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mnt/pfs/yuchen/groot')
CODE = ROOT / 'nu_hold7k_20260908'
REPO = ROOT / 'Isaac-GR00T'
sys.path.insert(0, str(CODE))
from nu_hold7k_eval import TASKS

os.chdir(REPO)
os.environ.update(HF_HOME='/mnt/pfs/yuchen/hf_home', WANDB_MODE='disabled',
                  GROOT_ROPE_CACHE='1', PYTHONUNBUFFERED='1',
                  NO_ALBUMENTATIONS_UPDATE='1', PYTHONPATH=str(REPO)+':'+str(CODE))
os.environ.pop('GR00T_EVAL_SEED', None)
status = ROOT / 'ft_fr_nustair_20260907/hold7k_pipeline_status.json'

def record(phase, **extra):
    status.write_text(json.dumps(dict(dataset='fractal', phase=phase, **extra), indent=2))

def evaluate(dataset, short, gpus):
    run = ROOT / f'ft_{short}_nustair_20260907'
    with (run/'eval-20000-unseeded.pipeline.log').open('x') as log:
        subprocess.run([str(REPO/'.venv/bin/python'), str(CODE/'nu_hold7k_eval.py'),
                        '--dataset', dataset, '--run-dir', str(run), '--step', '20000',
                        '--gpus', gpus], stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = json.loads((run/'eval-20000-unseeded/summary.json').read_text())
    assert summary['dataset'] == dataset and summary['step'] == 20000 and summary['nu'] == 7
    assert sorted(t['task'] for t in summary['tasks']) == sorted(TASKS[dataset])
    assert all(t['seed'] is None for t in summary['tasks'])
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    try:
        record('reevaluating_old_20k_unseeded', also='widowx')
        with ThreadPoolExecutor(max_workers=2) as pool:
            fr = pool.submit(evaluate, 'fractal', 'fr', '0,1,2,3,4,5')
            wx = pool.submit(evaluate, 'widowx', 'wx', '6,7')
            fr.result()
            wx.result()
        record('training_fresh_hold7k')
        subprocess.run(['bash', str(CODE/'nu_hold7k_train.sh'), 'fractal'], check=True)
        record('complete_including_new_18k_unseeded_evaluation')
    except Exception as exc:
        record('failed', error=repr(exc))
        raise
