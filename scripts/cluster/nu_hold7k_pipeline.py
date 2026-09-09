"""Evaluate the old 20k checkpoint, then launch the fresh 7k-hold experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mnt/pfs/yuchen/groot')
CODE = ROOT / 'nu_hold7k_20260908'
sys.path.insert(0, str(CODE))
p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=('widowx', 'fractal'), required=True)
p.add_argument('--run-dir', type=Path, required=True)
p.add_argument('--eval-only', action='store_true')
args = p.parse_args()
repo = ROOT / 'Isaac-GR00T'
os.chdir(repo)
os.environ.update(HF_HOME='/mnt/pfs/yuchen/hf_home', WANDB_MODE='disabled',
                  GROOT_ROPE_CACHE='1', PYTHONUNBUFFERED='1',
                  NO_ALBUMENTATIONS_UPDATE='1',
                  PYTHONPATH=str(repo) + ':' + str(CODE))
python = str(repo / '.venv/bin/python')
status = args.run_dir / 'hold7k_pipeline_status.json'
def record(phase, **extra):
    status.write_text(json.dumps(dict(dataset=args.dataset, phase=phase, **extra), indent=2))
try:
    summary_path = args.run_dir / 'eval-20000-unseeded' / 'summary.json'
    if not summary_path.exists():
        record('evaluating_old_20k')
        with (args.run_dir / 'eval-20000-unseeded.pipeline.log').open('x') as log:
            subprocess.run([python, str(CODE/'nu_hold7k_eval.py'), '--dataset', args.dataset,
                            '--run-dir', str(args.run_dir), '--step', '20000'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = json.loads(summary_path.read_text())
    from nu_hold7k_eval import TASKS
    assert summary['dataset'] == args.dataset and summary['step'] == 20000
    assert summary['nu'] == 7
    assert sorted(t['task'] for t in summary['tasks']) == sorted(TASKS[args.dataset])
    assert all(t['seed'] is None for t in summary['tasks'])
    assert all(t['episodes'] == (50 if args.dataset == 'widowx' else 100)
               for t in summary['tasks'])
    if args.eval_only:
        record('old_20k_evaluation_complete_waiting_for_training_pod')
        sys.exit(0)
    record('training_fresh_hold7k')
    subprocess.run(['bash', str(CODE/'nu_hold7k_train.sh'), args.dataset], check=True)
    record('complete_including_new_18k_evaluation')
except Exception as exc:
    record('failed', error=repr(exc))
    raise
