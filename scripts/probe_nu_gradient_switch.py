"""Paired, no-update gradient audit of Fractal's joint Student-t nu staircase.

Replays identical processed inputs and dropout RNG for every nu comparison.
Uses actual trainable parameters and train-mode forward/backward, not an
output-gradient proxy. Checkpoints and optimizer state are never modified.
"""
import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from probe_widowx_general_scale import load_model, map_tensors

ROOT = Path('/mnt/pfs/yuchen/groot')
RUN = ROOT / 'ft_fr_nu1024_hold7k_to14_20260908'
TRANSITIONS = [(7000, 1024., 512.), (8000, 512., 256.),
               (9000, 256., 128.), (10000, 128., 64.),
               (11000, 64., 32.), (12000, 32., 14.)]
SEED = 20260908


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare(out, batch_size, cohorts):
    rows = []
    for ordinal, line in enumerate((ROOT/'fractal_lerobot/meta/episodes.jsonl').open()):
        ep = json.loads(line)
        if ep['length'] >= 12:
            rows.append(dict(ep, ordinal=ordinal))
    rng = np.random.default_rng(SEED)
    pools = {'mixed_0': rows, 'mixed_1': rows,
             'move_near': [r for r in rows if r['tasks'][0].lower().startswith('move ') and ' near ' in r['tasks'][0].lower()],
             'close_drawer': [r for r in rows if r['tasks'][0].lower().startswith('close ') and 'drawer' in r['tasks'][0].lower()]}
    batches = {}
    for name, pool in pools.items():
        if name not in cohorts:
            continue
        assert len(pool) >= batch_size, (name, len(pool))
        chosen = rng.choice(len(pool), batch_size, replace=False)
        batches[name] = [dict(pool[int(i)], step=int(rng.integers(0, pool[int(i)]['length']-7))) for i in chosen]
    protocol = dict(seed=SEED, run=str(RUN), transitions=TRANSITIONS,
                    batch_size=batch_size, batches=batches,
                    pool_counts={k: len(v) for k, v in pools.items()},
                    task_scope='Instruction-filtered training demonstrations, not simulator rollouts.',
                    model_mode='train; checkpoint trainable mask; paired augmentation and dropout seeds',
                    gradients='Actual raw loss parameter gradients, before clipping, Adam, or weight decay.',
                    channels='Exact training action mask, including gripper; joint Student-t.',
                    comparisons='nu_before vs 0.9*nu_before vs nu_after; same-nu independent-dropout control on mixed_0',
                    updates=0)
    out.mkdir(parents=True, exist_ok=True)
    dest = out/'protocol.json'
    if dest.exists():
        assert json.loads(dest.read_text()) == json.loads(json.dumps(protocol))
    else:
        dest.write_text(json.dumps(protocol, indent=2)+'\n')


def parameter_group(name):
    if 'action_head.action_decoder.' in name:
        return 'action_decoder'
    if 'action_head.sigma_decoder.' in name:
        return 'sigma_decoder'
    if name.startswith('action_head.'):
        return 'shared_action_head'
    return 'backbone'


def compare(reference, candidate, names):
    sums = {g: [0., 0., 0.] for g in ['all', 'action_decoder', 'sigma_decoder', 'shared_action_head', 'backbone']}
    # Float64 reductions in bounded chunks avoid allocating float64 full gradients.
    for name, a, b in zip(names, reference, candidate):
        for offset in range(0, a.numel(), 1_000_000):
            x = a.reshape(-1)[offset:offset+1_000_000].double()
            y = b.reshape(-1)[offset:offset+1_000_000].double()
            vals = [float(x.dot(x)), float(y.dot(y)), float(x.dot(y))]
            for key in ['all', parameter_group(name)]:
                sums[key] = [u+v for u, v in zip(sums[key], vals)]
    result = {}
    for key, (aa, bb, ab) in sums.items():
        if aa > 0 and bb > 0:
            cosine = float(np.clip(ab / np.sqrt(aa*bb), -1, 1))
            result[key] = dict(cosine=cosine, angle_degrees=float(np.degrees(np.arccos(cosine))),
                               norm_before=np.sqrt(aa), norm_after=np.sqrt(bb), norm_ratio=np.sqrt(bb/aa),
                               relative_difference=np.sqrt(max(0., aa+bb-2*ab)/aa))
        else:
            result[key] = dict(norm_before=np.sqrt(aa), norm_after=np.sqrt(bb), cosine=None)
    return result


def run(args):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType

    torch.set_num_threads(4)
    meta = json.loads((args.output/'protocol.json').read_text())
    tag = EmbodimentTag.resolve('SIMPLER_ENV_GOOGLE')
    transitions = [t for t in TRANSITIONS if t[0] in args.checkpoint_steps]
    for step, old, new in transitions[args.rank::args.world_size]:
        dest = args.output/f'checkpoint_{step}.json'
        if dest.exists():
            continue
        start = time.monotonic()
        seed_all(SEED)
        model, proc = load_model(RUN/f'checkpoint-{step}', 'hetero_t')
        cfg = model.action_head.config
        assert cfg.ht_mvt and cfg.ht_df == old, (step, cfg.ht_df, old)
        assert cfg.ht_beta == 0 and not cfg.ht_shrink and cfg.ht_hg_steps == 0 and cfg.ht_mse_steps == 0
        assert not getattr(cfg, 'ht_gripper_bce', False)
        assert not getattr(cfg, 'ht_gate_floor', 0.) and getattr(cfg, 'ht_sigma_clamp', -1.) < 0
        model.train()
        proc.train()
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        names, params = zip(*named)
        assert all(n.startswith('action_head.') for n in names), 'Unexpected trainable backbone'
        result = dict(step=step, nu_before=old, nu_after=new, batches=[],
                      trainable_parameters=sum(p.numel() for p in params),
                      parameter_dtypes=sorted(set(str(p.dtype) for p in params)),
                      micro_batch=args.micro_batch,
                      parameter_groups={g: sum(p.numel() for n, p in named if parameter_group(n) == g)
                                        for g in ['action_decoder','sigma_decoder','shared_action_head','backbone']})
        (args.output/f'parameters_{step}.json').write_text(json.dumps({n: list(p.shape) for n,p in named}))
        mods = proc.modality_configs[tag.value]
        loader = LeRobotEpisodeLoader(ROOT/'fractal_lerobot', mods)
        captured = {}
        model.action_head.action_decoder.register_forward_hook(lambda m, i, o: captured.__setitem__('prediction', o))
        model.action_head.sigma_decoder.register_forward_hook(lambda m, i, o: captured.__setitem__('raw_sigma', o))
        print('MODEL_READY', step, result['parameter_groups'], flush=True)
        for bi, (batch_name, records) in enumerate(meta['batches'].items()):
            processed = []
            for mi in range(0, len(records), args.micro_batch):
                features = []
                for j, ep in enumerate(records[mi:mi+args.micro_batch]):
                    seed_all(SEED+bi*10000+mi+j)
                    episode = loader[ep['ordinal']]
                    item = extract_step_data(episode, ep['step'], mods, tag, allow_padding=False)
                    features.append(proc([{'type': MessageType.EPISODE_STEP.value, 'content': item}]))
                batch = proc.collator(features)
                inner = batch.get('inputs', batch)
                assert (inner['action_mask'].sum((1,2)) == 56).all()
                processed.append(inner)
                if (mi+args.micro_batch) % 128 == 0:
                    print('INPUTS_READY',step,batch_name,mi+args.micro_batch,flush=True)
            denom = sum(float(x['action_mask'].sum()) for x in processed)
            reference = None
            reference_outputs = None
            settings = [('before', old, 0), ('small_drop', old*.9, 0), ('halving', new, 0)]
            if batch_name == 'mixed_0':
                settings.append(('dropout_control', old, 1000000))
            for setting, nu, seed_offset in settings:
                model.zero_grad(set_to_none=True)
                raw_rows, predictions, scales, losses = [], [], [], []
                source_difference = 0.
                for mi, batch in enumerate(processed):
                    seed_all(SEED+bi*10000+mi+seed_offset)
                    inner = map_tensors(batch, lambda t: t.cuda())
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        native = model(inner)
                        target = inner['action'].float()
                        mask = inner['action_mask'].float()
                        pred = captured['prediction'][:, -target.shape[1]:].float()
                        raw = captured['raw_sigma'][:, -target.shape[1]:].float()
                        d = mask.sum((1,2))
                        s = ((pred-target).square()*mask).sum((1,2))
                        sigma = (torch.nn.functional.softplus(raw+cfg.ht_sbias)*mask).sum((1,2))/d+.001
                        q = s/sigma.square()
                        loss = (.5*(nu+d)*torch.log1p(q/nu)+d*torch.log(sigma)).sum()/denom
                        if setting == 'before':
                            check = loss.detach() * denom / mask.sum()
                            diff = float((check-native['loss'].detach()).abs())
                            source_difference = max(source_difference, diff)
                            assert diff < 2e-5, ('Native loss mismatch', diff)
                    loss.backward()
                    losses.append(float(loss.detach()))
                    predictions.append(pred.detach().cpu().numpy())
                    scales.append(sigma.detach().cpu().numpy())
                    raw_rows.append(dict(s=s.detach().cpu().numpy(), sigma=sigma.detach().cpu().numpy(),
                                         q=q.detach().cpu().numpy(), d=d.detach().cpu().numpy()))
                    captured.clear()
                    del native, inner, target, mask, pred, raw, d, s, sigma, q, loss
                    if (mi+1)*args.micro_batch % 128 == 0:
                        print('BACKWARD_PROGRESS',step,batch_name,setting,(mi+1)*args.micro_batch,flush=True)
                grads = [torch.zeros(p.shape, dtype=torch.float32) if p.grad is None
                         else p.grad.detach().to(device='cpu', dtype=torch.float32).clone() for p in params]
                model.zero_grad(set_to_none=True)
                assert all(torch.isfinite(g).all() for g in grads)
                outputs = (np.concatenate(predictions), np.concatenate(scales))
                stats = {k: np.concatenate([r[k] for r in raw_rows]) for k in raw_rows[0]}
                stats['gate'] = (nu+stats['d'])/(nu+stats['q'])
                np.savez_compressed(args.output/f'raw_{step}_{batch_name}_{setting}.npz', **stats,
                                    prediction=outputs[0], episode=np.array([r['episode_index'] for r in records]),
                                    step=np.array([r['step'] for r in records]), nu=nu)
                if reference is None:
                    reference = grads
                    reference_outputs = outputs
                    result['batches'].append(dict(batch=batch_name, loss_before=sum(losses),
                                                   native_loss_max_difference=source_difference, comparisons={}))
                else:
                    if seed_offset == 0:
                        for a,b in zip(reference_outputs, outputs):
                            np.testing.assert_array_equal(a,b)
                    measured = compare(reference, grads, names)
                    result['batches'][-1]['comparisons'][setting] = dict(nu=nu, loss=sum(losses), groups=measured)
                    print('GRADIENT_RESULT', step, batch_name, setting, json.dumps(measured), flush=True)
                    del grads
                (args.output/f'partial_{step}.json').write_text(json.dumps(result, indent=2))
                print('PASS_DONE',step,batch_name,setting,'elapsed',round(time.monotonic()-start),flush=True)
            del reference, processed
            gc.collect()
        result['elapsed_seconds'] = time.monotonic()-start
        dest.write_text(json.dumps(result, indent=2)+'\n')
        del named, names, params, model, proc, loader
        gc.collect()
        torch.cuda.empty_cache()
        print('CHECKPOINT_DONE', step, result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--micro-batch', type=int, default=2)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=1)
    p.add_argument('--cohorts', nargs='+', default=['mixed_0','mixed_1','move_near','close_drawer'])
    p.add_argument('--checkpoint-steps', nargs='+', type=int, default=[t[0] for t in TRANSITIONS])
    args = p.parse_args()
    if args.prepare_only:
        prepare(args.output, args.batch_size, args.cohorts)
    else:
        run(args)
