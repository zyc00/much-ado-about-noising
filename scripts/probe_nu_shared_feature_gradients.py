"""No-update detach audit at the actual shared input of the two GR00T decoders.

Train-mode features are captured once per microbatch. Actual nonlinear decoders
are replayed in FP32 to isolate branch gradients without BF16 reduction error.
All nu values therefore use exactly the same features, targets, and heads.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from probe_nu_gradient_switch import ROOT, RUN, SEED, seed_all
from probe_widowx_general_scale import load_model, map_tensors


def objective(mu, sigma, target, mask, nu):
    d = mask.sum((1, 2))
    q = ((mu - target).square() * mask).sum((1, 2)) / sigma.square()
    if nu is None:
        return .5 * q / d + sigma.log()
    return .5 * (nu + d) / d * torch.log1p(q / nu) + sigma.log()


def branch_grads(mu, sigma, x, target, mask, nu):
    gm = torch.autograd.grad(objective(mu, sigma.detach(), target, mask, nu).sum(),
                             x, retain_graph=True)[0]
    gs = torch.autograd.grad(objective(mu.detach(), sigma, target, mask, nu).sum(),
                             x, retain_graph=True)[0]
    both = torch.autograd.grad(objective(mu, sigma, target, mask, nu).sum(),
                               x, retain_graph=True)[0]
    torch.testing.assert_close(both, gm + gs, atol=2e-6, rtol=2e-4)
    return gm, gs


def stats(gm, gs):
    m, s = gm.detach().flatten(1).double(), gs.detach().flatten(1).double()
    nm, ns = m.norm(dim=1), s.norm(dim=1)
    dot = (m * s).sum(1)
    return dict(mu_norm=nm.cpu().numpy(), sigma_norm=ns.cpu().numpy(),
                dot=dot.cpu().numpy(), combined_norm=(m+s).norm(dim=1).cpu().numpy(),
                cosine=(dot / (nm * ns).clamp_min(1e-30)).cpu().numpy())


def summarize(raw):
    nm, ns = raw['mu_norm'], raw['sigma_norm']
    total = raw['combined_norm']
    return dict(mu_norm_mean=float(nm.mean()), sigma_norm_mean=float(ns.mean()),
                sigma_over_mu_norm_median=float(np.median(ns / nm)),
                sigma_over_mu_norm_sum=float(ns.sum() / nm.sum()),
                mu_stacked_norm=float(np.linalg.norm(nm)),
                sigma_stacked_norm=float(np.linalg.norm(ns)),
                combined_stacked_norm=float(np.linalg.norm(total)),
                cosine_median=float(np.median(raw['cosine'])),
                cosine_negative_fraction=float(np.mean(raw['cosine'] < 0)),
                combined_over_sum_norms=float(total.sum() / (nm + ns).sum()))


def run(args):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads(args.protocol.read_text())
    records = protocol['batches']['mixed_0'][:args.samples]
    assert len(records) == args.samples
    args.output.mkdir(parents=True, exist_ok=True)
    nus = [None, 1024., 448., 224., 128., 64., 32., 14., 7.]
    manifest = dict(source_protocol=str(args.protocol), records=records, nus=nus,
                    checkpoints=args.steps, updates=0, feature='model_output shared by both actual decoders',
                    mode='train-mode BF16 upstream; decoder replay FP32; same features for all nu',
                    norm='Per-example gradient of per-dimension mean NLL w.r.t. all shared feature tokens; no batch averaging',
                    scope='Training demonstrations; not held-out success evaluation; includes training gripper mask')
    (args.output / 'protocol.json').write_text(json.dumps(manifest, indent=2))
    for step in args.steps:
        dest = args.output / f'checkpoint_{step}.json'
        if dest.exists():
            raise FileExistsError(dest)
        seed_all(SEED)
        model, proc = load_model(RUN / f'checkpoint-{step}', 'hetero_t')
        model.train()
        proc.train()
        cfg = model.action_head.config
        assert cfg.ht_mvt and not getattr(cfg, 'ht_gripper_bce', False)
        tag = EmbodimentTag.resolve('SIMPLER_ENV_GOOGLE')
        mods = proc.modality_configs[tag.value]
        loader = LeRobotEpisodeLoader(ROOT / 'fractal_lerobot', mods)
        captured = {}
        def hook(name):
            def capture(module, inputs, output):
                captured[name] = (inputs, output)
            return capture
        handles = [model.action_head.action_decoder.register_forward_hook(hook('mu')),
                   model.action_head.sigma_decoder.register_forward_hook(hook('sigma'))]
        collected = {str(nu): [] for nu in nus}
        q_rows, sigma_rows, forward_errors = [], [], []
        for mi in range(0, len(records), args.micro_batch):
            features = []
            for j, ep in enumerate(records[mi:mi+args.micro_batch]):
                seed_all(SEED + mi + j)
                episode = loader[ep['ordinal']]
                item = extract_step_data(episode, ep['step'], mods, tag, allow_padding=False)
                features.append(proc([{'type': MessageType.EPISODE_STEP.value, 'content': item}]))
            batch = proc.collator(features)
            inner = map_tensors(batch.get('inputs', batch), lambda t: t.cuda())
            seed_all(SEED + mi // args.micro_batch)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                native = model(inner)
            mu_inputs, native_mu = captured['mu']
            sigma_inputs, native_sigma = captured['sigma']
            assert mu_inputs[0] is sigma_inputs[0], 'Heads do not share the same feature'
            x = mu_inputs[0].detach().float().requires_grad_(True)
            embodiment = mu_inputs[1]
            target, mask = inner['action'].float(), inner['action_mask'].float()
            d = mask.sum((1, 2))
            assert (d == 56).all()
            mu = model.action_head.action_decoder(x, embodiment)[:, -target.shape[1]:].float()
            raw = model.action_head.sigma_decoder(x, embodiment)[:, -target.shape[1]:].float()
            sigma = (torch.nn.functional.softplus(raw + cfg.ht_sbias) * mask).sum((1, 2)) / d + .001
            # Record the small precision-only change introduced by FP32 head replay.
            native_pred = native_mu[:, -target.shape[1]:].float()
            native_raw = native_sigma[:, -target.shape[1]:].float()
            native_scale = (torch.nn.functional.softplus(native_raw + cfg.ht_sbias) * mask).sum((1, 2)) / d + .001
            reconstructed = objective(native_pred, native_scale, target, mask, cfg.ht_df).mean()
            torch.testing.assert_close(reconstructed, native['loss'].float(), atol=2e-5, rtol=2e-5)
            forward_errors.append(float((((mu-native_pred).square()*mask).sum()/mask.sum()).sqrt().detach()))
            q = (((mu-target).square()*mask).sum((1, 2)) / sigma.square()).detach()
            q_rows.append(q.cpu().numpy())
            sigma_rows.append(sigma.detach().cpu().numpy())
            hg_mu, hg_sigma = branch_grads(mu, sigma, x, target, mask, None)
            for nu in nus:
                gm, gs = (hg_mu, hg_sigma) if nu is None else branch_grads(mu, sigma, x, target, mask, nu)
                if nu is not None:
                    shape = (-1,) + (1,) * (x.ndim-1)
                    torch.testing.assert_close(gm, hg_mu*((nu+d)/(nu+q)).reshape(shape), atol=2e-6, rtol=3e-4)
                    torch.testing.assert_close(gs, hg_sigma*(nu/(nu+q)).reshape(shape), atol=2e-6, rtol=3e-4)
                collected[str(nu)].append(stats(gm, gs))
            captured.clear()
            print('BATCH_DONE', step, mi + len(features), flush=True)
            del native, mu, raw, sigma, hg_mu, hg_sigma, gm, gs, x, inner
        result = dict(checkpoint=step, samples=len(records), d=56,
                      native_nu=cfg.ht_df, fp32_replay_prediction_rms_difference_max=max(forward_errors),
                      gradient_identity_tests='branch sum and both HG scaling identities passed', settings={})
        raw_save = dict(q=np.concatenate(q_rows), sigma=np.concatenate(sigma_rows))
        for nu in nus:
            key = str(nu)
            raw = {k: np.concatenate([row[k] for row in collected[key]]) for k in collected[key][0]}
            result['settings']['HG' if nu is None else str(int(nu))] = summarize(raw)
            raw_save.update({f'{key}_{k}': v for k, v in raw.items()})
        np.savez_compressed(args.output / f'checkpoint_{step}_raw.npz', **raw_save)
        dest.write_text(json.dumps(result, indent=2))
        print('CHECKPOINT_DONE', json.dumps(result), flush=True)
        for handle in handles:
            handle.remove()
        del model, proc, loader, handles, mu_inputs, sigma_inputs, captured
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', nargs='+', type=int, default=[10000, 12000])
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--micro-batch', type=int, default=8)
    run(parser.parse_args())
