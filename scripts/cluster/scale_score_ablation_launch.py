"""Three fresh Fractal arms; process-local replacement loss, no shared edits.

Read scale_score_recipe.json before resuming training or probing sigma.
Vanilla deterministic action inference remains valid; vanilla sigma dumps do NOT
understand the exp parameterization. This entrypoint only accepts fresh starts.
"""
import json
import math
import os
from pathlib import Path
import runpy

NU = 14.0
D = 56
K = 16
SBIAS = -.5093
EPS = .001
SIGMA0 = math.log1p(math.exp(SBIAS)) + EPS
LOG_SIGMA0 = math.log(SIGMA0)
ARMS = {'se_exp', 'se_softplus', 'ht_exp_gaussaux1'}


def kappa(d=D, nu=NU):
    return math.exp(.5*math.log(nu) + math.lgamma((d+1)/2)-math.lgamma(d/2)
                    + math.lgamma(nu-.5)-math.lgamma(nu-1)
                    + 2*math.lgamma((nu-1)/2)-2*math.lgamma(nu/2))


def scale_from_raw(raw, mask, arm):
    import torch
    assert arm in ARMS
    raw, mask = raw.float(), mask.float()
    d = mask.sum((1, 2))
    if arm == 'se_softplus':
        sigma = (torch.nn.functional.softplus(raw + SBIAS)*mask).sum((1, 2))/d + EPS
        log_sigma = sigma.log()
    else:
        # Pool log-scale FIRST, then exp. No hard clamp or additive floor.
        log_sigma = (raw*mask).sum((1, 2))/d + LOG_SIGMA0
        sigma = log_sigma.exp()
    return sigma, log_sigma, d


def sample_joint_t(batch, device, generator, count=K):
    import torch
    # nu=14 is an integer: chi-square is exactly a sum of 14 squared normals.
    # One independent denominator per whole d-vector, NOT per coordinate.
    g = torch.randn(count, batch, D, device=device, generator=generator)
    chi2 = torch.randn(count, batch, int(NU), device=device, generator=generator).square().sum(-1)
    return g / (chi2/NU).sqrt().unsqueeze(-1)


def objective(pred, raw, target, mask, arm, samples=None):
    import torch
    sigma, log_sigma, d = scale_from_raw(raw, mask, arm)
    # Route all likelihood terms through log_sigma, including softplus arm,
    # so the logged autograd dL/dlog_sigma contains the entire derivative.
    sigma = log_sigma.exp()
    assert (d == D).all(), 'This experiment requires exactly 56 valid dimensions.'
    assert ((mask == 0) | (mask == 1)).all()
    residual = (pred.float()-target.float()).masked_select(mask.bool()).reshape(len(d), D)
    squared = residual.square().sum(-1)
    q = squared / sigma.square()
    ht = .5*(NU+d)/d * torch.log1p(q/NU) + log_sigma
    aux = .5*squared.detach()/(d*sigma.square()) + log_sigma
    if arm.startswith('se_'):
        assert samples is not None and samples.shape[1:] == residual.shape
        z = -residual / sigma[:, None]
        per = (2/kappa())*torch.linalg.vector_norm(z[None]-samples, dim=-1).mean(0) + log_sigma
        # log(kappa) is omitted: nu,d fixed. This is NOT divided by d again.
    else:
        per = ht + aux
    return per.mean(), dict(per=per, sigma=sigma, log_sigma=log_sigma,
                            squared=squared, q=q, d=d, ht=ht, aux=aux)


def main():
    import torch
    from transformers import TrainerCallback
    from huggingface_hub import snapshot_download
    from gr00t.experiment.trainer import Gr00tTrainer
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing
    from nu14_gaussian_aux_launch import validate_config

    arm = os.environ['SCALE_SCORE_ARM']
    assert arm in ARMS and os.environ.get('GROOT_LEARN_NU', '0') == '0'
    original_builder = processing.build_processor
    def cached_processor(name, kwargs):
        local = snapshot_download(name, local_files_only=True) if not name.startswith('/') else name
        return original_builder(local, kwargs)
    processing.build_processor = cached_processor

    class Replacement(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            assert state.global_step == 0
            assert args.max_steps == 20000 and args.get_warmup_steps(20000) == 1000
            assert args.learning_rate == 1e-4 and args.weight_decay == 1e-5
            assert args.world_size == 8 and args.per_device_train_batch_size == 128
            assert args.gradient_accumulation_steps == 1
            assert str(args.optim).lower().endswith('adamw_torch')
            core = model.module if hasattr(model, 'module') else model
            head = core.action_head
            assert not hasattr(head, 'nu_decoder')
            # Same constant sigma for every example in all three new arms.
            # Do NOT zero layer1: layer2 gets gradients immediately, layer1/trunk
            # via sigma starts after layer2's first update. Mu path is untouched.
            with torch.no_grad():
                head.sigma_decoder.layer2.W.zero_()
                head.sigma_decoder.layer2.b.zero_()
            recipe = dict(arm=arm, nu=NU, d=D, samples=K if arm.startswith('se_') else 0,
                          kappa=kappa(), auxiliary_weight=1 if arm.startswith('ht_') else 0,
                          sigma_initial=SIGMA0, log_sigma_initial=LOG_SIGMA0,
                          softplus_bias=SBIAS, softplus_floor=EPS, exp_floor=0,
                          initialization='zero sigma_decoder.layer2 W and b in ALL three arms',
                          scale='masked mean of softplus + epsilon' if arm=='se_softplus' else 'exp(masked mean raw + log_sigma_initial)',
                          start='fresh optimizer, original pretrained GR00T; not full-model random init',
                          score='scaled energy (log kappa omitted)' if arm.startswith('se_') else 'per-dim joint HT NLL + detached-mu Gaussian NLL',
                          noise='iid joint t vectors; dedicated rank-local RNG, no model-input noise',
                          seed=int(args.seed), updates=20000,
                          inference='vanilla deterministic mu works; sigma probes and training require this wrapper',
                          reference_caveat='old softplus+aux run did not zero sigma final layer; new three arms match each other exactly')
            for cfg in (core.config, head.config):
                validate_config(cfg)
                assert float(cfg.ht_sbias) == SBIAS
                cfg.ht_scale_score_ablation = recipe
            captured = {}
            head.action_decoder.register_forward_hook(
                lambda module, inputs, output: captured.__setitem__('prediction', output))
            head.sigma_decoder.register_forward_hook(
                lambda module, inputs, output: captured.__setitem__('raw', output))
            out = Path(args.output_dir)
            self.last_logged = None
            self.first = True
            self.generator = None

            def replace_loss(module, inputs, output):
                batch = inputs[1]
                if 'loss' not in output or not hasattr(batch, 'action'):
                    captured.clear()
                    return output
                target, mask = batch.action, batch.action_mask
                pred = captured.pop('prediction')[:, -target.shape[1]:]
                raw = captured.pop('raw')[:, -target.shape[1]:]
                if self.generator is None:
                    self.generator = torch.Generator(device=pred.device)
                    self.generator.manual_seed(314159 + int(args.process_index))
                with torch.autocast(device_type=pred.device.type, enabled=False):
                    samples = sample_joint_t(len(pred), pred.device, self.generator) if arm.startswith('se_') else None
                    loss, info = objective(pred, raw, target, mask, arm, samples)
                assert torch.isfinite(loss), 'Nonfinite objective; stop instead of silently clamping exp.'
                if self.first:
                    torch.testing.assert_close(raw, torch.zeros_like(raw), atol=0, rtol=0)
                    torch.testing.assert_close(info['sigma'], torch.full_like(info['sigma'], SIGMA0), atol=1e-7, rtol=1e-6)
                    # At initialization both parameterizations exactly equal the native scale.
                    torch.testing.assert_close(output['loss'].float(), info['ht'].mean(), atol=2e-5, rtol=2e-5)
                    self.first = False
                    if state.is_world_process_zero:
                        (out/'initial_batch_verified.json').write_text(json.dumps(dict(step=state.global_step, sigma=SIGMA0, recipe=recipe), indent=2))
                output['loss'] = loss
                output['action_loss'] = info['per']*info['d']
                step = state.global_step
                if state.is_world_process_zero and step % 100 == 0 and step != self.last_logged:
                    # Per-example gradients, NOT gradients divided by batch size.
                    gs, gp, gr = torch.autograd.grad(info['per'].sum(), (info['log_sigma'], pred, raw), retain_graph=True)
                    with torch.no_grad():
                        quant = lambda x: torch.quantile(x.detach().float(), torch.tensor([.1,.5,.9], device=x.device)).cpu().tolist()
                        row = dict(step=step, arm=arm, loss=float(loss), nu=NU,
                                   sigma=quant(info['sigma']), log_sigma=quant(info['log_sigma']),
                                   residual_rms=quant((info['squared']/D).sqrt()), q_over_d=quant(info['q']/D),
                                   mse=float(info['squared'].mean()/D), ht_loss=float(info['ht'].mean()),
                                   hg_aux_diagnostic=float(info['aux'].mean()),
                                   log_sigma_grad_abs_mean=float(gs.abs().mean()),
                                   log_sigma_grad_mean=float(gs.mean()),
                                   prediction_grad_norm_mean=float(gp.float().flatten(1).norm(dim=1).mean()),
                                   raw_sigma_grad_norm_mean=float(gr.float().flatten(1).norm(dim=1).mean()))
                    with (out/'training_diagnostics.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                    print('SCALE_SCORE_DIAGNOSTICS', json.dumps(row), flush=True)
                    self.last_logged = step
                return output

            head.register_forward_hook(replace_loss)
            if state.is_world_process_zero:
                (out/'scale_score_recipe.json').write_text(json.dumps(recipe, indent=2))
                print('SCALE_SCORE_READY', json.dumps(recipe), flush=True)

    original_init = Gr00tTrainer.__init__
    original_train = Gr00tTrainer.train
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(Replacement())
    def train(self, resume_from_checkpoint=None, **kwargs):
        assert resume_from_checkpoint in (None, False), 'Fresh-start ablation only.'
        return original_train(self, resume_from_checkpoint=None, **kwargs)
    Gr00tTrainer.__init__ = initialize
    Gr00tTrainer.train = train
    runpy.run_module('gr00t.experiment.launch_finetune', run_name='__main__')


if __name__ == '__main__':
    main()
