"""Fresh Fractal HT nu=14 + Gaussian scale auxiliary, detached mean.

Process-local loss hook; shared model/trainer files and old runs are untouched.
Checkpoints retain inference-compatible modules, and record the auxiliary recipe
in model configs. To resume TRAINING this objective, use this loss wrapper, not
the vanilla launcher (this entrypoint intentionally only allows fresh starts).
"""
import json
import os
from pathlib import Path
import runpy

NU = 14.0
AUX_WEIGHT = 1.0


def gaussian_scale_aux(prediction, raw_sigma, target, mask, sbias):
    import torch
    mask = mask.float()
    d = mask.sum((1, 2))
    sigma = (torch.nn.functional.softplus(raw_sigma.float() + sbias) * mask).sum((1, 2)) / d + .001
    squared = ((prediction.detach().float() - target.float()).square() * mask).sum((1, 2))
    loss = (.5 * squared / sigma.square() + d * sigma.log()).sum() / d.sum()
    return loss, sigma, squared, d


def validate_config(cfg):
    assert cfg.loss_type == 'hetero_t' and cfg.ht_mvt and cfg.ht_df == NU
    for key in ('ht_beta', 'ht_shrink', 'ht_gripper_bce', 'ht_mse_steps',
                'ht_hg_steps', 'ht_nu_start', 'ht_nu_target', 'ht_gate_floor'):
        assert not getattr(cfg, key, 0), (key, getattr(cfg, key))
    assert getattr(cfg, 'ht_sigma_clamp', -1) < 0


def main():
    import torch
    from transformers import TrainerCallback
    from huggingface_hub import snapshot_download
    from gr00t.experiment.trainer import Gr00tTrainer
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing

    original_builder = processing.build_processor
    def cached_processor(name, kwargs):
        local = snapshot_download(name, local_files_only=True) if not name.startswith('/') else name
        return original_builder(local, kwargs)
    processing.build_processor = cached_processor
    assert os.environ.get('GROOT_LEARN_NU', '0') == '0'

    class Auxiliary(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            assert state.global_step == 0
            assert args.max_steps == 20000 and args.get_warmup_steps(20000) == 1000
            assert args.learning_rate == 1e-4 and args.weight_decay == 1e-5
            assert args.world_size == 8 and args.per_device_train_batch_size == 128
            assert args.gradient_accumulation_steps == 1
            assert str(args.optim).lower().endswith('adamw_torch')
            core = model.module if hasattr(model, 'module') else model
            assert not hasattr(core.action_head, 'nu_decoder')
            recipe = dict(nu=NU, gaussian_aux_weight=AUX_WEIGHT,
                          detached='prediction in Gaussian auxiliary only',
                          sigma='native masked-mean softplus + 0.001, shared between both losses',
                          reduction='each loss normalized by total valid action dimensions',
                          start='fresh optimizer and step 0 from original pretrained GR00T',
                          updates=20000, online_gradient_balancing=False,
                          inference='unchanged deterministic mean; loss wrapper required for training')
            for cfg in (core.config, core.action_head.config):
                validate_config(cfg)
                cfg.ht_gaussian_scale_aux = recipe
            captured = {}
            core.action_head.action_decoder.register_forward_hook(
                lambda module, inputs, output: captured.__setitem__('prediction', output))
            core.action_head.sigma_decoder.register_forward_hook(
                lambda module, inputs, output: captured.__setitem__('sigma', output))
            out = Path(args.output_dir)
            self.last_logged = None

            def add_aux(head, inputs, output):
                validate_config(head.config)
                batch = inputs[1]
                if 'loss' not in output or not hasattr(batch, 'action'):
                    captured.clear()
                    return output
                target = batch.action
                mask = batch.action_mask
                pred = captured.pop('prediction')[:, -target.shape[1]:]
                raw = captured.pop('sigma')[:, -target.shape[1]:]
                aux, sigma, squared, d = gaussian_scale_aux(pred, raw, target, mask, head.config.ht_sbias)
                native = output['loss']
                total = native + AUX_WEIGHT * aux
                # Store only the combined loss in the ordinary model output.
                # Native action_loss is a diagnostic, not used by Trainer.
                output['loss'] = total
                step = state.global_step
                if state.is_world_process_zero and step % 100 == 0 and step != self.last_logged:
                    with torch.no_grad():
                        assert (d == 56).all()
                        q = squared / sigma.square()
                        reconstructed = (.5*(NU+d)*torch.log1p(q/NU)+d*sigma.log()).sum()/d.sum()
                        torch.testing.assert_close(native.float(), reconstructed, atol=2e-5, rtol=2e-5)
                        assert torch.isfinite(total)
                        quant = lambda x: torch.quantile(x.float(), torch.tensor([.1,.5,.9],device=x.device)).cpu().tolist()
                        g_hg = 1-q/d
                        g_ht = NU/(NU+q)*g_hg
                        row = dict(step=step, rank=0, batch=len(d), nu=NU, auxiliary_weight=AUX_WEIGHT,
                                   ht_loss=float(native), gaussian_aux_loss=float(aux), total_loss=float(total),
                                   mse_per_dim=float(squared.sum()/d.sum()), sigma=quant(sigma),
                                   residual_rms=quant((squared/d).sqrt()), q_over_d=quant(q/d),
                                   log_sigma_ht_gradient_abs_mean=float(g_ht.abs().mean()),
                                   log_sigma_aux_gradient_abs_mean=float(g_hg.abs().mean()),
                                   log_sigma_total_gradient_mean=float((g_ht+AUX_WEIGHT*g_hg).mean()),
                                   log_sigma_total_gradient_abs_mean=float((g_ht+AUX_WEIGHT*g_hg).abs().mean()))
                    with (out/'training_diagnostics.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                    print('NU14_GAUSSIAN_AUX', json.dumps(row), flush=True)
                    self.last_logged = step
                return output

            core.action_head.register_forward_hook(add_aux)
            if state.is_world_process_zero:
                (out/'fresh_start_verified.json').write_text(json.dumps(dict(step=state.global_step, recipe=recipe), indent=2))
                print('FRESH_NU14_AUX_READY', json.dumps(recipe), flush=True)

    original_init = Gr00tTrainer.__init__
    original_train = Gr00tTrainer.train
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(Auxiliary())
    def train(self, resume_from_checkpoint=None, **kwargs):
        assert resume_from_checkpoint in (None, False), 'This experiment must start fresh.'
        return original_train(self, resume_from_checkpoint=None, **kwargs)
    Gr00tTrainer.__init__ = initialize
    Gr00tTrainer.train = train
    runpy.run_module('gr00t.experiment.launch_finetune', run_name='__main__')


if __name__ == '__main__':
    main()
