"""Fresh Fractal HT nu=224 with an exponential scale parameterization.

This is an isolated, process-local loss replacement.  Relative to the
ft_fr_nu224 recipe, only the map from the sigma decoder output to sigma is
changed.  Deterministic inference still uses only the mean action.
"""
import json
import math
import os
from pathlib import Path
import runpy

NU = 224.0
D = 56
SBIAS = -0.5093
EPS = 0.001
SIGMA0 = math.log1p(math.exp(SBIAS)) + EPS
LOG_SIGMA0 = math.log(SIGMA0)


def scales_from_raw(raw, mask):
    import torch
    raw, mask = raw.float(), mask.float()
    d = mask.sum((1, 2))
    pooled_raw = (raw * mask).sum((1, 2)) / d
    # Calibrated so raw=0 gives exactly the original softplus initialization.
    log_sigma = pooled_raw + LOG_SIGMA0
    sigma = log_sigma.exp()
    original_sigma = (
        torch.nn.functional.softplus(raw + SBIAS) * mask
    ).sum((1, 2)) / d + EPS
    return sigma, log_sigma, original_sigma, d


def exp_ht_objective(prediction, raw_sigma, target, mask):
    import torch
    sigma, log_sigma, original_sigma, d = scales_from_raw(raw_sigma, mask)
    residual = prediction.float() - target.float()
    squared = (residual.square() * mask.float()).sum((1, 2))
    q = squared / sigma.square()
    per = 0.5 * (NU + d) / d * torch.log1p(q / NU) + log_sigma
    return per.mean(), dict(
        per=per, sigma=sigma, log_sigma=log_sigma,
        original_sigma=original_sigma, squared=squared, q=q, d=d,
    )


def validate_config(cfg):
    assert cfg.loss_type == "hetero_t" and cfg.ht_mvt
    assert float(cfg.ht_df) == NU
    assert float(cfg.ht_sbias) == SBIAS
    for key in (
        "ht_beta", "ht_shrink", "ht_gripper_bce", "ht_mse_steps",
        "ht_hg_steps", "ht_nu_start", "ht_nu_target", "ht_gate_floor",
    ):
        assert not getattr(cfg, key, 0), (key, getattr(cfg, key))
    assert getattr(cfg, "ht_sigma_clamp", -1) < 0


def main():
    import torch
    from transformers import TrainerCallback
    from huggingface_hub import snapshot_download
    from gr00t.experiment.trainer import Gr00tTrainer
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing

    assert os.environ.get("GROOT_LEARN_NU", "0") == "0"
    original_builder = processing.build_processor

    def cached_processor(name, kwargs):
        local = snapshot_download(name, local_files_only=True) if not name.startswith("/") else name
        return original_builder(local, kwargs)

    processing.build_processor = cached_processor

    class ExpScale(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            assert state.global_step == 0
            assert args.max_steps == 20000 and args.get_warmup_steps(20000) == 1000
            assert args.learning_rate == 1e-4 and args.weight_decay == 1e-5
            assert args.world_size == 8 and args.per_device_train_batch_size == 128
            assert args.gradient_accumulation_steps == 1
            assert str(args.optim).lower().endswith("adamw_torch")
            core = model.module if hasattr(model, "module") else model
            head = core.action_head
            assert not hasattr(head, "nu_decoder")

            # Do not zero or otherwise alter decoder parameters: use the same
            # seeded initialization as the original recipe.  Only calibrate the
            # exp offset so a zero raw output maps to the old initial sigma.
            recipe = dict(
                objective="per-dimension joint Student-t NLL",
                nu=NU, d=D,
                scale="exp(masked_mean(raw_sigma) + log_sigma0)",
                sigma0=SIGMA0, log_sigma0=LOG_SIGMA0,
                reference_scale="masked_mean(softplus(raw_sigma + sbias)) + floor",
                reference_sbias=SBIAS, reference_floor=EPS,
                initialization=(
                    "original seeded sigma-decoder initialization retained; "
                    "raw=0 maps exactly to reference sigma0"
                ),
                auxiliary_loss=None, learn_nu=False,
                start="fresh optimizer and original pretrained GR00T",
                updates=20000,
                evaluation=(
                    "six Fractal tasks, seed=1234, 100 rollouts/task, "
                    "5 envs, NAS=1, max_steps=300"
                ),
                inference="unchanged deterministic mean",
            )
            for cfg in (core.config, head.config):
                validate_config(cfg)
                cfg.ht_exp_scale_recipe = recipe

            captured = {}
            head.action_decoder.register_forward_hook(
                lambda module, inputs, output: captured.__setitem__("prediction", output)
            )
            head.sigma_decoder.register_forward_hook(
                lambda module, inputs, output: captured.__setitem__("raw", output)
            )
            out_dir = Path(args.output_dir)
            self.first = True
            self.last_logged = None

            def replace_loss(module, inputs, output):
                batch = inputs[1]
                if "loss" not in output or not hasattr(batch, "action"):
                    captured.clear()
                    return output
                target, mask = batch.action, batch.action_mask
                prediction = captured.pop("prediction")[:, -target.shape[1]:]
                raw = captured.pop("raw")[:, -target.shape[1]:]
                with torch.autocast(device_type=prediction.device.type, enabled=False):
                    loss, info = exp_ht_objective(prediction, raw, target, mask)
                assert torch.isfinite(loss), "Non-finite exp-scale HT loss"
                assert (info["d"] == D).all()

                if self.first:
                    ratio = info["sigma"] / info["original_sigma"]
                    median_ratio = float(ratio.median())
                    # The original decoder is centered close to raw=0.  This
                    # catches an accidental uncalibrated exp initialization.
                    assert 0.90 < median_ratio < 1.10, median_ratio
                    quantile = lambda x: torch.quantile(
                        x.detach().float(),
                        torch.tensor([0.1, 0.5, 0.9], device=x.device),
                    ).cpu().tolist()
                    init = dict(
                        step=state.global_step,
                        recipe=recipe,
                        exp_sigma_quantiles=quantile(info["sigma"]),
                        softplus_reference_sigma_quantiles=quantile(info["original_sigma"]),
                        exp_over_softplus_quantiles=quantile(ratio),
                        raw_sigma_quantiles=quantile(
                            raw.detach().float().masked_select(mask.bool())
                        ),
                        seed=int(args.seed),
                        data_seed=int(args.data_seed if args.data_seed is not None else args.seed),
                    )
                    if state.is_world_process_zero:
                        (out_dir / "initialization_verified.json").write_text(
                            json.dumps(init, indent=2) + "\n"
                        )
                        print("NU224_EXP_INITIALIZATION", json.dumps(init), flush=True)
                    self.first = False

                output["loss"] = loss
                output["action_loss"] = info["per"] * info["d"]
                step = state.global_step
                if state.is_world_process_zero and step % 100 == 0 and step != self.last_logged:
                    g_log_sigma, g_prediction = torch.autograd.grad(
                        info["per"].sum(), (info["log_sigma"], prediction), retain_graph=True
                    )
                    quantile = lambda x: torch.quantile(
                        x.detach().float(),
                        torch.tensor([0.1, 0.5, 0.9], device=x.device),
                    ).cpu().tolist()
                    row = dict(
                        step=step, nu=NU, loss=float(loss),
                        sigma=quantile(info["sigma"]),
                        residual_rms=quantile((info["squared"] / D).sqrt()),
                        q_over_d=quantile(info["q"] / D),
                        mse_per_dim=float(info["squared"].mean() / D),
                        log_sigma_grad_abs_mean=float(g_log_sigma.abs().mean()),
                        log_sigma_grad_mean=float(g_log_sigma.mean()),
                        prediction_grad_norm_mean=float(
                            g_prediction.float().flatten(1).norm(dim=1).mean()
                        ),
                    )
                    with (out_dir / "training_diagnostics.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print("NU224_EXP_DIAGNOSTICS", json.dumps(row), flush=True)
                    self.last_logged = step
                return output

            head.register_forward_hook(replace_loss)
            if state.is_world_process_zero:
                (out_dir / "exp_scale_recipe.json").write_text(
                    json.dumps(recipe, indent=2) + "\n"
                )

    original_init = Gr00tTrainer.__init__
    original_train = Gr00tTrainer.train

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(ExpScale())

    def train(self, resume_from_checkpoint=None, **kwargs):
        assert resume_from_checkpoint in (None, False), "This run must start fresh"
        return original_train(self, resume_from_checkpoint=None, **kwargs)

    Gr00tTrainer.__init__ = initialize
    Gr00tTrainer.train = train
    runpy.run_module("gr00t.experiment.launch_finetune", run_name="__main__")


if __name__ == "__main__":
    main()
