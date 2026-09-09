"""Fractal HT: hold nu=224 for 7k, decay to 64, with exact scale compensation.

The auxiliary Gaussian term is mean-detached and receives a per-example,
detached weight chosen so the total direct log-sigma gradient equals the
fixed-nu=224 Student-t gradient.  It therefore restores only the scale
gradient attenuated by lowering nu and cannot overshoot that reference at the
loss level.
"""
import json
import math
import os
from pathlib import Path
import runpy

NU_REFERENCE = 224.0
NU_TARGET = 64.0
HOLD_UPDATES = 7000
DECAY_UPDATES = 5000
TOTAL_UPDATES = 20000
D = 56
SBIAS = -0.5093
EPS = 0.001


def nu_for_completed_steps(completed_steps):
    """Nu used by the next optimizer update."""
    if completed_steps < HOLD_UPDATES:
        return NU_REFERENCE
    transition_update = completed_steps - HOLD_UPDATES + 1
    progress = min(1.0, max(0.0, transition_update / DECAY_UPDATES))
    return math.exp(
        (1.0 - progress) * math.log(NU_REFERENCE)
        + progress * math.log(NU_TARGET)
    )


def compensated_objective(prediction, raw_sigma, target, mask, nu):
    import torch
    mask = mask.float()
    d = mask.sum((1, 2))
    sigma = (
        torch.nn.functional.softplus(raw_sigma.float() + SBIAS) * mask
    ).sum((1, 2)) / d + EPS
    log_sigma = sigma.log()
    # Route every likelihood term through this named log-scale so both the
    # analytic check and raw-head backpropagation include the complete scale
    # derivative.
    sigma = log_sigma.exp()

    residual = prediction.float() - target.float()
    squared = (residual.square() * mask).sum((1, 2))
    q = squared / sigma.square()
    ht = 0.5 * (nu + d) / d * torch.log1p(q / nu) + log_sigma

    # This branch must update sigma but not mu.  Detaching the coefficient is
    # essential: its derivative is not part of the desired compensation.
    q_scale = squared.detach() / sigma.square()
    q_for_weight = q.detach()
    gate_reference = NU_REFERENCE / (NU_REFERENCE + q_for_weight)
    gate_current = nu / (nu + q_for_weight)
    weight = (gate_reference - gate_current).detach()
    hg = 0.5 * q_scale / d + log_sigma
    auxiliary = weight * hg
    total = ht + auxiliary
    return total.mean(), dict(
        total=total, ht=ht, auxiliary=auxiliary, hg=hg, weight=weight,
        sigma=sigma, log_sigma=log_sigma, squared=squared, q=q, d=d,
        gate_current=gate_current, gate_reference=gate_reference,
    )


def validate_config(cfg):
    assert cfg.loss_type == "hetero_t" and cfg.ht_mvt
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

    class ScheduleAndCompensate(TrainerCallback):
        def set_nu(self, args, state, model):
            core = model.module if hasattr(model, "module") else model
            nu = nu_for_completed_steps(state.global_step)
            for cfg in (core.config, core.action_head.config):
                validate_config(cfg)
                cfg.ht_df = float(nu)
            self.nu = float(nu)
            if state.is_world_process_zero and state.global_step % 100 == 0:
                row = dict(
                    completed_steps=state.global_step,
                    next_update=state.global_step + 1,
                    nu=self.nu,
                    compensation_active=self.nu < NU_REFERENCE,
                )
                with (Path(args.output_dir) / "nu_schedule.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print("NU224_TO64_SCHEDULE", json.dumps(row), flush=True)
            return core

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            assert state.global_step == 0
            assert args.max_steps == TOTAL_UPDATES
            assert args.get_warmup_steps(TOTAL_UPDATES) == 1000
            assert args.learning_rate == 1e-4 and args.weight_decay == 1e-5
            assert args.world_size == 8 and args.per_device_train_batch_size == 128
            assert args.gradient_accumulation_steps == 1
            assert str(args.optim).lower().endswith("adamw_torch")
            core = self.set_nu(args, state, model)
            head = core.action_head
            assert not hasattr(head, "nu_decoder")
            recipe = dict(
                objective="joint Student-t NLL plus scale-only Gaussian compensation",
                nu_reference=NU_REFERENCE, nu_target=NU_TARGET,
                hold_updates=HOLD_UPDATES, decay_updates=DECAY_UPDATES,
                decay="log-linear from updates 7001 through 12000; hold 64 thereafter",
                total_updates=TOTAL_UPDATES, d=D,
                sigma="native masked-mean softplus(raw - 0.5093) + 0.001",
                compensation=(
                    "detached [224/(224+q)-nu/(nu+q)] times HG(stopgrad(mu),sigma)"
                ),
                scale_gradient_guarantee=(
                    "per example, direct dL/dlog_sigma equals fixed-nu=224 HT at current mu,sigma"
                ),
                initialization="unchanged original seeded sigma decoder",
                start="fresh optimizer and original pretrained GR00T",
                evaluation=(
                    "six Fractal tasks, seed=1234, 100 rollouts/task, "
                    "5 envs, NAS=1, max_steps=300"
                ),
                inference="unchanged deterministic mean",
            )
            for cfg in (core.config, head.config):
                cfg.ht_nu64_hgcomp_recipe = recipe

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
                native = output["loss"].float()
                with torch.autocast(device_type=prediction.device.type, enabled=False):
                    loss, info = compensated_objective(
                        prediction, raw, target, mask, self.nu
                    )
                assert torch.isfinite(loss)
                assert (info["d"] == D).all()
                torch.testing.assert_close(
                    native, info["ht"].mean(), atol=2e-5, rtol=2e-5
                )

                # Exact no-overshoot identity for the direct scale gradient.
                g_hg = 1.0 - info["q"].detach() / info["d"]
                g_total = (info["gate_current"] + info["weight"]) * g_hg
                g_reference = info["gate_reference"] * g_hg
                torch.testing.assert_close(g_total, g_reference, atol=2e-7, rtol=2e-6)
                assert (info["weight"] >= -1e-8).all()

                output["loss"] = loss
                output["action_loss"] = info["total"] * info["d"]
                step = state.global_step
                if state.is_world_process_zero and (
                    self.first or (step % 100 == 0 and step != self.last_logged)
                ):
                    quantile = lambda x: torch.quantile(
                        x.detach().float(),
                        torch.tensor([0.1, 0.5, 0.9], device=x.device),
                    ).cpu().tolist()
                    row = dict(
                        step=step, next_update=step + 1, nu=self.nu,
                        loss=float(loss), ht_loss=float(info["ht"].mean()),
                        auxiliary_loss=float(info["auxiliary"].mean()),
                        compensation_weight=quantile(info["weight"]),
                        sigma=quantile(info["sigma"]),
                        residual_rms=quantile((info["squared"] / D).sqrt()),
                        q_over_d=quantile(info["q"] / D),
                        current_gate=quantile(info["gate_current"]),
                        reference_gate=quantile(info["gate_reference"]),
                        scale_gradient_abs_mean=float(g_total.abs().mean()),
                        scale_gradient_reference_abs_mean=float(g_reference.abs().mean()),
                        scale_gradient_max_error=float((g_total - g_reference).abs().max()),
                    )
                    with (out_dir / "training_diagnostics.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print("NU224_TO64_HGCOMP", json.dumps(row), flush=True)
                    self.first = False
                    self.last_logged = step
                return output

            head.register_forward_hook(replace_loss)
            if state.is_world_process_zero:
                (out_dir / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")

        def on_step_begin(self, args, state, control, model=None, **kwargs):
            self.set_nu(args, state, model)

    original_init = Gr00tTrainer.__init__
    original_train = Gr00tTrainer.train

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(ScheduleAndCompensate())

    def train(self, resume_from_checkpoint=None, **kwargs):
        assert resume_from_checkpoint in (None, False), "This run must start fresh"
        return original_train(self, resume_from_checkpoint=None, **kwargs)

    Gr00tTrainer.__init__ = initialize
    Gr00tTrainer.train = train
    runpy.run_module("gr00t.experiment.launch_finetune", run_name="__main__")


if __name__ == "__main__":
    main()
