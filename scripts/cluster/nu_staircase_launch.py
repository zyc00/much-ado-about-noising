"""Process-local joint Student-t nu schedule; shared GR00T sources are untouched."""
import json
from pathlib import Path
import runpy

VALUES = (1792, 896, 448, 224, 112, 56, 28, 14, 7)


def nu_at_step(completed_steps):
    """Nu for the NEXT optimizer update (HF global_step counts completed ones)."""
    if completed_steps < 0:
        raise ValueError(completed_steps)
    return float(VALUES[min(completed_steps // 1000, len(VALUES) - 1)])


def main():
    from transformers import TrainerCallback
    from gr00t.experiment.trainer import Gr00tTrainer

    class StaircaseNu(TrainerCallback):
        last_nu = None

        def update(self, args, state, model):
            # The head reads its own config.ht_df on EVERY forward pass.
            core = model.module if hasattr(model, 'module') else model
            nu = nu_at_step(state.global_step)
            for cfg in (core.config, core.action_head.config):
                assert cfg.loss_type == 'hetero_t' and cfg.ht_mvt
                assert float(getattr(cfg, 'ht_nu_start', 0)) == 0
                assert float(getattr(cfg, 'ht_nu_target', 0)) == 0
                cfg.ht_df = nu
                cfg.ht_step_schedule = {'values': list(VALUES), 'interval': 1000}
            if nu != self.last_nu:
                if state.is_world_process_zero:
                    record = {'completed_steps': state.global_step,
                              'next_update': state.global_step + 1, 'nu': nu}
                    print('NU_STAIRCASE ' + json.dumps(record), flush=True)
                    with (Path(args.output_dir) / 'nu_schedule.jsonl').open('a') as f:
                        f.write(json.dumps(record) + '\n')
                self.last_nu = nu

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            assert args.max_steps == 20000
            assert args.get_warmup_steps(args.max_steps) == 1000
            assert args.learning_rate == 1e-4
            assert args.save_steps == 1000 and args.save_total_limit == 20
            self.update(args, state, model)

        def on_step_begin(self, args, state, control, model=None, **kwargs):
            self.update(args, state, model)

    original_init = Gr00tTrainer.__init__

    def init_with_schedule(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(StaircaseNu())

    Gr00tTrainer.__init__ = init_with_schedule
    runpy.run_module('gr00t.experiment.launch_finetune', run_name='__main__')


if __name__ == '__main__':
    main()
