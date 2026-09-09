"""Original 20k MSE schedule with configurable early saves and stopping.

Only this process installs the callback; no shared training code is edited.
Defaults reproduce the 500-step run; MSE_PROBE_STOP_STEP and
MSE_PROBE_SAVE_STEPS configure the requested 50/100/200/250 replay.
"""
import os
import runpy

from transformers import TrainerCallback
from gr00t.experiment.trainer import Gr00tTrainer


STOP_STEP = int(os.environ.get('MSE_PROBE_STOP_STEP', '500'))
SAVE_STEPS = {int(v) for v in os.environ.get('MSE_PROBE_SAVE_STEPS', str(STOP_STEP)).split(',')}
assert 0 < STOP_STEP <= 500 and STOP_STEP in SAVE_STEPS
assert all(0 < step <= STOP_STEP for step in SAVE_STEPS)


class StopAtProbeStep(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        assert args.max_steps == 20000, args.max_steps
        assert args.get_warmup_steps(args.max_steps) == 1000
        assert args.learning_rate == 1e-4
        assert model.config.loss_type == 'mse'
        assert state.global_step == 0, 'Must start fresh, not resume'
        print(f'MSE_PROBE_AUDIT: fresh run; 20k schedule; 1000 warmup steps; save {sorted(SAVE_STEPS)}; stop at {STOP_STEP}', flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in SAVE_STEPS:
            control.should_save = True
        if state.global_step >= STOP_STEP:
            assert state.global_step == STOP_STEP
            control.should_training_stop = True
        return control


original_init = Gr00tTrainer.__init__


def init_with_stop(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.add_callback(StopAtProbeStep())


Gr00tTrainer.__init__ = init_with_stop
runpy.run_module('gr00t.experiment.launch_finetune', run_name='__main__')
