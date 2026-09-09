"""Task-local shim honoring the requested normalization on pretrained models.

The installed processor's from_pretrained override whitelist omits
use_percentiles, silently ignoring --no-use-percentiles. Set and recompute
its normalizer explicitly after loading; no shared GR00T source is changed.
"""
import runpy
import json


def install_pipeline_overrides():
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    original = Gr00tN1d7Pipeline._create_model
    if getattr(original, '_robomimic_override', False):
        return

    def create_model(self):
        from gr00t.utils.dist_utils import run_or_wait_on_rank0
        model = original(self)
        model.config.use_percentiles = self.config.model.use_percentiles
        with run_or_wait_on_rank0(label='robomimic model normalization audit') as rank0:
            if rank0:
                (self.save_cfg_dir / 'final_model_config.json').write_text(model.config.to_filtered_json())
        print(f'ROBOMIMIC_NORMALIZATION use_percentiles={model.config.use_percentiles}', flush=True)
        return model

    create_model._robomimic_override = True
    Gr00tN1d7Pipeline._create_model = create_model
    original_dataset = Gr00tN1d7Pipeline._create_dataset

    def create_dataset(self, *args, **kwargs):
        from gr00t.utils.dist_utils import run_or_wait_on_rank0
        result = original_dataset(self, *args, **kwargs)
        value = self.config.model.use_percentiles
        self.processor.use_percentiles = value
        self.processor.state_action_processor.use_percentiles = value
        self.processor.set_statistics(dict(self.processor.statistics), override=True)
        with run_or_wait_on_rank0(label='robomimic processor normalization audit') as rank0:
            if rank0:
                (self.save_cfg_dir / 'robomimic_processor_overrides.json').write_text(
                    json.dumps({'use_percentiles': value,
                                'reason': 'Honor CLI override omitted by pretrained processor loader'})+'\n')
        print(f'ROBOMIMIC_PROCESSOR use_percentiles={self.processor.use_percentiles}', flush=True)
        return result

    Gr00tN1d7Pipeline._create_dataset = create_dataset


if __name__ == '__main__':
    install_pipeline_overrides()
    runpy.run_module('gr00t.experiment.launch_finetune', run_name='__main__')
