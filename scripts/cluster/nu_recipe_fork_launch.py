"""Controlled continuation from the SAME 7k model/Adam/scheduler checkpoint."""
import hashlib
import json
import math
import os
from pathlib import Path
import runpy

SOURCE = Path('/mnt/pfs/yuchen/groot/ft_fr_nu1024_hold7k_to14_20260908/checkpoint-7000')
ARMS = ('stair14', 'smooth14', 'smooth7')
STOP = 13000


def nu_value(step, arm):
    if arm not in ARMS or step < 7000:
        raise ValueError((step, arm))
    if arm == 'stair14':
        return float((512,256,128,64,32,14)[min((step-7000)//1000,5)])
    end = 14. if arm == 'smooth14' else 7.
    progress = min(1., max(0., (step-7000)/5000.))
    return float(math.exp((1-progress)*math.log(1024.)+progress*math.log(end)))


def optimizer_steps(opt):
    pending=[opt];seen=set()
    found=[]
    while pending:
        obj=pending.pop()
        if id(obj) in seen:continue
        seen.add(id(obj))
        state=getattr(obj,'state',None)
        if isinstance(state,dict):
            for value in state.values():
                if isinstance(value,dict) and 'step' in value:
                    found.append(float(value['step']))
        for key in ('optimizer','optim'):
            child=getattr(obj,key,None)
            if child is not None:pending.append(child)
    return found


def main():
    import torch
    from transformers import TrainerCallback
    from gr00t.experiment.trainer import Gr00tTrainer
    # This Transformers build performs a model-info HTTP request even with
    # HF_HUB_OFFLINE. Resolve the identical cached processor by local path,
    # as in the evaluation server; no tokenizer or preprocessing change.
    from huggingface_hub import snapshot_download
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing
    original_builder=processing.build_processor
    def cached_processor(name,kwargs):
        local=snapshot_download(name,local_files_only=True) if not name.startswith('/') else name
        return original_builder(local,kwargs)
    processing.build_processor=cached_processor
    arm=os.environ['NU_DEBUG_ARM']
    assert arm in ARMS
    original_train=Gr00tTrainer.train
    original_init=Gr00tTrainer.__init__

    class Audit(TrainerCallback):
        active_step=None
        previous=None

        def update(self,args,state,model):
            core=model.module if hasattr(model,'module') else model
            assert not hasattr(core.action_head,'nu_decoder')
            nu=nu_value(state.global_step,arm)
            for cfg in (core.config,core.action_head.config):
                assert cfg.ht_mvt and cfg.loss_type=='hetero_t'
                assert not cfg.ht_nu_start and not cfg.ht_nu_target and not cfg.ht_beta
                assert not getattr(cfg,'ht_gripper_bce',False)
                assert not getattr(cfg,'ht_mse_steps',0) and not getattr(cfg,'ht_hg_steps',0)
                cfg.ht_df=nu
                cfg.ht_debug_schedule=dict(arm=arm,source=str(SOURCE),start=7000,end=12000,stop=STOP)
            if state.is_world_process_zero and state.global_step%100==0:
                self.active_step=state.global_step
                row=dict(step=state.global_step,next_update=state.global_step+1,nu=nu)
                with (Path(args.output_dir)/'nu_schedule.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print('NU_DEBUG',json.dumps(row),flush=True)
            return core

        def on_train_begin(self,args,state,control,model=None,optimizer=None,lr_scheduler=None,**kwargs):
            assert state.global_step==7000,state.global_step
            assert args.max_steps==18000 and args.get_warmup_steps(18000)==1000
            assert args.learning_rate==1e-4 and args.gradient_accumulation_steps==1
            assert args.world_size==8 and args.per_device_train_batch_size==128
            assert args.weight_decay==1e-5
            steps=optimizer_steps(optimizer)
            assert steps and min(steps)==max(steps)==7000,('Adam state was not restored',steps[:10])
            expected=torch.load(SOURCE/'scheduler.pt',map_location='cpu',weights_only=False)['_last_lr']
            actual=lr_scheduler.get_last_lr()
            assert len(expected)==len(actual) and all(abs(a-b)<1e-12 for a,b in zip(expected,actual)),(expected,actual)
            core=self.update(args,state,model)
            if not state.is_world_process_zero:return
            out=Path(args.output_dir)
            (out/'resume_verified.json').write_text(json.dumps(dict(
                source=str(SOURCE),step=state.global_step,optimizer_step_min=min(steps),
                optimizer_step_max=max(steps),optimizer_states=len(steps),lr=actual,
                data_stream='Trainer reset_seed(base_seed+7000), same across resumed arms; not uninterrupted replay.'),indent=2))
            captured={}
            def capture_inputs(head,inputs):
                if self.active_step is not None:
                    features=inputs[0].backbone_features[:2].detach().float().cpu().contiguous().numpy()
                    captured['backbone_digest']=hashlib.sha256(features.tobytes()).hexdigest()
            core.action_head.register_forward_pre_hook(capture_inputs)
            core.action_head.action_decoder.register_forward_hook(lambda m,i,o:captured.__setitem__('prediction',o.detach()))
            core.action_head.sigma_decoder.register_forward_hook(lambda m,i,o:captured.__setitem__('sigma_raw',o.detach()))

            def record(head,inputs,output):
                if self.active_step is None:
                    captured.clear();return
                step=self.active_step;self.active_step=None
                batch=inputs[1]
                a=batch.action.float();mask=batch.action_mask.float()
                p=captured.pop('prediction')[:,-a.shape[1]:].float()
                raw=captured.pop('sigma_raw')[:,-a.shape[1]:].float()
                with torch.no_grad():
                    d=mask.sum((1,2));s=((p-a).square()*mask).sum((1,2))
                    sigma=(torch.nn.functional.softplus(raw+head.config.ht_sbias)*mask).sum((1,2))/d+.001
                    q=s/sigma.square();nu=float(head.config.ht_df);gate=(nu+d)/(nu+q)
                    joint_nll=(.5*(nu+d)*torch.log1p(q/nu)+d*torch.log(sigma)
                               +math.lgamma(nu/2)-torch.lgamma((nu+d)/2)+.5*d*math.log(nu*math.pi))
                    quant=lambda x:torch.quantile(x,torch.tensor([.1,.5,.9],device=x.device)).cpu().tolist()
                    fingerprint=hashlib.sha256()
                    for t in (a,mask,batch.state.float()):fingerprint.update(t.detach().cpu().contiguous().numpy().tobytes())
                    row=dict(step=step,nu=nu,batch=len(a),rank=0,
                             data_fingerprint=fingerprint.hexdigest(),
                             backbone_input_fingerprint=captured.pop('backbone_digest'),
                             sigma=quant(sigma),rms=quant(torch.sqrt(s/d)),q_over_d=quant(q/d),gate=quant(gate),
                             gate_below_half=float((gate<.5).float().mean()),
                             training_loss_without_nu_normalizer=float(output['loss']),
                             joint_nll_per_dim=float(joint_nll.sum()/d.sum()),mse_per_dim=float(s.sum()/d.sum()))
                    reconstructed=(.5*(nu+d)*torch.log1p(q/nu)+d*torch.log(sigma)).sum()/d.sum()
                    assert abs(float(reconstructed)-float(output['loss']))<2e-5, 'Loss branch mismatch'
                    with (out/'training_diagnostics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            core.action_head.register_forward_hook(record)

        def on_step_begin(self,args,state,control,model=None,**kwargs):
            self.update(args,state,model)

        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step>=STOP:
                control.should_save=True
                control.should_training_stop=True
            return control

    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        self.add_callback(Audit())

    def train(self,resume_from_checkpoint=None,**kwargs):
        # Explicit source path; never rely on auto-resume from the output folder.
        return original_train(self,resume_from_checkpoint=str(SOURCE),**kwargs)

    Gr00tTrainer.__init__=initialize
    Gr00tTrainer.train=train
    runpy.run_module('gr00t.experiment.launch_finetune',run_name='__main__')


if __name__=='__main__':main()
