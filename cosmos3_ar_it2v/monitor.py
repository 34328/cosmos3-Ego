"""Pure-video measurements; stop for numerical failures, not normal clipping."""
from pathlib import Path
import json
import time
import torch
import torch.distributed as dist
import wandb
from cosmos_framework.utils.callback import Callback
from cosmos_framework.callbacks.grad_clip import GradClip, _group_params_by_mesh, _total_norm_by_mesh
from cosmos_framework.utils.functional.lr_scheduler import LambdaWarmUpCosineScheduler
from collections import deque
import math


class VideoStopPolicy:
    """Video adaptation: clipping alone is normal, not a divergence criterion."""
    def __init__(self):
        self.losses=deque(maxlen=10);self.memory=deque(maxlen=10);self.baseline=None

    @classmethod
    def from_history(cls, path, iteration):
        """Restore loss policy from this run's log; memory is process-local.

        Native DCP exposes a state adapter for the dataloader, not arbitrary
        callbacks. Reuse the existing per-step receipt rather than inventing a
        second checkpoint format. A backwards/repeated step starts a replacement
        log branch; entries after the selected checkpoint never affect its policy.
        """
        policy = cls()
        if iteration == 0:
            return policy
        path = Path(path)
        if not path.is_file():
            raise ValueError(f'Cannot restore stop policy at step {iteration}: missing {path}')
        losses = {}
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
                step = row['step']
                if type(step) is not int or step < 1:
                    raise ValueError('step must be a positive integer')
                if step > iteration:
                    continue
                loss = float(row['video_loss'])
                if not math.isfinite(loss):
                    raise ValueError('nonfinite video_loss')
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f'Cannot restore stop policy: {path}:{line_number}: {error}') from error
            # Later resumed runs supersede the abandoned suffix, not merely the
            # duplicate row. Missing replacement steps must not reuse stale loss.
            for old_step in tuple(losses):
                if old_step >= step:
                    del losses[old_step]
            losses[step] = loss
        missing = [step for step in range(1, iteration + 1) if step not in losses]
        if missing:
            raise ValueError(f'Cannot restore stop policy at step {iteration}: missing history steps {missing[:10]}')
        for step, loss in losses.items():
            policy.update(False, [loss], 0., step=step)
        # Allocation growth is meaningful only inside one process lifetime.
        policy.memory.clear()
        return policy
    def update(self,clipped,losses,resident_gib,*,step=None):
        if not all(math.isfinite(x) for x in (*losses,resident_gib)):
            return 'nonfinite loss or memory metric'
        self.losses.append(float(losses[0]));self.memory.append(resident_gib)
        if len(self.losses)==10:
            mean=sum(self.losses)/10
            if self.baseline is None: self.baseline=mean
            elif mean>3*max(self.baseline,1e-12): return '10-step mean loss exceeds 3x first-10-step mean'
        if len(self.memory)==10 and self.memory[-1]-self.memory[0]>2 and all(b>a for a,b in zip(self.memory,list(self.memory)[1:])):
            return 'resident allocated memory increased every step for 10 steps by >2 GiB'
        return None


class VideoTrainingMonitor(Callback):
    def on_optimizer_init_start(self):
        cfg = self.config
        s = cfg.scheduler
        params = {k:list(s[k]) for k in ('warm_up_steps','cycle_lengths','f_start','f_max','f_min')}
        scheduler = LambdaWarmUpCosineScheduler(**params,verbosity_interval=0)
        row = dict(base_lr=float(cfg.optimizer.lr),scheduler=params,
            weight_decay=float(cfg.optimizer.weight_decay),
            theoretical_lr={str(i):float(cfg.optimizer.lr)*scheduler(i)
                for i in (0,100,int(cfg.trainer.max_iter)//2,int(cfg.trainer.max_iter))},
            action_gen=bool(cfg.model.config.action_gen),frames_per_chunk=int(cfg.model.config.frames_per_chunk),
            local_attention_frames=int(cfg.model.config.local_attention_frames),wandb_mode=str(cfg.job.wandb_mode),sigma_sampler='uniform_shift_postclamp',
            sigma_min=float(cfg.model.config.sigma_min),sigma_max=float(cfg.model.config.sigma_max),sigma_shift=float(cfg.model.config.sigma_shift))
        if not dist.is_initialized() or dist.get_rank()==0:
            root=Path(cfg.job.path_local);root.mkdir(parents=True,exist_ok=True)
            (root/'learning_rate_receipt.json').write_text(json.dumps(row,indent=2))
            print('AR_IT2V_RECIPE '+json.dumps(row),flush=True)

    def on_train_start(self,model,iteration=0):
        self.root=Path(self.config.job.path_local)
        self.policy=VideoStopPolicy.from_history(self.root/'formal_monitor.jsonl',iteration)
        self.clip=next(c for c in self.trainer.callbacks._callbacks if isinstance(c,GradClip))
        self.params=_group_params_by_mesh([p for p in model.net.parameters() if p.requires_grad])
        forbidden=[n for n,p in model.net.named_parameters() if any(x in n for x in ('action2llm','llm2action','action_modality','action_state'))]
        if forbidden:
            raise RuntimeError('Action parameters present in video-only model: '+str(forbidden[:5]))
        if not dist.is_initialized() or dist.get_rank()==0:
            self.root.mkdir(parents=True,exist_ok=True)
            (self.root/'video_only_contract.json').write_text(json.dumps(dict(action_parameters=forbidden,action_gen=False,
                frames_per_chunk=model.config.frames_per_chunk,local_attention_frames=model.config.local_attention_frames),indent=2))

    def on_training_step_start(self,model,data,iteration=0):
        torch.cuda.synchronize();self.resident=torch.cuda.memory_allocated()/2**30
        torch.cuda.reset_peak_memory_stats();self.start=time.perf_counter();self.clips=len(data['video'])
        self.packed_tokens=int(data['_num_tokens'])

    def on_before_backward(self,model,loss,iteration=0):
        bad=(~torch.isfinite(loss.detach())).any().to(torch.int32)
        if dist.is_initialized(): dist.all_reduce(bad,op=dist.ReduceOp.MAX)
        if bad.item(): raise FloatingPointError('AR IT2V nonfinite loss before backward')

    def on_before_optimizer_step(self,model,optimizer,scheduler,grad_scaler,iteration=0):
        # Snapshot this update's rate before scheduler.step(); native optim/lr stays enabled.
        self.learning_rates=[float(x) for x in scheduler.get_last_lr()]

    def on_after_backward(self,model,iteration=0):
        # Before native clipping/sanitization, including across FSDP mesh shards.
        norm,_=_total_norm_by_mesh(self.params)
        bad=(~torch.isfinite(norm)).to(torch.int32)
        if dist.is_initialized(): dist.all_reduce(bad,op=dist.ReduceOp.MAX)
        if bad.item(): raise FloatingPointError('AR IT2V nonfinite raw gradient')

    def on_training_step_batch_end(self,model,data_batch,output_batch,loss,iteration=0):
        if int(output_batch.get('action_token_length',0) or 0):
            raise RuntimeError('Action tokens entered pure-video training')
        torch.cuda.synchronize()
        norm=float(self.clip._last_global_norm[self.clip._state_key])
        local=torch.tensor([self.clips,torch.cuda.max_memory_allocated()/2**30,torch.cuda.max_memory_reserved()/2**30,
            time.perf_counter()-self.start,self.resident,norm,float(norm>self.clip.clip_norm),float(loss.detach()),self.packed_tokens],device=loss.device,dtype=torch.float64)
        rows=[torch.empty_like(local) for _ in range(dist.get_world_size())] if dist.is_initialized() else [local]
        if dist.is_initialized(): dist.all_gather(rows,local)
        self.rows=torch.stack(rows).cpu()

    def on_training_step_end(self,model,data_batch,output_batch,loss,iteration=0):
        r=self.rows;mean_loss=float(r[:,7].mean())
        reason=self.policy.update(bool(r[:,6].max()),[mean_loss],float(r[:,4].max()),step=iteration)
        row=dict(step=iteration,global_batch=int(r[:,0].sum()),clips_per_rank=r[:,0].int().tolist(),
            peak_allocated_gib=float(r[:,1].max()),peak_reserved_gib=float(r[:,2].max()),train_step_seconds=float(r[:,3].max()),
            resident_allocated_gib=float(r[:,4].max()),preclip_norm=float(r[:,5].max()),grad_clip_triggered=bool(r[:,6].max()),
            video_loss=mean_loss,stop_reason=reason,
            learning_rate_min=min(self.learning_rates),learning_rate_max=max(self.learning_rates),
            packed_tokens_per_rank=r[:,8].int().tolist(),
            packed_tokens_mean=float(r[:,8].mean()))
        budget=self.config.dataloader_train.max_sequence_length
        if budget is not None:
            row.update(token_budget=int(budget),token_fill_mean=float(r[:,8].mean())/int(budget),
                token_fill_min=float(r[:,8].min())/int(budget),token_fill_max=float(r[:,8].max())/int(budget))
        if not dist.is_initialized() or dist.get_rank()==0:
            with (self.root/'formal_monitor.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
            if wandb.run is not None:
                wandb.log({'ar_it2v/'+k:v for k,v in row.items() if isinstance(v,(int,float))},step=iteration,commit=False)
            if iteration<=20 or iteration%100==0 or reason: print('AR_IT2V_PROGRESS '+json.dumps(row),flush=True)
            if reason: (self.root/'STOPPED.json').write_text(json.dumps(row,indent=2))
        if reason: raise RuntimeError('AR IT2V stopped: '+reason)
