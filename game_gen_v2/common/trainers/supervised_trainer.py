import torch
import wandb

from .base import BaseTrainer
from ..utils import Stopwatch

class SimpleSupervisedTrainer(BaseTrainer):
    def train(self, model, train_loader, val_loader = None, sampler = None, compile = False):
        model, train_loader, val_loader, opt, scheduler, _ = self.train_preamble(model, train_loader, val_loader)

        sw = Stopwatch()
        sw.reset()

        if compile:
            model = torch.compile(
                model,
                mode="max-autotune",
                fullgraph=True
            )

        for epoch in range(self.config.epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                with self.accelerator.accumulate(train_loader):
                    loss, extra = model(inputs, labels)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        if self.config.grad_clip > 0: self.accelerator.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    
                    opt.step()
                    if scheduler:
                        scheduler.step()
                    opt.zero_grad()

                    if self.accelerator.sync_gradients:
                        self.total_step_counter += 1
                        if self.ema is not None:
                            self.ema.update()
                    
                    should = self.get_should()
                    if should['log']:
                        wandb_dict = {
                            'loss' : loss.item(),
                            "time_per_1k_steps" : sw.hit(self.config.log_interval)
                        }
                        if extra is not None:
                            wandb_dict.update(extra)
                        if should['sample'] and sampler is not None:
                            if self.ema is None:
                                samples = sampler(self.accelerator.unwrap_model(model))
                            else:
                                samples = sampler(self.ema.ema_model)

                            wandb_dict.update({
                                "samples" : samples
                            }) 
                        wandb.log(wandb_dict)
                        sw.reset()
                    if should['save']:
                        self.save(self.total_step_counter)
