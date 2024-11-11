import torch
from accelerate import Accelerator
from dataclasses import asdict
import os
from ema_pytorch import EMA
import wandb

from ..configs import TrainConfig, ConfigClass
from ..utils import Stopwatch
from ..opts import get_scheduler_cls, get_extra_optimizer

class BaseTrainer:
    def __init__(self, config : TrainConfig, model_config : ConfigClass = None):
        self.config = config
        self.model_config = model_config

        self.accum_steps = self.config.target_batch_size // self.config.batch_size
        self.accelerator = Accelerator(
            log_with = "wandb",
            gradient_accumulation_steps = self.accum_steps
        )

        tracker_kwargs = {}
        if config.run_name is not None:
            tracker_kwargs['wandb'] = {
                'name' :   config.run_name,
                'entity' : config.wandb_entity,
                'mode' : 'online'
            }

            config_dict = asdict(config)
            if model_config is not None:
                config_dict.update(asdict(model_config))


            self.accelerator.init_trackers(
                project_name = config.wandb_project,
                config = config_dict,
                init_kwargs = tracker_kwargs
            )

        self.world_size = self.accelerator.state.num_processes
        self.total_step_counter = 0
        self.ema = None
        self.log = config.run_name is not None

    def get_should(self, step = None):
        # Get a dict of bools that determines if certain things should be done at the current step
        if step is None:
            step = self.total_step_counter

        def should_fn(interval):
            return step % interval == 0 and self.accelerator.sync_gradients

        return {
            "log" : should_fn(self.config.log_interval) and self.log,
            "save" : should_fn(self.config.save_interval),
            "sample" : should_fn(self.config.sample_interval),
            "val" : should_fn(self.config.val_interval)
        }

    def save(self, step = None, dir = None):
        """
        In directory, save checkpoint of accelerator state using step and self.config.run_name
        """
        if step is None:
            step = self.total_step_counter
        if dir is None:
            dir = os.path.join(self.config.checkpoint_root_dir, f"{self.config.run_name}_{step}")
        
        os.makedirs(dir, exist_ok = True)

        self.accelerator.save_state(output_dir = dir)
        if self.ema is not None:
            ema_path = os.path.join(dir, "ema_model.pth")
            torch.save(self.ema.state_dict(), ema_path)
            ema_model_path = os.path.join(dir, "out.pth")
            torch.save(self.ema.ema_model.state_dict(), ema_model_path)
    
    def load(self):
        """
        Load the latest checkpoint from the checkpoint directory.
        """
        checkpoint_dir = self.config.checkpoint_root_dir
        run_name = self.config.run_name
        
        # Get all directories that match the run name pattern
        matching_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith(run_name) and d[len(run_name):].strip('_').isdigit()]
        
        if not matching_dirs:
            print(f"No checkpoints found for {run_name}")
            _ = input("Are you sure you want to continue?")
            return
        
        # Sort directories by the number at the end and get the latest
        latest_dir = max(matching_dirs, key=lambda x: int(x[len(run_name):].strip('_')))
        latest_checkpoint = os.path.join(checkpoint_dir, latest_dir)
        
        print(f"Loading checkpoint from {latest_checkpoint}")
        
        # Load accelerator state
        self.accelerator.load_state(latest_checkpoint)
        
        # Load EMA if it exists
        ema_path = os.path.join(latest_checkpoint, "ema_model.pth")
        if os.path.exists(ema_path) and self.ema is not None:
            self.ema.load_state_dict(torch.load(ema_path))
        
        print("Checkpoint loaded successfully")
    
    def train_preamble(self, model, train_loader, val_loader = None):
        # First part of training setup that is common across all usecases
        try:
            opt_class = getattr(torch.optim, self.config.opt)
        except:
            opt_class = get_extra_optimizer(self.config.opt)
        
        opt = opt_class(model.parameters(), **self.config.opt_kwargs)

        # scheduler setup
        scheduler = None
        if self.config.scheduler is not None:
            try:
                scheduler_class = getattr(torch.optim.lr_scheduler, self.config.scheduler)
            except:
                scheduler_class = get_scheduler_cls(self.config.scheduler)
            scheduler = scheduler_class(opt, **self.config.scheduler_kwargs)
        
        model.train()
        model, train_loader, opt = self.accelerator.prepare(model, train_loader, opt)
        if scheduler: scheduler = self.accelerator.prepare(scheduler)
        if val_loader: val_loader = self.accelerator.prepare(val_loader)

        if self.config.use_ema:
            self.ema = EMA(
                beta = self.config.ema_beta,
                update_after_step = self.config.ema_start_offset,
                update_every = self.config.ema_every,
                ignore_names = self.config.ema_ignore,
                coerce_dtype = True
            )
            accel_ema = self.accelerator.prepare(self.ema)
        else:
            accel_ema = None

        return model, train_loader, val_loader, opt, scheduler, accel_ema
    
    def train(self, model, train_loader, val_loader = None):
        pass