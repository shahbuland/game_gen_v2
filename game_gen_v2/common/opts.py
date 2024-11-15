"""
Optimizers and schedulers
"""

from torch.optim.lr_scheduler import _LRScheduler
import torch
from torch import nn
import math

# === SCHEDULERS ===

def get_scheduler_cls(scheduler_name: str):
    """
    Returns the scheduler class based on the given name.

    Args:
        scheduler_name (str): The name of the scheduler.

    Returns:
        _LRScheduler: The scheduler class.

    Raises:
        ValueError: If an invalid scheduler name is provided.
    """
    scheduler_map = {
        "CosineDecayAfterWarmup": CosineDecayAfterWarmup,
        "CosineDecay": CosineDecay,
        "Warmup" : LinearWarmup
    }

    scheduler_cls = scheduler_map.get(scheduler_name)
    if scheduler_cls is None:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}")
    
    return scheduler_cls

class LinearWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # Keep constant after warmup
            return self.base_lrs

class CosineDecayAfterWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, T_max, eta_min, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineDecayAfterWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        elif self.last_epoch < self.warmup_steps + self.T_max:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / self.T_max
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]
        else:
            # Constant minimum learning rate
            return [self.eta_min for _ in self.base_lrs]

class CosineDecay(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_max:
            # Cosine decay
            progress = self.last_epoch / self.T_max
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]
        else:
            # Constant minimum learning rate
            return [self.eta_min for _ in self.base_lrs]
            

# === OPTIMIZERS ===

def get_extra_optimizer(name):
    if name.lower() == "soap":
        from .soap import SOAP
        return SOAP
    if name.lower() == "heavyball":
        from heavyball import PalmForEachSoap
        return PalmForEachSoap
    if name.lower() == "adopt":
        from .adopt import ADOPT
        return ADOPT
    if name.lower() == "heavyball2":
        from heavyball import SFPaLMForeachSOAP
        return SFPaLMForeachSOAP
    if name.lower() == "lion":
        from lion_pytorch import Lion
        return Lion
