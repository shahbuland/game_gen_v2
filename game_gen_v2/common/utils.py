import torch
from torch import nn

def dict_accum(old_dict, new_dict):
    for key, new_value in new_dict.items():
        if key in old_dict:
            old_value = old_dict[key]
            if isinstance(new_value, type(old_value)):
                old_dict[key] = old_value + new_value
            else:
                old_dict[key] = new_value
        else:
            old_dict[key] = new_value
    return old_dict

def count_parameters(model):
    """
    Count and print the number of learnable parameters in a model.
    
    Args:
        model (nn.Module): The PyTorch model to analyze.
    """
    total_params = sum(p.numel() for p in model.core.parameters() if p.requires_grad)
    return total_params

def pretty_print_parameters(model):
    """
    Same as above func but doesn't return anything, just prints in a pretty format
    """
    params = count_parameters(model)
    formatted_params = params
    if params < 1_000_000:
        formatted_params = f"{params // 1000}K"
    elif params < 1_000_000_000:
        formatted_params = f"{params // 1_000_000}M"
    elif params < 1_000_000_000_000:
        formatted_params = f"{params // 1_000_000_000}B"
    else:
        formatted_params = f"{params // 1_000_000_000_000}T"
    
    print(f"Model has {formatted_params} trainable parameters.")

def freeze(module: nn.Module):
    """
    Set all parameters in a module to not require gradients.
    
    Args:
        module (nn.Module): The PyTorch module to freeze.
    """
    for param in module.parameters():
        param.requires_grad = False

def unfreeze(module: nn.Module):
    """
    Set all parameters in a module to require gradients.
    
    Args:
        module (nn.Module): The PyTorch module to unfreeze.
    """
    for param in module.parameters():
        param.requires_grad = True

# TIMING

import time

class Stopwatch:
    def __init__(self):
        self.start_time = None

    def reset(self):
        """Prime the stopwatch for measurement."""
        self.start_time = time.time()

    def hit(self, samples: int) -> float:
        """
        Measure the average time per 1000 samples since the last reset.

        Args:
            samples (int): The number of samples processed.

        Returns:
            float: The time in seconds per 1000 samples.
        """
        if self.start_time is None:
            raise ValueError("Stopwatch must be reset before calling hit.")

        elapsed_time = time.time() - self.start_time
        avg_time_per_sample = elapsed_time / samples
        return avg_time_per_sample * 1000  # Return time per 1000 samples
