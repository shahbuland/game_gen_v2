import torch
from torch import nn

def truncated_normal_init(module: nn.Module, std: float = 0.02):
    """
    Initialize the parameters of a module using a truncated normal distribution.
    
    Args:
        module (nn.Module): The PyTorch module to initialize.
        std (float): The standard deviation of the normal distribution. Default is 0.02.
    """
    for p in module.parameters():
        nn.init.trunc_normal_(p, mean=0.0, std=std, a=-std*2, b=std*2)

def normal_init(module: nn.Module, std: float = 0.02):
    """
    Initialize the parameters of a module using a normal distribution.
    
    Args:
        module (nn.Module): The PyTorch module to initialize.
        std (float): The standard deviation of the normal distribution. Default is 0.02.
    """
    for p in module.parameters():
        nn.init.normal_(p, mean=0.0, std=std)
