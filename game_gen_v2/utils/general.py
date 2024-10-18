import torch
from torch import nn

def count_parameters(model):
    """
    Count and print the number of learnable parameters in a model.
    
    Args:
        model (nn.Module): The PyTorch model to analyze.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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