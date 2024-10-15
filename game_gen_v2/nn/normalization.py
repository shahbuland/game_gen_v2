import torch
from torch import nn
import torch.nn.functional as F

def norm(data, eps = 1.0e-6):
    return F.normalize(data, p = 2, dim = -1, eps = eps)

def norm_layer(module : nn.Linear):
    """
    Normalize linear layer along embedding dimension
    """
    module.weight.data = norm(module.weight.data)

def norm_mlp(module : 'MLP'):
    """
    Normalize MLP
    """
    norm_layer(module.fc1)
    norm_layer(module.fc2)

def norm_dit_block(block : nn.Module):
    """
    Shorthand for normalizing a whole dit block
    """
    norm_mlp(block.mlp)
    norm_layer(block.attn.qkv)
    norm_layer(block.attn.out)
    if hasattr(block.attn, 'cross_qkv'):
        norm_layer(block.attn.cross_qkv)

class ScalingLayer(nn.Module):
    """
    Scaling layer from normalized transformer.
    Produces some scaling value "treated" with some init and scale
    """
    def __init__(self, d_model, init, scale):
        super().__init__()

        self.scale = scale
        self.init = init

        self.alpha = nn.Parameter(torch.full((d_model,), scale))

    def forward(self, x):
        alpha = (self.alpha * (self.init / self.scale))
        if x.ndim == 2:
            alpha = alpha[None,:]
        else:
            alpha = alpha[None,None,:]

        return alpha * x
    
class NormalizedLerp(nn.Module):
    """
    Wrapper around scaling layer to simplify lerps for residual signals
    """
    def __init__(self, d_model, init, scale):
        super().__init__()

        self.scale = ScalingLayer(d_model, init, scale)
    
    def forward(self, x, res):
        return norm(res + self.scale(x - res))