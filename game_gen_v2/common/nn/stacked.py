import torch
from torch import nn

from ..configs import TransformerConfig

class StackedLayers(nn.Module):
    """
    Stacks some layers in a way that makes returning hidden states easy
    """
    def __init__(self, layer_cls, config : TransformerConfig):
        super().__init__()

        self.layers = nn.ModuleList([layer_cls(config) for _ in range(config.n_layers)])

    def forward(self, x, out_h : bool = False):
        h = []
        for layer in self.layers:
            x = layer(x)
            h.append(x)
        
        if out_h:
            return x,h
        return x

class StackedMultiLayers(nn.Module):
    """
    For layers that take many inputs
    """
    def __init__(self, layer_cls, config : TransformerConfig):
        super().__init__()

        self.layers = nn.ModuleList([layer_cls(config) for _ in range(config.n_layers)])

    def forward(self, *x, out_h : bool = False):
        h = []
        for layer in self.layers:
            x = layer(*x)
            h.append(x[0])
        
        x = x[0]
        if out_h:
            return x,h
        return x