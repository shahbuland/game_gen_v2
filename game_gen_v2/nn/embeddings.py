import torch
from torch import nn
import einops as eo
import math

from rotary_embedding_torch import RotaryEmbedding

from .mlp import MLP
from .normalization import norm

class AbsEmbedding(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()

        self.embedding = nn.Parameter(torch.randn(seq_len, dim))
    
    def forward(self, x):
        # x: [b,n,d]
        p = eo.repeat(self.embedding, 'n d -> b n d', b = x.shape[0])
        return x + p
    
class RoPEEmbedding(nn.Module):
    """
    "Flat" Version of RoPE
    """
    def __init__(self, dim, flash = False):
        super().__init__()

        self.rope = RotaryEmbedding(dim // 2)
        self.flash = flash
    
    def forward(self, q, k):
        if self.flash: # [b,n,h,d] -> [b,h,n,d]
            q = q.transpose(1,2)
            k = k.transpose(1,2)
        q, k = self.rope.rotate_queries_or_keys(q), self.rope.rotate_queries_or_keys(k)
        if self.flash: # [b,h,n,d] -> [b,n,h,d]
            q = q.transpose(1,2)
            k = k.transpose(1,2)
        return q,k