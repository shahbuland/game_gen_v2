import torch
from torch import nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
except:
    print("Could not import flash attention. You should stick to linear attention.")

from .normalization import LayerNorm, RMSNorm
from .embeddings import RoPEEmbedding
from .mlp import MLP, MixFFN3D

import einops as eo

def qkv_chunk(qkv):
    return [x.contiguous() for x in qkv.chunk(3, dim = -1)]

def kv_chunk(kv):
    return [x.contiguous() for x in kv.chunk(2, dim = -1)]

def head_split(x, n_heads):
    return eo.rearrange(x, 'b n (h d) -> b n h d', h = n_heads)

def head_merge(x):
    return eo.rearrange(x, 'b n h d -> b n (h d)')

def default_attn_func(q, k, v):
    # q,k,v are all b n h d
    def swap_n_h(x):
        return x.transpose(-3,-2)
    
    q = swap_n_h(q)
    k = swap_n_h(k)
    v = swap_n_h(v)

    out = F.scaled_dot_product_attention(q,k,v)

    return swap_n_h(out)

class Attn(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model
        dim_head = config.d_model // config.n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias = False)
        self.out = nn.Linear(dim, dim)

        self.split = lambda x: head_split(x, n_heads = config.n_heads)
        self.merge = lambda x: head_merge(x)

        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)

        self.cross = config.cross
        if self.cross:
            self.cross_kv = nn.Linear(dim, 2*dim, bias = False)
            self.cross_k_norm = RMSNorm(dim_head)

        self.rope = RoPEEmbedding(dim_head)

        try:
            self.attn_func = flash_attn_func
        except:
            self.attn_func = default_attn_func

    def forward(self, x, y=None):
        _,n,_ = x.shape
        qkv = self.qkv(x)
        q,k,v = [self.split(x) for x in qkv_chunk(qkv)]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.cross:
            cross_kv = self.cross_kv(y)
            c_k,c_v = [self.split(x) for x in kv_chunk(cross_kv)]
            c_k = self.cross_k_norm(c_k)
            k = torch.cat([k, c_k], 1)
            v = torch.cat([v, c_v], 1)

        q, k = self.rope(q,k)

        attn_out = self.attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)).to(q.dtype)
        attn_out = self.merge(attn_out)

        return self.out(attn_out)

class LinearAttn(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model
        dim_head = config.d_model // config.n_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.eps = 1.0e-6

        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)

        self.cross = config.cross
        if self.cross:
            self.cross_kv = nn.Linear(dim, 2*dim)
            self.cross_k_norm = RMSNorm(dim_head)

        self.split = lambda x: head_split(x, n_heads = config.n_heads)
        self.merge = lambda x: head_merge(x)

    def forward(self, x, y=None):
        _,n,_ = x.shape
        q,k,v = [self.split(x) for x in qkv_chunk(self.qkv(x))]

        # all [b,n,h,d] now
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.cross:
            c_k,c_v = [self.split(x) for x in kv_chunk(self.cross_kv(y))]
            c_k = self.cross_k_norm(c_k)
            k = torch.cat([k,c_k],1)
            v = torch.cat([v,c_v],1)

        q = self.relu(q)
        k = self.relu(k)

        # einsum bmhd,bmhd->bhdd
        k_sum = k.sum(1) # [b,h,d]
        k = k.permute(0, 2, 3, 1) # [b,h,d,n]
        v = v.permute(0, 2, 1, 3) # [b,h,n,d]
        kv = torch.matmul(k,v) # [b,h,d,d]

        # [b,n,h,d] -> [b,h,n,d]
        q = q.permute(0, 2, 1, 3)
        attn_out = torch.matmul(q,kv) # [b,h,n,d]
        scale = torch.einsum('bhnd,bhd->bhn',q,k_sum).unsqueeze(-1) # [b,h,n,1]

        attn_out = attn_out/(scale + self.eps)
        attn_out = attn_out.permute(0,2,1,3) # -> [b,n,h,d]
        x = self.merge(attn_out)
        return self.out(x)
