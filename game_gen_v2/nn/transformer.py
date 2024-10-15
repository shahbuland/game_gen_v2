import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from .mlp import MLP
from .normalization import NormalizedLerp, ScalingLayer, norm
from .embeddings import RoPEEmbedding
from .modulation import DoubleMod
from ..configs import ModelConfig

def contiguous_qkv_chunk(qkv):
    q, k, v = qkv.chunk(3, dim = -1)
    return q.contiguous(), k.contiguous(), v.contiguous()

def head_split(x, n_heads, flash = False):
    if flash:
        return eo.rearrange(x, 'b n (h d) -> b n h d', h = n_heads)
    else:
        return eo.rearrange(x, 'b n (h d) -> b h n d', h = n_heads)
    
def head_merge(x, flash = False):
    if flash:
        return eo.rearrange(x, 'b n h d -> b n (h d)')
    else:
        return eo.rearrange(x, 'b h n d -> b n (h d)')
    
class Attn(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias = False)
        self.out = nn.Linear(config.d_model, config.d_model)

        self.head_split = lambda x: head_split(x, n_heads = config.n_heads, flash = config.flash)
        self.head_merge = lambda x: head_merge(x, flash = config.flash)

        self.cross_qkv = nn.Linear(config.d_model, 3 * config.d_model, biase = False)
    
        self.qk_scale = ScalingLayer(config.d_model, 1, config.d_model ** -.5)
        self.qk_cross_scale = ScalingLayer(config.d_model, 1, config.d_model ** -.5)

        self.rope = RoPEEmbedding(config.d_model // config.n_heads, flash = config.flash)
        self.attn_scale = (config.d_model // config.n_heads) ** .5

        if not config.flash:
            self.attn_fn = F.scaled_dot_product_attention
        else:
            from flash_attn import flash_attn_func
            self.attn = flash_attn_func
        self.flash = config.flash

    def forward(self, x, c = None):
        b,n,d = x.shape

        # get main q,k ready
        q,k,v = [self.head_split(i) for i in contiguous_qkv_chunk(self.qkv(x))]
        q = self.qk_scale(norm(q))
        k = self.qk_scale(norm(k))

        # get cross q,k ready
        c_q, c_k, c_v = [self.head_split(i) for i in contiguous_qkv_chunk(self.cross_qkv(c))]
        c_q = self.qk_cross_scale(norm(c_q))
        c_k = self.qk_cross_scale(norm(c_k))

        if self.flash:
            seq_dim = 1
        else:
            seq_dim = 2

        q = torch.cat([q, c_q], dim = seq_dim)
        k = torch.cat([k, c_k], dim = seq_dim)
        v = torch.cat([v, c_v], dim = seq_dim)

        q, k = self.rope(q, k)

        if self.flash:
            orig_dtype = q.dtype
            attn_out = self.attn(q.half(), k.half(), v.half(), softmax_scale = self.attn_scale).to(orig_dtype)
        else:
            attn_out = self.attn(q, k, v, scale = self.attn_scale)

        attn_out = self.head_merge(attn_out)
        attn_out = attn_out[:,:n]

        return self.out(attn_out)

class DiTBlock(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()

        self.mod = DoubleMod(config.d_model)
        self.attn = Attn(config)
        self.mlp = MLP(config.d_model)

        self.lerp_attn = NormalizedLerp(config.d_model, 0.05, config.d_model ** -.5)
        self.lerp_mlp = NormalizedLerp(config.d_model, 0.05, config.d_model ** -.5)
    
    def forward(self, x, t_emb, c):
        # x [b,n,d]
        # t_emb [b,d]
        # c [b,m,d]

        mod1, mod2 = self.mod(t_emb)

        resid_1 = x.clone()
        x = mod1.first_step(x)
        attn_out = self.attn(x, c)
        attn_out = mod1.second_step(attn_out)
        x = self.lerp_attn(x, resid_1)

        resid_2 = x.clone()
        x = mod2.first_step(x)
        x = self.mlp(x)
        x = mod2.second_step(x)
        x = self.lerp_mlp(x, resid_2)

        return x