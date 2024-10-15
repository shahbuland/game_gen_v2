from transformers import AutoModel, AutoProcessor
import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from .mlp import MLP
from ..configs import ModelConfig
from ..utils import freeze

def dino_proc(x):
    """
    DINO processor as a function to map directly from [-1,1] tensors

    x : [b, c, h, w]
    """
    # Convert from [-1, 1] to [0, 1]
    x = (x + 1) / 2

    # Resize
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    x = (x - mean) / std

    return x

def patch_pool(x, config : ModelConfig):
    """
    Pooling to reduce sequence length by powers of 4
    Allows longer seq lengths while still using REPA
    """
    # x : [b, n, d] -> [b,n//4,d*4]
    n_patches = config.sample_size // config.patch_size
    x = eo.rearrange(x, 'b (n_p_h n_p_w) d -> b n_p_h n_p_w d', n_p_h = n_patches)

    top_rows = x[:,::2]
    bottom_rows = x[:,1::2]

    top_rows = eo.rearrange(top_rows, 'b n_p_h n_p_w d -> b (n_p_h n_p_w) d')
    top_left_patch = top_rows[:,::2]
    top_right_patch = top_rows[:,1::2]

    bottom_rows = eo.rearrange(bottom_rows, 'b n_p_h n_p_w d -> b (n_p_h n_p_w) d')
    bottom_left_patch = bottom_rows[:,::2]
    bottom_right_patch = bottom_rows[:,1::2]

    # Now should all be [b,k,d] where k is n / 4
    # want [b,k,d*4]
    pooled = torch.cat([top_left_patch, top_right_patch, bottom_left_patch, bottom_right_patch], dim = -1)
    return pooled

class REPA(nn.Module):
    def __init__(self, config : ModelConfig, dino_path = "facebook/dinov2-small"):
        super().__init__()

        self.pool_factor = config.repa_pool_factor

        self.dino = AutoModel.from_pretrained(dino_path)
        freeze(self.dino)
        self.batch_size = config.repa_batch_size

        self.mlp = MLP(
            config.d_model * (self.pool_factor ** 2),
            dim_out = self.dino.config.hidden_size,
            d_middle = config.d_model * 4
        )

        self.patch_pool = None
        if self.pool_factor > 1:
            self.patch_pool = lambda x: patch_pool(x, config)
    
    @torch.no_grad()
    def dino_features(self, x):
        # x is [b,c,h,w] [-1,1]
        inputs = dino_proc(x)
        input_batches = inputs.split(self.batch_size)

        h_all = []
        for batch in input_batches:
            h = self.dino(pixel_values = batch, output_hidden_states = True).hidden_states[-2][:,1:] # skip cls
            h_all.append(h)

        h_all = torch.cat(h_all)

        return h_all
    
    def forward(self, x, features):
        # x [b,c,h,w] original image
        # features [b,n,d] latents

        if self.pool_factor > 1:
            features = self.patch_pool(features)

        # both become b,n,d
        h = self.dino_features(x)
        h_rft = self.mlp(features)

        h = F.normalize(h, p = 2, dim = -1)
        h_rft = F.normalize(h_rft, p = 2, dim = -1)

        cos_sims = torch.einsum('bnd,bnd->bn', h, h_rft)
        return -cos_sims.mean()