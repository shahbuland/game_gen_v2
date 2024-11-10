import torch
from torch import nn
import einops as eo

from ..configs import TransformerConfig

class MLP(nn.Module):
  """
  MLP with inferred middle/out/etc.

  :param use_scale: Set to true if using normalized transformer
  """
  def __init__(self, dim, dim_out = None, d_middle = None, n_extra_layers : int = 0):
    super().__init__()
    if dim_out is None:
      dim_out = dim
    if d_middle is None:
      d_middle = 4 * dim

    self.fc1 = nn.Linear(dim, d_middle) # hiddden size in transformer MLPs is normally 4x the input size
    self.act = nn.GELU()
    self.fc2 = nn.Linear(d_middle, dim_out)

    self.hidden_layers = nn.ModuleList([nn.Linear(d_middle, d_middle)] * n_extra_layers)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)

    for layer in self.hidden_layers:
      x = layer(x)
      x = self.act(x)

    x = self.fc2(x)
    return x
  
class MixFFN(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()

        dim_in = config.d_model
        dim_middle = 4 * dim_in

        self.reshape_in = lambda x: eo.rearrange(
            x,
            'b (n_y n_x) d -> b d n_y n_x',
            n_y = config.sample_size // config.patch_size
        )
        self.reshape_out = lambda x: eo.rearrange(
            x,
            'b d n_y n_x -> b (n_y n_x) d'
        )

        self.act = nn.ReLU()
        self.conv_1 = nn.Conv2d(dim_in,dim_middle,1)
        self.conv_2 = nn.Conv2d(dim_middle, dim_middle, 3, padding = 1, groups = dim_middle)
        self.conv_3 = nn.Conv2d(dim_middle//2,dim_in,1)
    
    def forward(self, x):
        # x is [b,n,d]
        b,n,d = x.shape
        x = self.reshape_in(x)
        x = self.conv_1(x) # [b, d, n_y, n_x]
        x = self.conv_2(x) # [b, d*4, n_y, n_x]
        gate, x = x.chunk(2, dim = 1) # 2*[b, d*2, n_y, n_x] 

        gate = self.act(gate)
        x = x * gate
        x = self.conv_3(x)
        x = self.reshape_out(x)
        return x
    
class MixFFN3D(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()

        dim_in = config.d_model
        dim_middle = 4 * dim_in

        self.reshape_in = lambda x: eo.rearrange(
            x,
            'b (n_t n_y n_x) d -> b d n_t n_y n_x',
            n_y = config.sample_size // config.patch_size,
            n_t = config.temporal_sample_size // config.temporal_patch_size
        )
        self.reshape_out = lambda x: eo.rearrange(
            x,
            'b d n_t n_y n_x -> b (n_t n_y n_x) d'
        )

        self.act = nn.ReLU()
        self.conv_1 = nn.Conv3d(dim_in, dim_middle, 1)
        self.conv_2 = nn.Conv3d(dim_middle, dim_middle, 3, padding = 1, groups = dim_middle)
        self.conv_3 = nn.Conv3d(dim_middle//2, dim_in, 1)

    def forward(self, x):
        # x is [b,n,d]
        x = self.reshape_in(x)
        x = self.conv_1(x)
        x = self.conv_2(x) # [b, d*4, n_t, n_y, n_x]
        gate, x = x.chunk(2, dim = 1)
        gate = self.act(gate)
        x = x * gate
        x = self.conv_3(x)
        x = self.reshape_out(x)
        return x