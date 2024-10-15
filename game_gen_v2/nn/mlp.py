import torch
from torch import nn

from .normalization import ScalingLayer

class MLP(nn.Module):
  """
  MLP with inferred middle/out/etc.

  :param use_scale: Set to true if using normalized transformer
  """
  def __init__(self, dim, dim_out = None, d_middle = None, use_scale = True):
    super().__init__()
    if dim_out is None:
      dim_out = dim
    if d_middle is None:
      d_middle = 4 * dim

    self.fc1 = nn.Linear(dim, d_middle) # hiddden size in transformer MLPs is normally 4x the input size
    self.act = nn.GELU()
    self.fc2 = nn.Linear(d_middle, dim_out)

    if use_scale:
        self.scale = nn.Parameter(torch.zeros(d_middle))
    self.use_scale = use_scale
    self.v_scale = dim ** .5

  def forward(self, x):
    x = self.fc1(x)
    if self.use_scale: x *= (1. + self.scale)[None,None,:] * self.v_scale
    x = self.act(x)
    x = self.fc2(x)
    return x