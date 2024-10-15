import torch
from torch import nn

from .normalization import norm

"""
Modulation layers inject information from timestep/vector into DiT states
"""

class HyperSphereModulation:
  def __init__(self, alpha, beta, gamma = None):
    self.alpha = alpha[:,None,:] # Add seq axis
    self.beta = beta[:,None,:]

    self.gamma = None
    if gamma is not None: self.gamma = gamma[:,None,:]

  def first_step(self, x):
    # SLERP x with beta using alpha as an interpolation factor
    x = norm(x)
    beta = norm(self.beta)
    scale = (1. + self.alpha)
    return norm(x + scale * (beta - x))
  
  def second_step(self, x):
    if self.gamma is None: return x
    return norm(x * self.gamma)
  
class SingleMod(nn.Module):
  def __init__(self, dim):
    super().__init__()

    self.act = nn.GELU()
    self.fc = nn.Linear(dim, 2 * dim)

  def forward(self, x, t):
    t = self.act(t)
    scale, shift = self.mod_params(t).chunk(2, dim = -1)
    mod = HyperSphereModulation(scale, shift)
    return mod.first_step(x)
  
class DoubleMod(nn.Module):
  def __init__(self, dim):
    super().__init__()

    self.act = nn.GELU()
    self.fc = nn.Linear(dim, 6 * dim)
  
  def forward(self, t):
    t = self.act(t)
    params = self.fc(t)
    a_1, b_1, g_1, a_2, b_2, g_2 = params.chunk(6, dim = -1)
    return [
      HyperSphereModulation(a_1, b_1, g_1),
      HyperSphereModulation(a_2, b_2, g_2)
    ]
    