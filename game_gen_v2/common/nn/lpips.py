"""
LPIPS with DINO
"""

import torch
from torch import nn
from transformers import AutoModel
import torch.nn.functional as F

def dino_proc(x):
    """
    DINO processor as a function

    x is [b,c,h,w] [-1,1] range
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

class DinoLPIPS(nn.Module):
    def __init__(self, dino_id, feature_inds):
        super().__init__()

        self.inds = feature_inds
        self.model = AutoModel.from_pretrained(dino_id)
    
    def forward(self, x, x_rec):
        x = dino_proc(x)
        h = self.model(pixel_values=x,output_hidden_states=True).hidden_states
        h = [h[i] for i in self.inds]

        x_rec = dino_proc(x_rec)
        h_rec = self.model(pixel_values=x_rec,output_hidden_states=True).hidden_states
        h_rec = [h_rec[i] for i in self.inds]

        total_loss = 0.
        for (h_i, h_rec_i) in zip(h, h_rec):
            total_loss += F.mse_loss(h_i, h_rec_i)

        return total_loss
