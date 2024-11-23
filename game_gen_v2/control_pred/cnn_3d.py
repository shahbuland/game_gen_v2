import torch
from torch import nn
import einops as eo
import torch.nn.functional as F

from game_gen_v2.common.nn.mlp import MLP
from game_gen_v2.common.nn.attention import Attn, LinearAttn
from game_gen_v2.common.configs import TransformerConfig
from game_gen_v2.common.nn.stacked import StackedLayers
from game_gen_v2.common.nn.normalization import LayerNorm

from .configs import ControlPredictorConfig

class Transformer(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()

        if config.attn_impl == "flash":
            self.attn = Attn(config)
        elif config.attn_impl == "linear":
            self.attn = LinearAttn(config)

        self.norm_1 = LayerNorm(config.d_model)
        self.norm_2 = LayerNorm(config.d_model)

        self.ffn = MLP(config.d_model)
    
    def forward(self, x):
        resid_1 = x.clone()
        x = self.norm_1(x)
        x = self.attn(x)
        x = x + resid_1

        resid_2 = x.clone()
        x = self.norm_2(x)
        x = self.ffn(x)
        return x + resid_2

class Encoder(nn.Module):
    def __init__(self, config : ControlPredictorConfig):
        super().__init__()

        self.config = config

        self.conv_block = nn.Sequential(
            self.make_layer_(3, 32, (3,7,7), (1,2,2), (1, 3, 3)), # -> 32x64x64
            self.make_layer_(32, 64, (3,3,3), (1,2,2), (1, 1, 1)), # -> 16x32x32
            self.make_layer_(64, 128, (3,3,3), (1,2,2), (1, 1, 1)), # -> 16x16x16
            self.make_layer_(128, 256, (3,3,3), (1,2,2), (1,1,1)) # -> 8x8x8
        )
        self.proj = nn.Linear(256*8*8, config.attn_d_model)
        self.tform_cfg = TransformerConfig(
            n_layers=config.attn_n_layers,
            n_heads=config.attn_n_heads, 
            d_model=config.attn_d_model,
            attn_impl=config.attn_impl
        )
        self.attn_layers = StackedLayers(Transformer, self.tform_cfg)
        
        self.final_norm = LayerNorm(config.attn_d_model)
        self.final_btn = nn.Linear(config.attn_d_model, config.n_controls)
        self.final_mouse = nn.Linear(config.attn_d_model, config.n_mouse_axes)

    def make_layer_(self, fi, fo, k, s, p):
        return nn.Sequential(
            nn.Conv3d(fi, fo, k, s, p),
            nn.BatchNorm3d(fo),
            nn.SiLU()
        )
    
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv_block(x)
        x = eo.rearrange(x, 'b c t h w -> b t (c h w)')
        x = self.proj(x)
        x = self.attn_layers(x)
        x = x.mean(1)

        x = self.final_norm(x)
    
        return self.final_btn(x), self.final_mouse(x)

class ControlPredictor(nn.Module):
    def __init__(self, config : ControlPredictorConfig):
        super().__init__()

        self.config = config
        self.core = Encoder(config)
        self.n_controls = config.n_controls

        self.btn_loss = nn.BCEWithLogitsLoss()
        self.mouse_loss = nn.MSELoss()

    def forward(self, x, labels):
        # x is [b,t,c,h,w] video
        # labels is [b,t,n_controls] where
        # labels[:,:,:n_controls] is key presses
        # labels[:,:,n_controls:] is mouse axes
        button_preds, mouse_preds = self.core(x) # -> [b,t,n_controls+2]

        middle_idx = labels.shape[1]//2
        button_labels = labels[:,middle_idx,:self.n_controls]
        mouse_targets = labels[:,middle_idx,-2:] 

        btn_loss = self.btn_loss(button_preds, button_labels)
        m_loss = self.mouse_loss(mouse_preds, mouse_targets)

        loss = btn_loss + m_loss

        # Calculate button prediction accuracy
        with torch.no_grad():
            # Calculate accuracy for middle frame predictions
            button_preds_binary = (torch.sigmoid(button_preds) > 0.5).float()
            button_accuracy = (button_preds_binary == button_labels).float().mean()

            # Calculate non-zero button accuracy
            nonzero_btn_idx = button_labels != 0
            nonzero_button_acc = (button_preds_binary[nonzero_btn_idx] == button_labels[nonzero_btn_idx]).float().mean()

            all_zero_buttons = (button_labels.sum(dim=-1) == 0).float().mean()
            all_zero_mouse = (mouse_targets.abs().sum(dim=-1) == 0).float().mean()

        extra = {
            'button_pred_loss': btn_loss.item(),
            'mouse_pred_loss': m_loss.item(),
            'button_sensitivity': nonzero_button_acc.item(),
        }
        return loss, extra


if __name__ == "__main__":
    from game_gen_v2.data.loader import create_loader
    from .configs import ControlPredDataConfig, ControlPredictorConfig
    from .augs import augtform_video, transform_video
    import time

    data_cfg = ControlPredDataConfig.from_yaml("configs/control_pred/v2/data_v2.yml")
    model_cfg = ControlPredictorConfig.from_yaml("configs/control_pred/v2/cnn_v2.yml")

    model = ControlPredictor(model_cfg)
    model.to(device='cuda', dtype=torch.bfloat16)

    dataset_kwargs = {
        'data_dir': data_cfg.data_path,
        'frame_count': model_cfg.temporal_sample_size,
        'image_transform': augtform_video
    }

    #dataloader_kwargs = {
    #    'batch_size': 64,
    #    'num_workers': 4,
    #    'pin_memory': True,
    #    'prefetch_factor': 8
    #}

    dataloader_kwargs = {
        'batch_size': 64,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 8
    }

    opt = torch.optim.AdamW(model.parameters(), lr = 1.0e-4)

    loader = create_loader(dataset_kwargs, dataloader_kwargs)


    from tinygrad.helpers import Timing

    loader_iter = iter(loader)
    with Timing("Time for 20 steps: "):
        for _ in range(20):
            with Timing("Time to get 1 batch: "): 
                batch = next(loader_iter)
            with Timing("Time for forward/bwd pass: "):
                opt.zero_grad()
                vid, ctrl = batch
                vid = vid.to(device='cuda',dtype=torch.bfloat16)
                ctrl = ctrl.to(device='cuda',dtype=torch.bfloat16)
                loss, extra = model(vid, ctrl)
                loss.backward()
                opt.step()
    

# Timing notes:
# all B = 64, 20 steps
# n_workers = 4, prefetch = 8 => 66s
# n_workers = 0, prefetch = 0 => 137s
# n_workers = 4, prefetch = 0 => 96s
# n_workers = 8, prefetch = 0 => 87s
# 8, 8 => 94s
