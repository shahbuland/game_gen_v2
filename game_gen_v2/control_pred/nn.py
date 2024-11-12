import torch
from torch import nn
import einops as eo
import torch.nn.functional as F

from game_gen_v2.common.nn.mlp import MixFFN3D, MLP
from game_gen_v2.common.nn.attention import LinearAttn
from game_gen_v2.common.nn.stacked import StackedLayers
from game_gen_v2.common.nn.normalization import LayerNorm

from game_gen_v2.common.configs import TransformerConfig
from .configs import ControlPredConfig

class ControlPredBlock(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()
        self.config = config

        self.norm_1 = LayerNorm(config.d_model)
        self.norm_2 = LayerNorm(config.d_model)
        self.n_video_tokens = config.n_video_tokens

        self.attn = LinearAttn(config)
        self.ffn = MixFFN3D(config)
        self.ffn_ctrl = MLP(config.d_model)

    def forward(self, x):
        # x is [b,n,d] and has two special tokens at start
        resid_1 = x.clone()
        x = self.norm_1(x)
        x = self.attn(x)
        x = x + resid_1
        
        resid_2 = x.clone()
        x = self.norm_2(x)

        x_video = x[:,:self.n_video_tokens]
        x_ctrl = x[:,self.n_video_tokens:]

        x_video = self.ffn(x_video)
        x_ctrl = self.ffn_ctrl(x_ctrl)
        x = torch.cat([x_video, x_ctrl], dim = 1)
        return x + resid_2

class ControlPredCore(nn.Module):
    def __init__(self, config : ControlPredConfig):
        super().__init__()

        self.config = config
        self.layers = StackedLayers(ControlPredBlock, config)

        self.n_video_tokens = config.n_video_tokens
        self.n_frames = config.temporal_sample_size//config.temporal_patch_size
        def randn_param(*shape):
            return nn.Parameter(torch.randn(*shape) * 0.02)
        
        self.button_tokens = randn_param(self.n_frames, config.d_model)
        self.mouse_tokens = randn_param(self.n_frames, config.d_model)

        self.patch_proj = nn.Conv3d(
            config.channels,
            config.d_model,
            (config.temporal_patch_size, config.patch_size, config.patch_size),
            (config.temporal_patch_size, config.patch_size, config.patch_size)
        )

        self.fc_btn = nn.Linear(config.d_model, config.n_controls)
        self.fc_mouse = nn.Linear(config.d_model, config.n_mouse_axes)

        self.final_norm = LayerNorm(config.d_model)

    def forward(self, x):
        """
        Takes [b,t,c,h,w] video, and returns predicted controls [b,t,n_controls] for every frame
        """
        b = x.shape[0]
        # x is [b,t,c,h,w] video
        x = x.transpose(1,2) # -> [b,c,t,h,w]
        x = self.patch_proj(x).flatten(2) # -> [b,d,n]
        x = x.transpose(1,2) # -> [b,n,d]

        btn_tokens = eo.repeat(self.button_tokens, 'n d -> b n d', b = b)
        m_tokens = eo.repeat(self.mouse_tokens, 'n d -> b n d', b = b)
        x = torch.cat([x, btn_tokens, m_tokens], dim = 1) # -> [b,N,d]

        x = self.layers(x)
        x = x[:,self.n_video_tokens:] # [b,2*temporal_n_patches,d]
        x = self.final_norm(x)

        x_btn = x[:,:self.n_frames]
        x_mouse = x[:,self.n_frames:]

        btn_pred = self.fc_btn(x_btn) # [b,t,n_btns]
        mouse_pred = self.fc_mouse(x_mouse) # [b,t,2]

        return btn_pred, mouse_pred

class ControlPredModel(nn.Module):
    def __init__(self, config : ControlPredConfig):
        super().__init__()

        self.config = config
        self.core = ControlPredCore(config)
        self.n_controls = config.n_controls

        self.btn_loss = nn.BCEWithLogitsLoss()
        self.mouse_loss = nn.MSELoss()

    def forward(self, x, labels):
        # x is [b,t,c,h,w] video
        # labels is [b,t,n_controls] where
        # labels[:,:,:n_controls] is key presses
        # labels[:,:,n_controls:] is mouse axes
        button_preds, mouse_preds = self.core(x) # -> [b,t,n_controls+2]

        button_labels = labels[:,:,:self.n_controls]
        mouse_targets = labels[:,:,self.n_controls:] 

        btn_loss = self.btn_loss(button_preds, button_labels)
        m_loss = self.mouse_loss(mouse_preds, mouse_targets)

        loss = self.config.btn_weight * btn_loss + self.config.mouse_weight * m_loss

        # Calculate button prediction accuracy
        with torch.no_grad():
            # Calculate accuracy for the last frame
            button_preds_binary_last = (torch.sigmoid(button_preds[:, -1]) > 0.5).float()
            button_labels_last = button_labels[:, -1]
            button_accuracy = (button_preds_binary_last == button_labels_last).float().mean()

            # Calculate non-zero button accuracy for the last frame
            nonzero_btn_idx_last = button_labels_last != 0
            nonzero_button_acc = (button_preds_binary_last[nonzero_btn_idx_last] == button_labels_last[nonzero_btn_idx_last]).float().mean()

            all_zero_buttons = (button_labels.sum(dim=-1) == 0).float().mean()
            all_zero_mouse = (mouse_targets.abs().sum(dim=-1) == 0).float().mean()

        extra = {
            'button_pred_loss': btn_loss.item(),
            'mouse_pred_loss': m_loss.item(),
            'button_accuracy': button_accuracy.item(),
            'button_sensitivity': nonzero_button_acc.item(),
            'prop_all_zero_buttons': all_zero_buttons.item(),
            'prop_all_zero_mouse': all_zero_mouse.item()
        }
        return loss, extra

if __name__ == "__main__":
    import torch
    from game_gen_v2.control_pred.configs import ControlPredConfig

    # Create a config
    config = ControlPredConfig.from_yaml("game_gen_v2/configs/control_pred_tiny.yml")

    # Create the model
    model = ControlPredModel(config)

    # Create random input tensor
    batch_size = 4
    x = torch.randn(batch_size, 32, 4, 32, 32)

    # Create random labels
    labels = torch.zeros(batch_size, config.n_controls + config.n_mouse_axes)
    labels[:, :config.n_controls] = torch.randint(0, 2, (batch_size, config.n_controls)).float()
    labels[:, config.n_controls:] = torch.randn(batch_size, config.n_mouse_axes)

    # Forward pass
    loss, extra = model(x, labels)

    print(f"Loss: {loss.item()}")
    print(f"Extra info:")
    for key, value in extra.items():
        print(f"  {key}: {value}")
