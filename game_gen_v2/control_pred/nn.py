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

        self.attn = LinearAttn(config)
        self.ffn = MixFFN3D(config)

    def forward(self, x):
        # x is [b,n,d] and has two special tokens at start
        resid_1 = x.clone()
        x = self.norm_1(x)
        x = self.attn(x)
        x = x + resid_1
        
        resid_2 = x.clone()
        x = self.norm_2(x)
        x[:,2:] = self.ffn(x[:,2:]) # Don't use MLP on the special tokens
        return x + resid_2

class ControlPredCore(nn.Module):
    def __init__(self, config : ControlPredConfig):
        super().__init__()

        self.config = config
        self.layers = StackedLayers(ControlPredBlock, config)

        self.button_token = nn.Parameter(torch.randn(config.d_model) * 0.02)
        self.mouse_token = nn.Parameter(torch.randn(config.d_model) * 0.02)

        self.patch_proj = nn.Conv3d(
            config.channels,
            config.d_model,
            (config.temporal_patch_size, config.patch_size, config.patch_size),
            (config.temporal_patch_size, config.patch_size, config.patch_size)
        )
        self.patch_proj_img = nn.Conv2d(
            config.channels,
            config.d_model,
            config.patch_size,
            config.patch_size
        ) # For image part

        self.fc_btn = nn.Linear(config.d_model, config.n_controls)
        self.fc_mouse = nn.Linear(config.d_model, config.n_mouse_axes)

        self.final_norm = LayerNorm(config.d_model)

    def forward(self, x):
        b = x.shape[0]
        # x is [b,t,c,h,w] video
        #x_final = x[:,-1]
        x = x.transpose(1,2) # -> [b,c,t,h,w]
        x = self.patch_proj(x).flatten(2) # -> [b,d,n]
        x = x.transpose(1,2) # -> [b,n,d]

        #x_final = self.patch_proj_img(x_final).flatten(2).transpose(1,2) # -> [b,m,d]

        btn_token = eo.repeat(self.button_token, 'd -> b n d', n = 1, b = b)
        m_token = eo.repeat(self.mouse_token, 'd -> b n d', n = 1, b = b)

        x = torch.cat([btn_token, m_token, x], dim = 1) # -> [b,n+1,d]
        x = self.layers(x)
        x = x[:,:2] # [b,2,d]
        x = self.final_norm(x)
        btn_pred = self.fc_btn(x[:,0])
        mouse_pred = self.fc_mouse(x[:,1])

        return btn_pred, mouse_pred
    
# Custom weighted loss functions to help with
# high prevalence of zeros in data
class WeightedMSELoss(nn.Module):
    def __init__(self, zero_weight):
        super().__init__()

        self.zero_weight = zero_weight
        self.non_zero_weight = 1 - zero_weight
    
    def forward(self, pred, target):
        # target pred both [b,2]
        sqr_err = (pred - target) ** 2

        # Weight is zero_weight if both |x|+|y|=0
        w = torch.where(
            target.abs().sum(-1, keepdim = True) == 0,
            torch.ones_like(target) * self.zero_weight,
            torch.ones_like(target) * self.non_zero_weight
        )

        return (sqr_err * w).mean()

class WeightedBCELogitsLoss(nn.Module):
    def __init__(self, btn_zero_weights):
        super().__init__()

        self.btn_weights = torch.tensor(
            [1 - prob for prob in btn_zero_weights]
        )
    
    def forward(self, pred, target):
        w_zero = self.btn_weights.to(device=pred.device,dtype=pred.dtype)
        w = torch.where(
            target==0,
            w_zero,
            1-w_zero
        )

        bce_loss = F.binary_cross_entropy_with_logits(
            pred,
            target,
            reduction='none'
        )
        return (bce_loss * w).mean()
        

class ControlPredModel(nn.Module):
    def __init__(self, config : ControlPredConfig):
        super().__init__()

        self.config = config
        self.core = ControlPredCore(config)
        self.n_controls = config.n_controls

        #self.btn_loss = WeightedBCELogitsLoss(config.button_zero_weights)
        #self.mouse_loss = WeightedMSELoss(config.mouse_zero_weights)

        self.btn_loss = nn.BCEWithLogitsLoss()
        self.mouse_loss = nn.MSELoss()
        
    def forward(self, x, labels):
        # x is [b,t,c,h,w] video
        # labels is [n_controls] where
        # labels[:n_controls] is key presses
        # labels[n_controls:] is mouse axes
        button_preds, mouse_preds = self.core(x) # -> [b,n_controls+2]

        button_labels = labels[:,:self.n_controls]
        mouse_targets = labels[:,self.n_controls:] 

        button_loss = self.btn_loss(button_preds, button_labels)
        mouse_loss = self.mouse_loss(mouse_preds, mouse_targets)

        loss = self.config.btn_weight * button_loss + self.config.mouse_weight * mouse_loss

        # Calculate button prediction accuracy
        with torch.no_grad():
            button_preds_binary = (torch.sigmoid(button_preds) > 0.5).float()
            button_accuracy = (button_preds_binary == button_labels).float().mean()

            nonzero_btn_idx = button_labels != 0
            
            nonzero_button_acc = (button_preds_binary[nonzero_btn_idx] == button_labels[nonzero_btn_idx]).float().mean()

        extra = {
            'button_pred_loss' : button_loss.item(),
            'mouse_pred_loss' : mouse_loss.item(),
            'button_accuracy' : button_accuracy.item(),
            'nonzero_button_acc' : nonzero_button_acc.item()
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
