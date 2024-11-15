import torch
from torch import nn
import einops as eo
import torch.nn.functional as F

from game_gen_v2.common.nn.mlp import MLP
from game_gen_v2.common.nn.attention import Attn, LinearAttn
from game_gen_v2.common.configs import TransformerConfig
from game_gen_v2.common.nn.stacked import StackedLayers
from game_gen_v2.common.nn.normalization import LayerNorm

from .configs import ControlPredResConfig

SameConv = lambda fi, fo: nn.Conv3d(fi, fo, 3, 1, 1, bias=False)

class ResBlock(nn.Module):
    def __init__(self, fi, fo):
        super().__init__()

        groups1 = 32
        groups2 = 32
        if fi <= 32:
            groups1 = fi//2
        if fo <= 32:
            groups2 = fo//2

        self.gn1 = nn.GroupNorm(groups1, fi)
        self.conv1 = SameConv(fi,fo)
        #self.gn2 = nn.GroupNorm(groups2, fo)
        #self.conv2 = SameConv(fo, fo)

        self.sc = nn.Sequential()
        if fi != fo:
            self.sc = nn.Sequential(
                nn.Conv3d(fi,fo,1,bias=False)
            )
        
        self.act = nn.SiLU()
    
    def forward(self, x):
        res = self.sc(x)
        x = self.act(x)
        x = self.gn1(x)
        x = self.conv1(x)
        #x = self.act(x)
        #x = self.gn2(x)
        #x = self.conv2(x)

        x = x + res
        return x

def make_res_chain(fi, fo, n_layers):
    layers = [ResBlock(fi, fo)]
    for _ in range(1, n_layers):
        layers.append(ResBlock(fo, fo))
    return nn.Sequential(*layers)

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
    def __init__(self, config : ControlPredResConfig):
        super().__init__()

        self.config = config

        self.conv_in = nn.Conv3d(3, config.channel_counts[0], (3, 7, 7), (1, 2, 2), (1, 3, 3))
        
        ch = config.channel_counts
        res_n = config.res_blocks
        self.layers = nn.ModuleList([
            make_res_chain(ch[i], ch[i+1], res_n[i])
        for i in range(self.config.n_layers)])

        self.spatial_pool = nn.AvgPool3d(
            kernel_size=(1,2,2),
            stride=(1,2,2),
            padding=(0,0,0)
        )

        self.n_frames = config.temporal_sample_size
        self.cls_tokens = nn.Parameter(torch.randn(2*self.n_frames, config.attn_d_model) * 0.02)
        self.attn_proj = nn.Linear(ch[-1] * 7 * 7, config.attn_d_model)

        self.tform_cfg = TransformerConfig(
            n_layers = config.attn_n_layers,
            n_heads = config.attn_n_heads,
            d_model = config.attn_d_model,
            attn_impl = "flash"
        )
        self.transformer = StackedLayers(Transformer, self.tform_cfg)

        self.fc_btn = nn.Linear(config.attn_d_model, config.n_controls)
        self.fc_mouse = nn.Linear(config.attn_d_model, config.n_mouse_axes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4) # -> [b,c,t,h,w] for conv3d
        x = self.conv_in(x)
        x = self.spatial_pool(x)

        for layer in self.layers:
            x = layer(x)
            x = self.spatial_pool(x)

        # -> [b, 512, n_frames, 7, 7] in standard setup
        x = eo.rearrange(x, 'b c n h w -> b n (c h w)')
        x = self.attn_proj(x) # [b,n,d]
        
        b,n,d = x.shape
        cls_toks = eo.repeat(self.cls_tokens, 'n d -> b n d', b = b)
        x = torch.cat([x, cls_toks], dim = 1)
        x = self.transformer(x)
        x = x[:,n:] # [b, n_frames, d]

        x_btn = x[:,:self.n_frames]
        x_mouse = x[:,self.n_frames:]

        btn_pred = self.fc_btn(x_btn) # [b, n_frames, buttons]
        mouse_pred = self.fc_mouse(x_mouse) # [b, n_frames, 2]

        return btn_pred, mouse_pred

class ControlPredResModel(nn.Module):
    def __init__(self, config : ControlPredResConfig):
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

        button_labels = labels[:,:,:self.n_controls]
        mouse_targets = labels[:,:,self.n_controls:] 

        btn_loss = self.btn_loss(button_preds, button_labels)
        m_loss = self.mouse_loss(mouse_preds, mouse_targets)

        loss = btn_loss + m_loss

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
            'button_sensitivity': nonzero_button_acc.item(),
        }
        return loss, extra


if __name__ == "__main__":
    cfg = ControlPredResConfig().from_yaml("configs/control_pred/resnet_config.yml")
    model = Encoder(cfg)
    model.to(device='cuda',dtype=torch.bfloat16)

    x = torch.randn(1, 3, 16, 224, 224,device='cuda',dtype=torch.bfloat16)
    y, z = model(x)
    print(x.shape)
    print(y.shape)
    print(z.shape)



