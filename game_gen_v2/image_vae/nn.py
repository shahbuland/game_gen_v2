import torch
from torch import nn
import einops as eo
import torch.nn.functional as F

from game_gen_v2.common.nn.mlp import MLP
from game_gen_v2.common.nn.attention import LinearAttn
from game_gen_v2.common.nn.stacked import StackedLayers
from game_gen_v2.common.nn.normalization import LayerNorm
from game_gen_v2.common.nn.lpips import DinoLPIPS

from game_gen_v2.common.configs import TransformerConfig
from .configs import TransformerVAEConfig

class PatchUpSample2D(nn.Module):
    def __init__(self):
        super().__init__()

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
    
    def forward(self, x):
        # x is [b,n,d]
        b,n,d = x.shape
        x = eo.rearrange(x, 'b (n_y n_x) d -> b d n_y n_x', n_y = round(n**.5))
        x = self.upsample(x)
        x = eo.rearrange(x, 'b d n_y n_x -> b (n_y n_x) d')
        return x

class PatchDownSample2D(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # x is [b,n,d]
        b,n,d = x.shape
        x = eo.rearrange(x, 'b (n_y n_x) d -> b d n_y n_x', n_y = round(n**.5))
        x = self.downsample(x)
        x = eo.rearrange(x, 'b d n_y n_x -> b (n_y n_x) d')
        return x

class MixFFN(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()

        dim_in = config.d_model
        dim_middle = 4 * dim_in

        self.reshape_out = lambda x: eo.rearrange(
            x,
            'b d n_y n_x -> b (n_y n_x) d'
        )

        self.act = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_1 = nn.Conv2d(dim_in,dim_middle,1)
        self.conv_2 = nn.Conv2d(dim_middle, dim_middle, 3, padding = 1, groups = dim_middle)
        self.conv_3 = nn.Conv2d(dim_middle//2,dim_in,1)
    
    def forward(self, x):
        # x is [b,n,d]
        b,n,d = x.shape
        #x = self.reshape_in(x)
        x = eo.rearrange(
            x,
            'b (n_y n_x) d -> b d n_y n_x',
            n_y = round(x.shape[1]**.5)
        )

        x = self.conv_1(x) # [b, d*4, n_y, n_x]
        x = self.conv_2(x) # [b, d*4, n_y, n_x]
        gate, x = x.chunk(2, dim = 1) # 2*[b, d*2, n_y, n_x] 

        gate = self.act(gate)
        x = x * gate
        x = self.conv_3(x)
        x = self.reshape_out(x)
        return x
    
class UpMixFFN(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()

        dim_in = config.d_model
        dim_middle = 4 * dim_in

        self.reshape_out = lambda x: eo.rearrange(
            x,
            'b d n_y n_x -> b (n_y n_x) d'
        )

        self.act = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_1 = nn.Conv2d(dim_in,dim_middle,1)
        self.conv_2 = nn.Conv2d(dim_middle, dim_middle, 3, padding = 1, groups = dim_middle)
        self.conv_3 = nn.Conv2d(dim_middle//2,dim_in,1)
    
    def forward(self, x):
        # x is [b,n,d]
        b,n,d = x.shape
        #x = self.reshape_in(x)
        x = eo.rearrange(
            x,
            'b (n_y n_x) d -> b d n_y n_x',
            n_y = round(x.shape[1]**.5)
        )

        x = self.upsample(x)  # [b, d, 2*n_y, 2*n_x]
        x = self.conv_1(x) # [b, d*4, 2*n_y, 2*n_x]
        x = self.conv_2(x) # [b, d*4, 2*n_y, 2*n_x]
        gate, x = x.chunk(2, dim = 1) # 2*[b, d*2, 2*n_y, 2*n_x] 

        gate = self.act(gate)
        x = x * gate
        x = self.conv_3(x)
        x = self.reshape_out(x)
        return x
    
class DownMixFFN(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()

        dim_in = config.d_model
        dim_middle = 4 * dim_in

        self.reshape_out = lambda x: eo.rearrange(
            x,
            'b d n_y n_x -> b (n_y n_x) d'
        )

        self.act = nn.ReLU()
        self.conv_1 = nn.Conv2d(dim_in,dim_middle,1)
        self.conv_2 = nn.Conv2d(dim_middle, dim_middle, 3, padding = 1, groups = dim_middle)
        self.conv_3 = nn.Conv2d(dim_middle//2,dim_in,1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # x is [b,n,d]
        b,n,d = x.shape
        x = eo.rearrange(
            x,
            'b (n_y n_x) d -> b d n_y n_x',
            n_y = round(x.shape[1]**.5)
        )

        x = self.conv_1(x) # [b, d*4, n_y, n_x]
        x = self.conv_2(x) # [b, d*4, n_y, n_x]
        gate, x = x.chunk(2, dim = 1) # 2*[b, d*2, n_y, n_x] 

        gate = self.act(gate)
        x = x * gate
        x = self.conv_3(x) # [b, d, n_y, n_x]
        x = self.pool(x)  # [b, d, n_y/2, n_x/2]
        x = self.reshape_out(x)
        return x
 
class Block(nn.Module):
    def __init__(self, config : TransformerConfig, mode="same"):
        super().__init__()
        self.config = config

        self.norm_1 = LayerNorm(config.d_model)
        self.norm_2 = LayerNorm(config.d_model)

        self.attn = LinearAttn(config)

        self.ffn = None
        self.resample = None
        if mode == 'up':
            self.ffn = UpMixFFN(config)
            self.resample = PatchUpSample2D()
        elif mode == 'down':
            self.ffn = DownMixFFN(config)
            self.resample = PatchDownSample2D()
        else:
            self.ffn = MixFFN(config)
    
    def forward(self, x):
        # x is [b,n,d] and has two special tokens at start
        resid_1 = x.clone()
        x = self.norm_1(x)
        x = self.attn(x)
        x = x + resid_1
        
        resid_2 = x.clone()
        x = self.norm_2(x)

        x = self.ffn(x)

        if self.resample:
            resid_2 = self.resample(resid_2)
        return x + resid_2

class MultiDownBlock(nn.Module):
    def __init__(self, config, mode = 'down'):
        super().__init__()

        self.main_layer = Block(config, mode = mode)
        self.extra_blocks = nn.ModuleList([Block(config)] * config.n_sublayers)

    def forward(self, x):
        x = self.main_layer(x)
        for block in self.extra_blocks:
            x = block(x)
        return x

class MultiUpBlock(MultiDownBlock):
    def __init__(self, config):
        super().__init__(config, mode = 'up')

def mu_logvar_sample(mu, logvar):
    # given both [b,c,h,w] mu and logvar
    # sample [b,c,h,w] z (image)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z

def calc_kl_loss(mu, logvar):
    # KL divergence between N(mu, var) and N(0, 1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
    return kl.mean()

class Encoder(nn.Module):
    def __init__(self, config : TransformerVAEConfig):
        super().__init__()

        self.config = config
        self.layers = StackedLayers(MultiDownBlock, config)
        self.patch_proj = nn.Conv2d(
            config.channels,
            config.d_model,
            config.patch_size,
            config.patch_size
        )

        downsampling_factor = 2 ** config.n_layers # Each layer downsamples
        # 512 -> 8 = 3 layers

        # For depatchify
        self.n_l = (config.sample_size // config.patch_size) // downsampling_factor
        self.l_p = config.patch_size // downsampling_factor


        self.final_norm = LayerNorm(config.d_model)
        proj_dim = config.latent_channels * self.l_p**2

        self.proj_out_mu = nn.Linear(config.d_model, proj_dim)
        self.proj_out_logvar = nn.Linear(config.d_model, proj_dim)

        self.depatchify = lambda x: eo.rearrange(
            x,
            'b (n_y n_x) (y x c) -> b c (n_y y) (n_x x)',
            n_y = self.n_l,
            y = self.l_p,
            x = self.l_p
        )
    
    def forward(self, x):
        x = self.patch_proj(x) # -> [b,d,n_y,n_x]
        x = x.flatten(2).transpose(1,2) # -> [b,d,n] -> [b,n,d]
        x = self.layers(x)
 
        x = self.final_norm(x)
        mu = self.proj_out_mu(x)
        logvar = self.proj_out_logvar(x)

        mu = self.depatchify(mu)
        logvar = self.depatchify(logvar)
        z = mu_logvar_sample(mu, logvar)
        
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, config : TransformerVAEConfig):
        super().__init__()

        self.config = config
        self.layers = StackedLayers(MultiUpBlock, config)

        downsampling_factor = 2 ** config.n_layers
        latent_patch_size = config.patch_size // downsampling_factor
        self.patch_proj = nn.Conv2d(
            config.latent_channels,
            config.d_model,
            latent_patch_size,
            latent_patch_size
        )

        self.final_norm = LayerNorm(config.d_model)
        proj_dim = config.channels * config.patch_size**2

        self.proj_out = nn.Linear(config.d_model, proj_dim)

        self.depatchify = lambda x: eo.rearrange(
            x,
            'b (n_y n_x) (y x c) -> b c (n_y y) (n_x x)',
            n_y = config.sample_size//config.patch_size,
            y = config.patch_size,
            x = config.patch_size
        )
    
    def forward(self, x):
        x = self.patch_proj(x).flatten(2).transpose(1,2)
        x = self.layers(x)
        x = self.final_norm(x)
        x = self.proj_out(x)

        x = self.depatchify(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, config : TransformerVAEConfig):
        super().__init__()

        self.config = config
        self.layers = StackedLayers(MultiDownBlock, config)
        self.patch_proj = nn.Conv2d(
            config.channels,
            config.d_model,
            config.patch_size,
            config.patch_size
        )

        self.final_norm = LayerNorm(config.d_model)
        self.proj_out = nn.Linear(config.d_model, 1)

    def forward(self, x):
        x = self.patch_proj(x).flatten(2).transpose(1,2)
        x = self.layers(x)
        x = self.final_norm(x)
        scores = self.proj_out(x)

        return scores

class TransformerVAE(nn.Module):
    def __init__(self, config : TransformerVAEConfig):
        super().__init__()

        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        #self.disc = Discriminator(config)
        #self.lpips = DinoLPIPS('facebook/dinov2-base', [3,6,8])

    def compute_disc_loss(self, x_real, x_fake):
        scores_real = self.disc(x_real)
        scores_fake = self.disc(x_fake)

    def compute_gan_loss(self):
        pass # TODO

    def compute_lpips_loss(self):
        pass # TODO

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_rec = self.decoder(z)

        rec_loss = F.mse_loss(x, x_rec)
        kl_loss = calc_kl_loss(mu, logvar)

        extra = {}
        extra['rec_loss'] = rec_loss.item()
        extra['kl_term'] = kl_loss.item()

        total_loss = 0.

        total_loss += rec_loss
        total_loss += self.config.kl_weight * kl_loss

        return total_loss, extra

if __name__ == "__main__":
    model_cfg = TransformerVAEConfig.from_yaml("configs/image_vae/tiny_deep.yml")
    model = TransformerVAE(model_cfg)

    x = torch.randn(2, model_cfg.channels, model_cfg.sample_size, model_cfg.sample_size)
    print("Original", x.shape)
    loss = model(x)
    print(loss)