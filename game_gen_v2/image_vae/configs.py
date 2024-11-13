from dataclasses import dataclass, field
from typing import List
from game_gen_v2.common.configs import TransformerConfig

@dataclass
class TransformerVAEConfig(TransformerConfig):
    n_sublayers : int = 0
    kl_weight : float = 1.0e-5
    adv_wight : float = 0.1
    latent_channels : int = 64