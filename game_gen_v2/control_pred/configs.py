from dataclasses import dataclass, field
from typing import List
from game_gen_v2.common.configs import TransformerConfig, ResNetConfig, ConfigClass, FrameDataConfig

@dataclass
class ControlPredConfig(TransformerConfig):
    n_controls : int = 8 
    n_mouse_axes : int = 2

    # weight of losses for CE (button prediction) and mouse regression
    btn_weight : float = 1.0
    mouse_weight : float = 1.0
    prev_weight : float = 0.5 # Weight for predicting controls on previous frames

    button_zero_weights: List[float] = field(default_factory=lambda: [
        0.9960,
        0.3526,
        0.7644,
        0.8965,
        0.7849,
        0.9891,
        0.9993,
        0.9947,
        0.9870,
        0.9998,
        1.0,
        1.0,
        0.8509,
        0.8526
    ])
    mouse_zero_weights : float = 0.3183

ControlPredDataConfig = FrameDataConfig

@dataclass 
class ControlPredResConfig(ResNetConfig):
    n_controls : int = 8 
    n_mouse_axes : int = 2

    # For the transformer
    attn_n_layers : int = 2
    attn_n_heads : int = 6
    attn_d_model : int = 384
    attn_impl : str = "flash"

@dataclass
class ControlPredictorConfig(ConfigClass):
    channels_in : int = 3
    sample_size : int = 128
    temporal_sample_size : int = 16

    n_controls : int = 8 
    n_mouse_axes : int = 2

    # For the transformer
    attn_n_layers : int = 2
    attn_n_heads : int = 6
    attn_d_model : int = 384
    attn_impl : str = "flash"    