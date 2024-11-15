from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Set
import yaml

@dataclass
class ConfigClass:
    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'ConfigClass':
        with open(yaml_file, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

    def save_yaml(self, yaml_file: str) -> None:
        with open(yaml_file, 'w') as file:
            yaml.dump(self.to_dict(), file, default_flow_style=False)

@dataclass
class TransformerConfig(ConfigClass):
    # Config for any transformer based model
    n_layers : int = 12
    n_heads : int = 12
    d_model : int = 768
    attn_impl : str = "linear" # Linear or flash
    cross : bool = False # Cross attention?

    # Only relevant for vision or video transformers
    channels : int = 4
    sample_size : int = 32
    patch_size : int = 2
    
    # Video
    temporal_patch_size : int = 10
    temporal_sample_size : int = 100

    @property
    def n_image_tokens(self) -> int:
        return (self.sample_size // self.patch_size) ** 2

    @property
    def n_video_tokens(self) -> int:
        return self.n_image_tokens * (self.temporal_sample_size // self.temporal_patch_size)
    
@dataclass
class ResNetConfig(ConfigClass):
    channel_counts : List[int] = field(default_factory=lambda :[
        64,
        128,
        256,
        512,
        512
    ])
    res_blocks : List[int] = field(default_factory=lambda :[
        2,
        2,
        2,
        2
    ])
    channels_in : int = 3
    sample_size : int = 224
    temporal_sample_size : int = 16

    @property
    def n_layers(self) -> int:
        return len(self.channel_counts) - 1
    
@dataclass
class TrainConfig(ConfigClass):
    # Generic train terms
    dataset_id : str = "coco"
    target_batch_size : int = 256
    batch_size : int = 64
    val_batch_mult : int = 2 # Bigger batch size for validation?
    grad_clip : float = -1 # set > 0 for gradient norm clipping
    epochs : int = 100

    # EMA
    use_ema : bool = False
    ema_beta : float = 0.999
    ema_every : int = 1
    ema_start_offset : int = 1
    ema_ignore : Set = field(default_factory=lambda: { # module names to ignore with ema

    })

    # Saving and intervals
    checkpoint_root_dir : str = "checkpoints"
    log_interval : int = 1
    sample_interval : int = 100 # Not all model sample
    save_interval : int = 20000
    val_interval : int = 1000
    resume : bool = False

    # optimizer and scheduler
    opt : str = "AdamW"
    opt_kwargs : Dict = field(default_factory = lambda : {
        "lr": 1.0e-4,
        "eps": 1.0e-15,
        "betas" : (0.9, 0.95),
        "weight_decay" : 0.1,
        #"precondition_frequency" : 2
    })
    scheduler: Optional[str] = None#"Warmup"
    scheduler_kwargs: Dict = field(default_factory=lambda: {
        #"T_max" : 15000,
        #"eta_min" : 1.0e-4
        "warmup_steps" : 150
    })


    # Logging
    run_name : str = "first attempt"
    wandb_entity : str = "shahbuland"
    wandb_project : str = "control_pred"

@dataclass
class FrameDataConfig(ConfigClass):
    data_path: str = ""
    val_path: str = ""
    sample_size : int = 256
    temporal_sample_size :int = 8
    channels : int = 3
    diversity : bool = True
    top_p : float = 0.5
    fps: int = 15

@dataclass
class SamplerConfig(ConfigClass):
    n_steps : int = 128
    cfg_scale : float = 3.5
    fast_steps : int = 1 # When sampling for distillation, what step size to use?
