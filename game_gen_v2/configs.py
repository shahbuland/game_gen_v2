from typing import Dict, List, Optional
from dataclasses import dataclass, field
import yaml

@dataclass
class ConfigClass:
    def to_yaml(self, fp):
        """
        Serialize config to yaml file
        """
        yaml.dump(self.__dict__, fp)

    @classmethod
    def from_yaml(cls, fp):
        """
        Load config from yaml
        """
        data = yaml.safe_load(fp)
        return cls(**data)

@dataclass
class ModelConfig(ConfigClass):
    # Transformer
    n_layers : int = 12
    n_heads : int = 12
    d_model : int = 768
    flash : bool = True

    # input/latent
    image_size : int = 512
    sample_size : int = 32
    channels : int = 4
    patch_size : int = 4
    use_vae : bool = True

    # Guidance
    cfg_prob : float = 0.1

    # REPA
    repa_weight : float = 1.0
    repa_batch_size : int = 32
    repa_layuer_ind : int = 4
    repa_pool_factor : int = 1

@dataclass
class TrainConfig(ConfigClass):
    # Data
    target_batch_size : int = 256
    batch_size : int = 64
    epochs : int = 100

    # Optimizer
    opt : str = "AdamW"
    opt_kwargs : Dict = field(default_factory = lambda : {
        "lr": 1.0e-3,
        "eps": 1.0e-15,
        "betas" : (0.9, 0.96),
        "weight_decay" : 0.00,
    })

    # Scheduler
    scheduler: Optional[str] = None#"CosineDecay"
    scheduler_kwargs: Dict = field(default_factory=lambda: {
        "T_max" : 1000000,
        "eta_min" : 5.0e-6
    })

    # Intervals
    log_interval : int = 1
    sample_interval : int = 100
    save_interval : int = 2500
    val_interval : int = 1000
    
    # Saving/loading
    checkpoint_root_dir : str = "./checkpoints"
    resume : bool = False

    # Validating
    val_batch_mult : int = 4

    # General
    grad_clip : float = -1

@dataclass
class LoggingConfig(ConfigClass):
    run_name : str = "coco 150M (ngpt, lr=1e-3 + repa + norm_repa)"
    wandb_entity : str = "shahbuland"
    wandb_project : str = "mnist_sanity"

@dataclass
class SamplerConfig(ConfigClass):
    scheduler_cls : str = "FlowMatchEulerDiscreteScheduler"
    scheduler_kwargs : Dict = field(default_factory=lambda : {
        "shift" : 3
    })
    n_steps : int = 100
    cfg_scale : float = 1.5

@dataclass
class MergedConfig(ConfigClass):
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    logging: LoggingConfig = LoggingConfig()
    sampler: SamplerConfig = SamplerConfig()

    @classmethod
    def from_yaml(cls, fp):
        data = yaml.safe_load(fp)
        return cls(
            model=ModelConfig(**data.get('model', {})),
            train=TrainConfig(**data.get('train', {})),
            logging=LoggingConfig(**data.get('logging', {})),
            sampler=SamplerConfig(**data.get('sampler', {}))
        )

    def to_yaml(self, fp):
        yaml.dump({
            'model': self.model.__dict__,
            'train': self.train.__dict__,
            'logging': self.logging.__dict__,
            'sampler': self.sampler.__dict__
        }, fp)