import torch

from game_gen_v2.common.trainers.supervised_trainer import SimpleSupervisedTrainer
from game_gen_v2.common.configs import TrainConfig
from game_gen_v2.data.loader import FrameDataset, create_loader

from .nn import ControlPredModel
from .configs import ControlPredConfig, ControlPredDataConfig
from .sampling import ControlPredSampler
from .augs import transform_video, augtform_video

if __name__ == "__main__":
    train_cfg = TrainConfig.from_yaml("configs/control_pred/train_raw.yml")
    model_cfg = ControlPredConfig.from_yaml("configs/control_pred/tiny_rgb.yml")
    data_cfg = ControlPredDataConfig.from_yaml("configs/control_pred/data_raw.yml")

    model = ControlPredModel(model_cfg)
    trainer = SimpleSupervisedTrainer(train_cfg, model_cfg)

    dataset_kwargs = {
        'data_dir': data_cfg.data_path,
        'frame_count': model_cfg.temporal_sample_size,
        'diversity': data_cfg.diversity,
        'top_p': data_cfg.top_p,
        'image_transform': augtform_video
    }

    dataloader_kwargs = {
        'batch_size': train_cfg.batch_size,
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 2
    }

    loader = create_loader(dataset_kwargs, dataloader_kwargs)
    
    sampler = ControlPredSampler(
        fps=data_cfg.fps,
        out_res=data_cfg.sample_size,
        input_directory=data_cfg.val_path,
        image_transform=transform_video
    )

    trainer.train(model, loader, sampler = sampler)
    