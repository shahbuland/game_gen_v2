import torch

from game_gen_v2.common.trainers.supervised_trainer import SimpleSupervisedTrainer
from game_gen_v2.common.configs import TrainConfig
from game_gen_v2.common.utils import pretty_print_parameters
from game_gen_v2.data.loader import create_loader

from .configs import ControlPredictorConfig, ControlPredDataConfig
from .sampling import ControlPredMiddleSampler
from .augs import transform_video, augtform_video
from .cnn_3d import ControlPredictor

import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('spawn')

    train_cfg = TrainConfig.from_yaml("configs/control_pred/v2/train_v2.yml")
    model_cfg = ControlPredictorConfig.from_yaml("configs/control_pred/v2/cnn_v2.yml") 
    data_cfg = ControlPredDataConfig.from_yaml("configs/control_pred/v2/data_v2.yml")

    model = ControlPredictor(model_cfg)
    pretty_print_parameters(model)

    trainer = SimpleSupervisedTrainer(train_cfg, model_cfg)
    
    dataset_kwargs = {
        'data_dir': data_cfg.data_path,
        'frame_count': model_cfg.temporal_sample_size,
        'image_transform': augtform_video
    }

    dataloader_kwargs = {
        'batch_size': train_cfg.batch_size,
        'num_workers': 1,
        'pin_memory': True,
        #'prefetch_factor': 8
    }

    loader = create_loader(dataset_kwargs, dataloader_kwargs)
    
    sampler = ControlPredMiddleSampler(
        fps=data_cfg.fps,
        out_res=data_cfg.sample_size,
        input_directory=data_cfg.val_path,
        image_transform=transform_video,
        keybinds=data_cfg.keybinds
    )

    trainer.train(model, loader, sampler=sampler)
