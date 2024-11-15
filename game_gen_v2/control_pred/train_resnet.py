import torch

from game_gen_v2.common.trainers.supervised_trainer import SimpleSupervisedTrainer
from game_gen_v2.common.configs import TrainConfig
from game_gen_v2.common.utils import pretty_print_parameters
from game_gen_v2.data.loader import FrameDataset, create_loader

from .resnet import ControlPredResModel
from .configs import ControlPredResConfig, ControlPredDataConfig
from .sampling import ControlPredSampler
from .augs import transform_video, augtform_video

import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('spawn')

    train_cfg = TrainConfig.from_yaml("configs/control_pred/train_resnet.yml")
    model_cfg = ControlPredResConfig.from_yaml("configs/control_pred/resnet_config.yml")
    data_cfg = ControlPredDataConfig.from_yaml("configs/control_pred/data_raw.yml")

    model = ControlPredResModel(model_cfg)
    pretty_print_parameters(model)

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
        'num_workers': 2,
        'pin_memory': True,
        #'prefetch_factor': 4
    }

    loader = create_loader(dataset_kwargs, dataloader_kwargs)
    
    sampler = ControlPredSampler(
        fps=data_cfg.fps,
        out_res=data_cfg.sample_size,
        input_directory=data_cfg.val_path,
        image_transform=transform_video
    )

    trainer.train(model, loader, sampler = sampler)
    