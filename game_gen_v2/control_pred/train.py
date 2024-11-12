import torch

from game_gen_v2.common.trainers.supervised_trainer import SimpleSupervisedTrainer
from game_gen_v2.common.configs import TrainConfig
from game_gen_v2.data.loader import FrameDataset

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
    ds = FrameDataset(
        data_cfg.data_path,
        frame_count=model_cfg.temporal_sample_size,
        top_p = data_cfg.top_p,
        image_transform = augtform_video
    )

    loader = ds.create_loader(train_cfg.batch_size, num_workers = 0)
    sampler = None
    sampler = ControlPredSampler(
        fps=data_cfg.fps,
        out_res=data_cfg.sample_size,
        input_directory=data_cfg.data_path,
        image_transform=transform_video
    )

    trainer.train(model, loader, sampler = sampler)
    