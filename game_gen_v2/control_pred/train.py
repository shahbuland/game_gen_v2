import torch

from game_gen_v2.common.trainers.supervised_trainer import SimpleSupervisedTrainer
from game_gen_v2.common.configs import TrainConfig
from game_gen_v2.data.loader import FrameDataset

from .nn import ControlPredModel
from .configs import ControlPredConfig
from .sampling import ControlPredSampler

if __name__ == "__main__":
    train_cfg = TrainConfig.from_yaml("configs/control_pred/train_1.yml")
    model_cfg = ControlPredConfig.from_yaml("configs/control_pred/tiny_many.yml")

    model = ControlPredModel(model_cfg)
    trainer = SimpleSupervisedTrainer(train_cfg, model_cfg)
    ds = FrameDataset(
        train_cfg.dataset_id,
        frame_count=model_cfg.temporal_sample_size
    )
    print("dataset generated")
    #sampler = ControlPredSampler()

    loader = ds.create_loader(train_cfg.batch_size, num_workers = 0)
    sampler = ControlPredSampler()

    trainer.train(model, loader, sampler = sampler)
    