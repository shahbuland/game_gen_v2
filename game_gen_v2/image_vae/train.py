import torch

from game_gen_v2.common.trainers.selfsupervised_trainer import SimpleSelfSupervisedTrainer
from game_gen_v2.common.configs import TrainConfig, FrameDataConfig
from game_gen_v2.data.loader import FrameDataset
from game_gen_v2.common.utils import pretty_print_parameters

from .nn import TransformerVAE
from .configs import TransformerVAEConfig
from .sampling import VAESampler
from ..control_pred.augs import transform_video, augtform_video

if __name__ == "__main__":
    train_cfg = TrainConfig.from_yaml("configs/image_vae/tiny_deep_train.yml")
    model_cfg = TransformerVAEConfig.from_yaml("configs/image_vae/tiny_deep.yml")
    data_cfg = FrameDataConfig.from_yaml("configs/image_vae/tiny_deep_data.yml")

    model = TransformerVAE(model_cfg)
    pretty_print_parameters(model)
    trainer = SimpleSelfSupervisedTrainer(train_cfg, model_cfg)
    ds = FrameDataset(
        data_cfg.data_path,
        frame_count=1,
        top_p = data_cfg.top_p,
        diversity = data_cfg.diversity,
        image_transform = augtform_video,
        ignore_controls = True
    )

    loader = ds.create_loader(train_cfg.batch_size, num_workers = 0)
    sampler = VAESampler(
        fps=data_cfg.fps,
        input_directory=data_cfg.val_path,
        image_transform=transform_video,
        sample_cache="vae_sampler_cache.pt"
    )

    trainer.train(model, loader, sampler = sampler)
