"""
Augments to make game footage look more like IRL footage,
so that the control prediction model can generalize for other games + IRL footage
"""

import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
import random

@torch.no_grad()
def transform_video(vid):
    return (vid.float()/127.5 - 1)

@torch.no_grad()
def augtform_video(vid):
    return augment_video(vid.float()/255)
 
@torch.no_grad()
def augment_video(vid):
    # vid is [b,n,c,h,w]
    b, n, c, h, w = vid.shape
    
    # List of augmentations

    # Augments that can be easily batched
    augmentations = [
        color_jitter,
        hue_rotate,
        add_noise,
        random_recrop
    ]
    
    # Randomly sample 0 to 4 augments and apply
    num_augments = random.randint(0, 4)
    selected_augments = random.sample(augmentations, num_augments)
    
    for augment in selected_augments:
        vid = augment(vid)

    vid = (vid*2 - 1)
    return vid

def color_jitter(vid, b_delta = 0.35, c_delta = 0.35, s_delta = 0.3):
    # vid is [b,n,c,h,w]
    b, n, c, h, w = vid.shape

    def uniform(size, min, max):
        return torch.rand(size,device=vid.device,dtype=vid.dtype) * (max - min) + min
    
    brightness = uniform(
        (b,1,1,1,1),
        max(0, 1 - b_delta), 
        1 + b_delta
    )

    contrast = uniform(
        (b,1,1,1,1),
        max(0, 1 - c_delta),
        1 + c_delta
    )

    saturation = uniform(
        (b,1,1,1,1),
        max(0, 1 - s_delta),
        1 + s_delta
    )



    vid = (vid * brightness).clamp(0,1)
    mean = vid.mean(dim=[-1,-2], keepdim=True) # [b,n,c,1,1]
    vid = (mean + (vid - mean) * contrast).clamp(0,1)

    # Saturation
    weights = torch.tensor([0.299, 0.587, 0.114],device=vid.device,dtype=vid.dtype).view(1,1,3,1,1)
    gray = (vid * weights).sum(dim=2, keepdim=True)
    gray = gray.expand_as(vid)
    vid = (gray + (vid - gray) * saturation).clamp(0,1)

    return vid

def hue_rotate(vid, h_delta=0.15):
    b, n, c, h, w = vid.shape

    def uniform(size, min, max):
        return torch.rand(size,device=vid.device,dtype=vid.dtype) * (max - min) + min
    
    hue_factors = uniform((b,), -h_delta, h_delta)
    for i in range(b):
        vid[i] = TF.adjust_hue(vid[i], float(hue_factors[i]))
    
    return vid

def add_noise(vid, std=0.01):
    # vid is [b,n,c,h,w]
    noise = torch.randn_like(vid) * std
    return torch.clamp(vid + noise, 0,1)

def random_recrop(vid, scale_range=(0.8,1.0)):
    # vid is [b,n,c,h,w]
    b, n, c, h, w = vid.shape
    
    min_scale, max_scale = scale_range
    scales = torch.rand(b, device=vid.device,dtype=vid.dtype) * (max_scale - min_scale) + min_scale

    def recrop(i):
        # [n,c,h,w]
        ret, scale = vid[i], scales[i].item()
        
        scale_h = round((h - h * scale)/2)
        scale_w = round((w - w *scale)/2)


        ret = vid[i][:,:,scale_h:h-scale_h,scale_w:w-scale_w]
        return F.interpolate(ret, (h, w), mode='bilinear', align_corners=True)
    
    for i in range(b):
        vid[i] = recrop(i)

    return vid

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from game_gen_v2.data.loader import FrameDataset
    from game_gen_v2.common.configs import TrainConfig
    from game_gen_v2.control_pred.configs import ControlPredConfig, ControlPredDataConfig

    from tinygrad.helpers import Timing
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

    # Load configurations
    model_cfg = ControlPredConfig.from_yaml("configs/control_pred/tiny_rgb.yml")
    data_cfg = ControlPredDataConfig.from_yaml("configs/control_pred/data_raw.yml")

    # Create dataset and dataloader
    ds = FrameDataset(
        data_cfg.data_path,
        frame_count=model_cfg.temporal_sample_size,
        top_p=data_cfg.top_p,
        image_transform=lambda x: augment_video(transform_video(x))
    )

    loader = ds.create_loader(batch_size=8, num_workers=0)

    # Get a batch of data
    with Timing("Time to get frame: "):
        frames, _ = next(iter(loader))
    frames = frames.detach().cpu()
    # Stack 4 random frames from each of the 8 videos into a grid
    from einops import rearrange
    from torchvision.utils import save_image

    grid = []
    for i in range(8):  # For each video
        video = frames[i]
        frame_indices = torch.randint(0, video.shape[0], (4,))  # Select 4 random frames
        
        video_frames = []
        for idx in frame_indices:
            frame = video[idx]
            # Convert from [-1, 1] to [0, 1] range
            frame = (frame + 1) / 2
            video_frames.append(frame)
        
        video_column = torch.stack(video_frames)
        grid.append(video_column)

    # Stack all video columns horizontally
    grid = torch.stack(grid, dim=1)
    
    # Rearrange the grid to have shape [3, H*8, W*4]
    grid = rearrange(grid, 'f b c h w -> c (b h) (f w)')

    # Save the image without lowering resolution
    save_image(grid, 'augmented_frames_grid.png')
