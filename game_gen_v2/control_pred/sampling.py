from .nn import ControlPredModel
from game_gen_v2.common.nn.vae import VAE

import os
import numpy as np
import torch
import random
import einops as eo
import cv2
from pathlib import Path
from tqdm import tqdm
import wandb

FPS = 30
KEYBINDS = ["SPACE", "W", "A", "S", "D", "R", "E", "G", "F", "Q", "CONTROL", "SHIFT"]

def write_np_array_to_video(frames, fps=FPS, controls=None):
    """
    Convert a numpy array of frames to a wandb.Video object.

    :param frames: numpy array of shape [n_frames, height, width, channels]
    :param fps: int, frames per second for the output video
    :param controls: numpy array of control inputs
    :return: wandb.Video object
    """
    if not isinstance(frames, np.ndarray):
        raise ValueError("Input frames must be a numpy array")
    
    if len(frames.shape) != 4:
        raise ValueError("Input frames must have shape [n_frames, height, width, channels]")
    
    n_frames, height, width, channels = frames.shape
    
    processed_frames = []
    
    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if controls is not None:
            frame = draw_controls(frame, controls[i])

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        processed_frames.append(frame)
        
    # Convert the list of frames to a numpy array
    video_array = np.stack(processed_frames) # [n,h,w,c] [0,255] uint8
    video_array = eo.rearrange(video_array, 'n h w c -> n c h w')
    
    # Create a wandb.Video object
    video = wandb.Video(video_array, fps=fps, format="mp4")
    
    return video

def draw_controls(frame, control_vector):
    key_labels = KEYBINDS + ["LMB", "RMB"]
    key_pressed = [bool(value) for value in control_vector[:-2]]
    mouse_x_axis = float(control_vector[-2]) 
    mouse_y_axis = float(control_vector[-1])

    # Draw mouse arrow
    circle_center = (20, 20)
    cv2.circle(frame, circle_center, 38, (255, 255, 255), 2)  # Increased thickness from 1 to 2
    arrow_end = (
        int(circle_center[0] + mouse_x_axis * 25),
        int(circle_center[1] + mouse_y_axis * 25)
    )
    cv2.arrowedLine(frame, circle_center, arrow_end, (0, 255, 0), 3)  # Increased thickness from 2 to 3

    # Draw key boxes
    box_size = 12
    box_gap = 5
    start_x = 10
    start_y = 226  # 256 - 30 (bottom margin)

    for i, (label, pressed) in enumerate(zip(key_labels, key_pressed)):
        box_color = (0, 255, 0) if pressed else (0, 0, 255)
        box_x = start_x + i * (box_size + box_gap)
        cv2.rectangle(frame, (box_x, start_y), (box_x + box_size, start_y + box_size), box_color, -1)
        
        # Add label above the box
        label_y = start_y - 5
        cv2.putText(frame, label, (box_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return frame

def torch_to_np(x):
    # x is [b,c,h,w]
    x = torch.clamp(x, -1, 1)
    x = (x + 1) * 127.5
    x = x.to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()

class ControlPredSampler:
    """
    This sampler works in same way the sanity check works
    """
    def __init__(
            self,
            fps=30, out_res=256,
            input_directory="datasets/train_data",
            sample_cache = "sampler_cache.pt",
            n_samples=8, sample_length=300
        ):
        self.vae = VAE()
        self.fps = fps
        self.out_res = out_res
        self.input_directory = input_directory
        self.n_samples = n_samples
        self.sample_length = sample_length

        if os.path.exists(sample_cache):
            self.input_samples = torch.load(sample_cache).cuda()
        else:
            self.input_samples = self._get_input_samples().cuda()
            torch.save(self.input_samples, sample_cache)

    def _get_input_samples(self):
        video_files = sorted(Path(self.input_directory).glob("*_video.pt"))
        
        # Randomly select n_samples files
        selected_files = random.sample(video_files, min(self.n_samples, len(video_files)))
        
        input_samples = []
        for file in selected_files:
            video_tensor = torch.load(file)
            total_frames = video_tensor.shape[0]
            
            if total_frames <= self.sample_length:
                # If video is shorter than sample_length, use the whole video
                input_samples.append(video_tensor)
            else:
                # Randomly select a starting point
                start_idx = random.randint(0, total_frames - self.sample_length)
                sample = video_tensor[start_idx:start_idx + self.sample_length]
                input_samples.append(sample)
        
        return torch.stack(input_samples)

    @torch.no_grad()
    def predict_on_samples(self, model, model_cfg):
        """
        Get model predictions over the input samples
        Batch has each input video, we iterate over slices of the video
        """
        t = model_cfg.temporal_sample_size
        n_controls = model_cfg.n_controls

        # Get model dtype and device from model parameters
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device

        samples = self.input_samples # [b,n,c,h,w] n >> t 
        # Model takes [b,t,c,h,w] as its input, let's do a sliding window

        return_samples = samples[:,t-1:] # Skip first t-1, since we can't make prediction for them anyways
        return_controls = []

        for i in tqdm(range(t, samples.shape[1]+1)):
            model_in = samples[:,i-t:i] # [b,t,c,h,w]
            model_in = model_in.to(device=model_device,dtype=model_dtype)
            assert model_in.shape[1] == 8
            model_out_btn, model_out_mouse = model(model_in) # [b,n_controls]

            # Denormalize the inputs back to something useful
            model_out_btn = model_out_btn.sigmoid().round()
            model_out_mouse[:,0] *= 0.231
            model_out_mouse[:,1] *= 0.0487
            model_out = torch.cat([model_out_btn, model_out_mouse], -1)
            
            return_controls.append(model_out)
        
        return_controls = torch.stack(return_controls, dim = 1) # [b,n_frames,n_controls]

        return return_samples, return_controls

    def __call__(self, model: ControlPredModel):
        """
        Given model, uses model to predict controls for internal samples.
        Returns decoded frames with model predictions pasted over them in same format as sanity stuff
        """
        model = model.core
        config = model.config
        n_controls = config.n_controls

        samples, controls = self.predict_on_samples(model, config)
        
        samples = torch.stack([self.vae.decode(sample) for sample in samples])

        res = []
        for i, (sample, control) in enumerate(zip(samples, controls)):
            res.append(write_np_array_to_video(torch_to_np(sample), controls=control))

        return res


if __name__ == "__main__":
    import torch
    from game_gen_v2.control_pred.nn import ControlPredModel
    from game_gen_v2.control_pred.configs import ControlPredConfig

    # Load the model configuration
    model_cfg = ControlPredConfig.from_yaml("configs/control_pred/tiny.yml")

    # Initialize the model with random weights
    model = ControlPredModel(model_cfg)
    model.cuda()
    model.half()

    # Initialize the sampler
    sampler = ControlPredSampler(
        input_directory="datasets/train_data",
        n_samples=2,
        sample_length=64
    )

    sampler(model)





