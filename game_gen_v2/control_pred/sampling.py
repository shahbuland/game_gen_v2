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

from game_gen_v2.data.pipelines.data_config import FPS_OUT

FPS = FPS_OUT

def write_np_array_to_video(frames, fps=FPS, controls=None, output_fmt = "wandb", keybinds=None):
    """
    Convert a numpy array of frames to a wandb.Video object.

    :param frames: numpy array of shape [n_frames, height, width, channels]
    :param fps: int, frames per second for the output video
    :param controls: numpy array of control inputs
    :param output_fmt: Returns:
        - "wandb" : WANDB video [RGB, NCHW]
        - "cv" : CV video [BGR, NHWC]
        - "np" : np video [RGB, NHWC]
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
        frame = cv2.resize(frame, (256, 256))

        if controls is not None:
            frame = draw_controls(frame, controls[i], keybinds)

        if output_fmt != "cv":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        processed_frames.append(frame)
        
    # Convert the list of frames to a numpy array
    video_array = np.stack(processed_frames)  # [n,h,w,c] [0,255] uint8

    if output_fmt == "wandb":
        video_array = eo.rearrange(video_array, 'n h w c -> n c h w')
        video = wandb.Video(video_array, fps=fps, format="mp4")
    elif output_fmt == "cv":
        video = video_array  # CV2 format is already [n,h,w,c] in BGR
    elif output_fmt == "np":
        video = video_array  # Already in [n,h,w,c] RGB format
    else:
        raise ValueError(f"Invalid output format: {output_fmt}")

    return video

def draw_controls(frame, control_vector, keybinds):

    key_labels = keybinds + ["LMB", "RMB"]
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
    box_gap = 20
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
            n_samples=8, sample_length=300,
            image_transform=None
        ):
        self.fps = fps
        self.out_res = out_res
        self.input_directory = input_directory
        self.n_samples = n_samples
        self.sample_length = sample_length
        self.image_transform = image_transform

        if os.path.exists(sample_cache):
            self.input_samples = torch.load(sample_cache).cuda()
        else:
            self.input_samples = self._get_input_samples().cuda()
            torch.save(self.input_samples, sample_cache)

        if self.input_samples.shape[-3] == 3:
            self.vae = None
        else:
            self.vae = VAE()
            

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
        
        res = torch.stack(input_samples)
        if self.image_transform is None:
            return res
        return self.image_transform(res)
    
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
            model_out_btn, model_out_mouse = model(model_in) # [b,n_controls]
            model_out_btn = model_out_btn[:,-1]
            model_out_mouse = model_out_mouse[:,-1]

            # Denormalize the inputs back to something useful
            model_out_btn = (torch.sigmoid(model_out_btn) > 0.5).float()
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
        
        if self.vae is not None:
            samples = torch.stack([self.vae.decode(sample) for sample in samples])

        res = []
        for i, (sample, control) in enumerate(zip(samples, controls)):
            res.append(write_np_array_to_video(torch_to_np(sample), controls=control, fps=self.fps))

        return res
    
class ControlPredMiddleSampler(ControlPredSampler):
    def __init__(self, fps=60, out_res=128, input_directory=None, image_transform=None, keybinds=None):
        super().__init__(fps, out_res, input_directory, image_transform=image_transform)

        self.keybinds = keybinds

    @torch.no_grad()
    def predict_on_samples(self, model, model_cfg):
        """
        Predicts controls for each frame using a sliding window approach.
        For each window of temporal_sample_size frames, predicts controls for the middle frame.
        """
        t = model_cfg.temporal_sample_size
        n_controls = model_cfg.n_controls
        mid_idx = t // 2

        # Get model dtype and device from model parameters
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device

        samples = self.input_samples # [b,n,c,h,w] n >> t
        # Model takes [b,t,c,h,w] as input

        # We'll return all frames but can only predict for frames after mid_idx
        return_samples = samples.clone()
        return_controls = []

        # Initialize empty predictions for first mid_idx frames
        empty_preds = torch.zeros((samples.shape[0], mid_idx, n_controls+2), 
                                device=model_device, dtype=model_dtype)
        return_controls.append(empty_preds)

        # Predict for middle frames using sliding window
        for i in tqdm(range(t, samples.shape[1] - mid_idx + 1)):
            model_in = samples[:,i-t:i] # [b,t,c,h,w]
            model_in = model_in.to(device=model_device, dtype=model_dtype)
            model_out_btn, model_out_mouse = model(model_in) # [b,d] # model out for middle
            
            # Denormalize the inputs back to something useful
            model_out_btn = (torch.sigmoid(model_out_btn) > 0.5).float()
            model_out = torch.cat([model_out_btn, model_out_mouse], -1) # [b,d]
            
            return_controls.append(model_out.unsqueeze(1))

        # Stack all predictions
        return_controls = torch.cat(return_controls, dim=1) # [b,n_frames,n_controls]
        # Add empty predictions for last few frames to match length
        n_remaining = return_samples.shape[1] - return_controls.shape[1]
        if n_remaining > 0:
            empty_preds = torch.zeros((samples.shape[0], n_remaining, n_controls+2), 
                                    device=model_device, dtype=model_dtype)
            return_controls = torch.cat([return_controls, empty_preds], dim=1)

        return return_samples, return_controls

    @torch.no_grad()
    def __call__(self, model: ControlPredModel):
        """
        Given model, uses model to predict controls for internal samples.
        Returns frames with model predictions in same format as sanity stuff
        """
        model = model.core
        config = model.config
        n_controls = config.n_controls

        samples, controls = self.predict_on_samples(model, config)

        res = []
        for i, (sample, control) in enumerate(zip(samples, controls)):
            res.append(write_np_array_to_video(torch_to_np(sample), controls=control, fps=self.fps, keybinds=self.keybinds))

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





