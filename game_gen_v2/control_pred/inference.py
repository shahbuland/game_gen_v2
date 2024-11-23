# We assume specific model classes
from .cnn_3d import ControlPredictor
from .configs import ControlPredictorConfig, ControlPredDataConfig
from .sampling import write_np_array_to_video, draw_controls

from game_gen_v2.data.videos import TAStreamVideoReader

import cv2
import os
import torch
from safetensors.torch import load_file
from dataclasses import dataclass
from tqdm import tqdm
import torch.nn.functional as F
from skvideo.io import vwrite
import einops as eo
import numpy as np

def load_model_ckpt_in_dir(path, model_cfg):
    if not path.endswith(".safetensors") and os.path.isdir(path):
        # Look for any .safetensors files in the directory
        for file in os.listdir(path):
            if file.endswith(".safetensors"):
                path = os.path.join(path, file)
                break
        else:
            path = None
    if path is None:
        raise ValueError("No checkpoint found.")
    
    ckpt = load_file(path)
    model = ControlPredictor(model_cfg)
    model.load_state_dict(ckpt)
    model = model.core
    model.eval()
    model.cuda()
    model.half()

    return model

def print_control_stats(control_vectors, keybinds):
    # Convert to numpy array if not already
    control_vectors = np.array(control_vectors)
    
    # Mouse stats
    mouse_x = control_vectors[:,-2]
    mouse_y = control_vectors[:,-1]
    
    print("\nMouse Statistics:")
    print(f"Mouse X - Min: {mouse_x.min():.3f}, Max: {mouse_x.max():.3f}, Mean: {mouse_x.mean():.3f}")
    print(f"Mouse Y - Min: {mouse_y.min():.3f}, Max: {mouse_y.max():.3f}, Mean: {mouse_y.mean():.3f}")
    
    print("\nButton Press Frequencies:")
    # For each keybind and mouse button
    all_buttons = keybinds + ["LMB", "RMB"]
    for i, button in enumerate(all_buttons):
        active_ratio = control_vectors[:,i].mean()  # Mean of 0s and 1s gives proportion of 1s
        print(f"{button:>4}: {active_ratio*100:.1f}% of frames")

def draw_controls(frame, control_vector, keybinds):
    h,w = frame.shape[:2]  # Get height and width, accounting for channels
    
    key_labels = keybinds + ["LMB", "RMB"]
    key_pressed = [bool(value) for value in control_vector[:-2]]
    mouse_x_axis = float(control_vector[-2]) 
    mouse_y_axis = float(control_vector[-1])

    # Draw mouse arrow - circle diameter should be ~1/4 screen width
    circle_radius = int(w * 0.125)  # Radius is half diameter
    circle_center = (circle_radius + 20, circle_radius + 20)  # Offset from corner
    cv2.circle(frame, circle_center, circle_radius, (255, 255, 255), max(2, int(w/500)))
    
    arrow_length = int(circle_radius * 0.8)  # Arrow length proportional to circle
    arrow_end = (
        int(circle_center[0] + mouse_x_axis * arrow_length),
        int(circle_center[1] + mouse_y_axis * arrow_length)
    )
    cv2.arrowedLine(frame, circle_center, arrow_end, (0, 255, 0), max(3, int(w/300)))

    # Draw key boxes spread across width
    n_boxes = len(key_labels)
    box_size = int(w * 0.03)  # Box size proportional to width
    total_width = w - 40  # Leave margins
    box_gap = (total_width - (n_boxes * box_size)) // (n_boxes - 1)
    
    start_x = 20  # Left margin
    start_y = h - int(h * 0.1)  # 10% from bottom

    for i, (label, pressed) in enumerate(zip(key_labels, key_pressed)):
        box_color = (0, 255, 0) if pressed else (0, 0, 255)
        box_x = start_x + i * (box_size + box_gap)
        cv2.rectangle(frame, (box_x, start_y), (box_x + box_size, start_y + box_size), box_color, -1)
        
        # Add label above the box with size proportional to frame
        label_y = start_y - 5
        font_scale = max(0.3, w/1000)  # Scale font with frame width
        cv2.putText(frame, label, (box_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, int(w/500)))

    return frame

def draw_video(frames, controls, keybinds):
    """
    Going into this function:

    :param frames: (n,h,w,c) uint8 numpy arr
    :param controls: (n,keysbinds+2+2) float arr where keybinds+2 (keybinds+lmb+rmb) is 0 or 1 and last 2 is mouse axes
    :param keybinds: List of keybinds as strings, LMB and RMB are assumed at end
    """
    # To draw we will use opencv, this makes some things easier. For this, convert all frames to BGR, draw the control, continue
    print("Drawing controls...")
    processed_frames = []
    for i, frame in tqdm(enumerate(frames)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = draw_controls(frame, controls[i], keybinds)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(frame)
    
    return np.stack(processed_frames)

class InferenceEngine:
    """
    Do inference with control prediction model
    """
    def __init__(self, model_cfg, data_cfg, model_ckpt_path):
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg

        self.fps = self.data_cfg.fps

        #self.video_res = (1080, 1920)
        self.video_res = (512, 512)

        self.sample_size = self.model_cfg.sample_size
        self.window_size = self.model_cfg.temporal_sample_size
        self.keybinds = self.data_cfg.keybinds
        
        self.model = load_model_ckpt_in_dir(model_ckpt_path, self.model_cfg)
        self.chunk_size = 64 # For reading, 64 is fine

        self.threshold = 0.5 # For button presses
        self.renormalize_mouse = False
        self.rescale_mouse = 1

        self.video_cache = None
        self.use_cache = False
    
    def change_model(self, new_path):
        self.model = load_model_ckpt_in_dir(new_path, self.model_cfg)

    @torch.no_grad()
    def predict_controls(self, frames):
        middle_idx = self.window_size//2

        n_frames, _, _, _ = frames.shape
        frame_inputs = F.interpolate(frames, (self.sample_size, self.sample_size))
        control_preds = torch.zeros(n_frames, len(self.data_cfg.keybinds) + 4)

        window_start = 0
        window_end = self.window_size

        print("Generating controls...")
        for _ in tqdm(range(n_frames)):
            out_idx = window_start + middle_idx

            frame_window = frame_inputs[window_start:window_end].clone().unsqueeze(0).to(device='cuda',dtype=torch.half)
            
            model_out = torch.cat(self.model(frame_window),-1) # Model separates buttons and mouse by default
            control_preds[out_idx] = model_out.squeeze(0)

            window_start += 1
            window_end += 1

            if window_end >= n_frames:
                break

        # Renormalize the scores to make them more interpretable
        # Split into buttons and mouse
        btn_preds = control_preds[:,:-2]  # All but last 2 columns are buttons
        mouse_x = control_preds[:,-2]     # Second to last column is mouse x
        mouse_y = control_preds[:,-1]     # Last column is mouse y

        # Normalize buttons with sigmoid and threshold
        btn_preds = (torch.sigmoid(btn_preds) > self.threshold).float()

        if self.renormalize_mouse:
            # Normalize mouse x based on max absolute value
            max_abs_x = torch.max(torch.abs(mouse_x))
            if max_abs_x > 0:  # Avoid division by zero
                mouse_x = mouse_x / max_abs_x

            # Normalize mouse y based on max absolute value 
            max_abs_y = torch.max(torch.abs(mouse_y))
            if max_abs_y > 0:  # Avoid division by zero
                mouse_y = mouse_y / max_abs_y
        
        mouse_y = mouse_y * self.rescale_mouse
        mouse_x = mouse_x * self.rescale_mouse

        # Recombine into control_preds
        control_preds = torch.cat([
            btn_preds,
            mouse_x.unsqueeze(-1),
            mouse_y.unsqueeze(-1)
        ], dim=-1)

        return control_preds


    @torch.no_grad()
    def predict_on_video(self, video_path, out_path=None):
        """
        Assumes video has previously been processed to 60 FPS
        """
        if out_path is None:
            out_path = video_path.rstrip(".mp4")+"_labelled.mp4"

        if self.video_cache is None or not self.use_cache:
            vid_reader = TAStreamVideoReader(
                1,
                self.chunk_size,
                self.video_res[0],
                self.video_res[1]
            )
            vid_reader.reset(video_path)
            frames = []
            
            print("Reading video...")
            for _ in tqdm(range(len(vid_reader))):
                frames.append(vid_reader.read(self.chunk_size))

            frames = torch.cat(frames,0) # [-1,1] [N,C,H,W]
            if self.use_cache:
                self.video_cache = frames
        else:
            frames = self.video_cache

        controls = self.predict_controls(frames).detach().cpu().numpy()
        
        #print_control_stats(controls, self.keybinds)
        #exit()

        # Process frames for CV to work with 
        frames = eo.rearrange(frames, 'n c h w -> n h w c')
        frames = (frames + 1) * 127.5
        frames = frames.detach().cpu().byte().numpy()

        final_frames = draw_video(frames, controls, self.keybinds)
        # Save video using skvideo
        vwrite(
            out_path, 
            final_frames,
            inputdict={'-r': str(self.fps)},
            outputdict={
                '-vcodec': 'libx264',
                '-r': str(self.fps),
                '-pix_fmt': 'yuv420p'
            }
        )


if __name__ == "__main__":
    model_cfg = ControlPredictorConfig.from_yaml("configs/control_pred/v2/cnn_v2.yml")
    data_cfg = ControlPredDataConfig.from_yaml("configs/control_pred/v2/data_v2.yml")
    cp_path = "checkpoints/control_pred/3d_cnn_47k_96sens"

    pipe = InferenceEngine(model_cfg, data_cfg, cp_path)
    pipe.predict_on_video("experiments/control_pred_irl/simple.mp4")





