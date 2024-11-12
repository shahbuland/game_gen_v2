import torch
import safetensors
from safetensors.torch import load_file
from tqdm import tqdm
import os
from skvideo.io import vwrite

from .nn import ControlPredModel
from .configs import ControlPredConfig
from .sampling import ControlPredSampler, write_np_array_to_video, draw_controls, torch_to_np

from game_gen_v2.common.nn.vae import VAE
from game_gen_v2.data.videos import TAStreamVideoReader

cp_path = "checkpoints/control_pred/many_tiny_16k/model.safetensors"

# Load the model configuration
model_cfg = ControlPredConfig.from_yaml("configs/control_pred/tiny_many.yml")

# Initialize the model
model = ControlPredModel(model_cfg)

checkpoint = load_file(cp_path)
model.load_state_dict(checkpoint)
model = model.core
model.eval()
model.cuda()
model.half()
vae = VAE()
vae.cuda()
vae.half()

@torch.no_grad()
def predict_on_samples(samples, ema_alpha_btn=0.0, ema_alpha_mouse=0.0):
    """
    Get model predictions over the input samples with EMA smoothing
    """
    t = model_cfg.temporal_sample_size
    n_controls = model_cfg.n_controls

    # Get model dtype and device from model parameters
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    samples = samples.unsqueeze(0)

    # Start in middle
    middle_idx = t // 2
    return_samples = samples[:,middle_idx:] # Skip first t-1, since we can't make prediction for them anyways
    return_controls = []

    # Initialize EMA states
    ema_btn_logits = None
    ema_mouse_controls = None

    for i in tqdm(range(t, samples.shape[1]+1)):
        model_in = samples[:,i-t:i].to(device=model_device, dtype=model_dtype)
        model_out_btn, model_out_mouse = model(model_in)
        model_out_btn = model_out_btn[:,middle_idx]
        model_out_mouse = model_out_mouse[:,middle_idx]

        # Apply EMA smoothing to button logits
        if ema_btn_logits is None:
            ema_btn_logits = model_out_btn
        else:
            ema_btn_logits = ema_alpha_btn * ema_btn_logits + (1 - ema_alpha_btn) * model_out_btn

        # Apply EMA smoothing to mouse controls
        if ema_mouse_controls is None:
            ema_mouse_controls = model_out_mouse
        else:
            ema_mouse_controls = ema_alpha_mouse * ema_mouse_controls + (1 - ema_alpha_mouse) * model_out_mouse

        # Use smoothed logits for button predictions
        smoothed_btn = (torch.sigmoid(ema_btn_logits) > 0.9).float()
        
        # Combine smoothed button and mouse controls
        model_out = torch.cat([smoothed_btn, 2*ema_mouse_controls], -1)
        
        return_controls.append(model_out)
    
    return_controls = torch.stack(return_controls, dim=1) # [b,n_frames,n_controls]

    return_samples = return_samples[:,:return_controls.shape[1]]

    return return_samples, return_controls

def label_video(video_path):
    samples = []
    fps_in = 60
    fps_out = 30
    batch_size = 64
    latent_path = "experiments/sample_latents.pt"
    if os.path.exists(latent_path):
        samples = torch.load(latent_path)
        print(f"Loaded existing latents from {latent_path}")
    else:
        reader = TAStreamVideoReader(
            fps_in//fps_out,
            batch_size,
            256,
            256
        )
        reader.reset(video_path)
        for _ in tqdm(range(len(reader))):
            samples.append(vae.encode(reader.read(batch_size)))
        
        samples = torch.cat(samples)
        torch.save(samples, latent_path)
        print(f"Saved latents to {latent_path}")

    samples, controls = predict_on_samples(samples)
    x = torch_to_np(vae.decode(samples[0]))
    

    c = controls[0]


    frames = write_np_array_to_video(x, fps=30,controls=c,output_fmt="np")


    output_path = "experiments/sample_labelled.mp4"
    vwrite(output_path, frames, outputdict={"-r": "30"})



label_video("experiments/control_pred_irl/erratic.mp4")
print("Done")

