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

cp_path = "checkpoints/control_pred/WASD_tiny_10k/model.safetensors"

# Load the model configuration
model_cfg = ControlPredConfig.from_yaml("configs/control_pred/tiny_wasd.yml")

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
def predict_on_samples(samples):
    """
    Get model predictions over the input samples
    Batch has each input video, we iterate over slices of the video
    """
    t = model_cfg.temporal_sample_size
    n_controls = model_cfg.n_controls

    # Get model dtype and device from model parameters
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    # samples =   [b,n,c,h,w] n >> t 
    # Model takes [b,t,c,h,w] as its input, let's do a sliding window

    samples = samples.unsqueeze(0)

    # Start in middle
    middle_idx = t // 2
    return_samples = samples[:,middle_idx:] # Skip first t-1, since we can't make prediction for them anyways
    return_controls = []

    for i in tqdm(range(t, samples.shape[1]+1)):
        model_in = samples[:,i-t:i] # [b,t,c,h,w]
        model_in = model_in.to(device=model_device,dtype=model_dtype)
        model_out_btn, model_out_mouse = model(model_in) # [b,n_controls]
        model_out_btn = model_out_btn[:,middle_idx]
        model_out_mouse = model_out_mouse[:,middle_idx]

        # Denormalize the inputs back to something useful
        model_out_btn = (torch.sigmoid(model_out_btn) > 0.5).float()
        model_out = torch.cat([model_out_btn, model_out_mouse], -1)
        
        return_controls.append(model_out)
    
    return_controls = torch.stack(return_controls, dim = 1) # [b,n_frames,n_controls]

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

    # Make video twice as fast
    samples = samples[::2]

    samples, controls = predict_on_samples(samples)
    x = torch_to_np(vae.decode(samples[0]))
    

    c = controls[0]


    frames = write_np_array_to_video(x, fps=30,controls=c,output_fmt="np")


    output_path = "experiments/sample_labelled.mp4"
    vwrite(output_path, frames, outputdict={"-r": "30"})



label_video("experiments/apartment.mp4")
print("Done")

