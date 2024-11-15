import torch
import safetensors
from safetensors.torch import load_file
from tqdm import tqdm
import os
from skvideo.io import vwrite

from .resnet import ControlPredResModel
from .configs import ControlPredResConfig
from .sampling import ControlPredSampler, write_np_array_to_video, draw_controls, torch_to_np

from game_gen_v2.common.nn.vae import VAE
from game_gen_v2.data.videos import TAStreamVideoReader

cp_path = "checkpoints/control_pred/3d_resnet_4k/model.safetensors"

# Load the model configuration
model_cfg = ControlPredResConfig.from_yaml("configs/control_pred/resnet_config.yml")

# Initialize the model
model = ControlPredResModel(model_cfg)

checkpoint = load_file(cp_path)
model.load_state_dict(checkpoint)
model = model.core
model.eval()
model.cuda()
model.half()

@torch.no_grad()
def predict_on_samples(samples):
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
    fps_out = 15
    batch_size = 64
    latent_path = "experiments/sample_latents.pt"
    if os.path.exists(latent_path):
        samples = torch.load(latent_path)
        print(f"Loaded existing latents from {latent_path}")
    else:
        reader = TAStreamVideoReader(
            fps_in//fps_out,
            batch_size,
            224,
            224
        )
        reader.reset(video_path)
        for _ in tqdm(range(len(reader))):
            samples.append(reader.read(batch_size))
        
        samples = torch.cat(samples)
        torch.save(samples, latent_path)
        print(f"Saved latents to {latent_path}")

    samples, controls = predict_on_samples(samples)

    x = torch_to_np(samples[0])
    c = controls[0]


    frames = write_np_array_to_video(x, fps=15,controls=c,output_fmt="np")


    output_path = "experiments/sample_labelled.mp4"
    vwrite(output_path, frames, outputdict={
        "-r": "15",
        "-c:v": "libx264",
        "-crf": "18",
        "-preset": "slow"
    })



label_video("experiments/control_pred_irl/erratic.mp4")
print("Done")

