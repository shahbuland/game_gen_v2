import torch
import os
import random
from pathlib import Path
import einops as eo
import wandb

class VAESampler:
    def __init__(
            self,
            input_directory,
            image_transform,
            fps = 30,
            sample_cache = "sampler_cache_vae.pt",
            n_samples = 8,
            sample_length = 300,
            batch_size = 64
    ):
        self.input_directory = input_directory
        self.image_transform = image_transform
        self.n_samples = n_samples
        self.sample_length = sample_length
        self.batch_size = batch_size
        self.fps = fps

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
        
        res = torch.stack(input_samples)
        if self.image_transform is None:
            return res
        return self.image_transform(res)     

    @torch.no_grad()
    def batch_enc_dec(self, model, batch):
        # batch is [b,n,c,h,w]
        b = batch.shape[0]
        batch = eo.rearrange(batch, 'b n c h w -> (b n) c h w')
        rec = model.decode(model.encode(batch))
        rec = eo.rearrange(rec, '(b n) c h w -> b n c h w', b = b).clamp(-1,1)
        return rec

    @torch.no_grad()
    def torch_to_wandb(self, data):
        # data is [b,n,c,h,w] [-1,1]
        videos = [(255*((video + 1)/2)).byte().detach().cpu().numpy() for video in data]
        videos = [wandb.Video(video, fps=self.fps,format='mp4') for video in videos]
        return videos
    
    @torch.no_grad()
    def __call__(self, model):
        data = self.input_samples.clone()
        seq_chunk_size = self.batch_size // self.n_samples

        for i in range(self.sample_length // seq_chunk_size):
            slice_start = i*seq_chunk_size
            slice_end = (i+1)*seq_chunk_size
            data[:,slice_start:slice_end] = self.batch_enc_dec(model, data[:,slice_start:slice_end])
        
        # both are b,n,c,h,w
        stacked_vids = torch.cat([self.input_samples,data],dim=-2) # Stack on top of their rec counterpart
        return self.torch_to_wandb(stacked_vids)


