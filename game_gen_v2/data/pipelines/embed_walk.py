"""
Generates VAE embeddings over some folder and subdirs within that folder.
Folder
-> GameFolder
-> Game2Folder
--> Recording1
---> Recording1.mp4
---> Recording1_inputs.csv
"""

import os
from game_gen_v2.data.videos import TAStreamVideoReader
from game_gen_v2.nn.vae import VAE
from .data_config import FRAME_SKIP, VAE_BATCH_SIZE, OUT_H, OUT_W, IN_DIR, LATENT

import torch
from tqdm import tqdm

# Constants for this script
DATA_DIR = IN_DIR

def clear_embeds(data_dir, latent=LATENT):
    """
    Clear embeddings if we want to produce new ones
    """
    suffix = "_vt.pt" if latent else "_raw_vt.pt"
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('_vt.pt'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

class FileWalkLoader:
    """
    Reads file paths in a directory and creates paths for where the new pt files should go
    """
    def __init__(self, data_dir, latent=True, overwrite=False):
        self.data_dir = data_dir
        self.pending_files = []
        self.processed_files = set()
        self.overwrite = overwrite

        suffix = "_vt.pt" if latent else "_raw_vt.pt"

        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.mp4'):
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(root, file[:-4] + suffix)
                    if os.path.exists(output_path) and not self.overwrite:
                        if os.path.getsize(output_path) < 10 * 1024:  # Less than 10KB
                            os.remove(output_path)
                            self.pending_files.append((input_path, output_path))
                        else:
                            self.processed_files.add(input_path)
                    else:
                        self.pending_files.append((input_path, output_path))

    def __len__(self) -> int:
        return len(self.pending_files)
        
    def get_next(self):
        if not self.pending_files:
            return None
        
        input_path, output_path = self.pending_files.pop(0)
        self.processed_files.add(input_path)
        return (input_path, output_path)
    
def walking_embed(
    vae_batch_size,
    frame_skip,
    out_h,
    out_w,
    data_dir,
    latent=True
):
    if latent:
        model = VAE(force_batch_size=vae_batch_size)
        model.cuda()
        model.half()
    else:
        model = None
    
    reader = TAStreamVideoReader(
        frame_skip, vae_batch_size, out_h, out_w,
        normalize=latent
    )

    loader = FileWalkLoader(data_dir, latent)

    for _ in range(len(loader)):
        in_path, out_path = loader.get_next()
        print(in_path)
        print(out_path)

        reader.reset(in_path)
        frames = []
        for _ in tqdm(range(len(reader))):
            x = reader.read(vae_batch_size)
            if latent:
                x_enc = model.encode(x)
            else:
                x_enc = x
            frames.append(x_enc)

        frames = torch.cat(frames)
        torch.save(frames, out_path)
    

if __name__ == "__main__":
    walking_embed(
        VAE_BATCH_SIZE, FRAME_SKIP,
        OUT_H, OUT_W,
        DATA_DIR, LATENT
    )
    print("done")