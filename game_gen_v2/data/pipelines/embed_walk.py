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
from .data_config import FRAME_SKIP, VAE_BATCH_SIZE, OUT_H, OUT_W, IN_DIR

import torch
from tqdm import tqdm

# Constants for this script
DATA_DIR = IN_DIR

def clear_embeddings(data_dir):
    """
    Clear embeddings if we want to produce new ones
    """
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
    def __init__(self, data_dir, overwrite=False):
        self.data_dir = data_dir
        self.pending_files = []
        self.processed_files = set()
        self.overwrite = overwrite

        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.mp4'):
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(root, file[:-4] + '_vt.pt')
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

if __name__ == "__main__":
    model = VAE(force_batch_size = VAE_BATCH_SIZE)
    model.cuda()
    model.half()

    reader = TAStreamVideoReader(
        FRAME_SKIP,
        VAE_BATCH_SIZE,
        OUT_H,
        OUT_W
    )

    loader = FileWalkLoader(DATA_DIR)
    
    for _ in range(len(loader)):
        in_path, out_path = loader.get_next()

        reader.reset(in_path)
        frames = []
        for _ in tqdm(range(len(reader))):
            x = reader.read(VAE_BATCH_SIZE)
            x_enc = model.encode(x)
            frames.append(x_enc)

        frames = torch.cat(frames)
        torch.save(frames, out_path)

    print("Done")