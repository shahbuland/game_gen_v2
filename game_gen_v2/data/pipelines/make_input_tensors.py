"""
Convert all input csv files into input tensors "_it.pt"
"""

import os
import torch

from game_gen_v2.data.controls.loading import load_inputs_tensor

DATA_DIR = "game_gen_v2/data/datasets"
FPS = 15
KEYBINDS = ["SPACE", "W", "A", "S", "D", "R", "E", "G", "F", "Q", "CONTROL", "SHIFT"]

def clear_embeddings(data_dir):
    """
    Clear embeddings if we want to produce new ones
    """
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('_it.pt'):
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
                if file.endswith('inputs.csv'):
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(root, file[:-4] + '_it.pt')
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
    loader = FileWalkLoader(DATA_DIR)

    while True:
        next_file = loader.get_next()
        if next_file is None:
            break

        input_path, output_path = next_file
        print(input_path)
        
        input_tensor = load_inputs_tensor(input_path, FPS, KEYBINDS)
        torch.save(input_tensor, output_path)
        print(f"Processed and saved: {output_path}")

    print("Processing complete.")