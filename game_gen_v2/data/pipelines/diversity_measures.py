from .data_config import SEGMENT_LENGTH, IN_DIR, OUT_DIR, FILES_PER_SUBFOLDER

import os
import torch
from pathlib import Path

def calculate_diversity_indices(controls_tensor):
    # controls is [n,d]
    btns = controls_tensor[:,:-2]
    mouse = controls_tensor[:,-2:]

    weights = btns.sum(-1) + (mouse[:,0]**2 + mouse[:,1]**2).sqrt()/140
    _, sorted_indices = torch.sort(weights, descending=True)
    return sorted_indices

def process_control_files(out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    
    # Iterate through all subdirectories
    for subdir in out_dir.iterdir():
        if subdir.is_dir():
            for filename in subdir.iterdir():
                if filename.name.endswith("_controls.pt"):
                    # Extract the numeric part of the filename
                    file_number = filename.stem[:8]
                    
                    # Load the controls tensor
                    controls_tensor = torch.load(filename)
                    
                    # Calculate diversity indices
                    diversity_indices = calculate_diversity_indices(controls_tensor)
                    
                    # Save the diversity indices in the same subdirectory
                    diversity_path = subdir / f"{file_number}_diversity_inds.pt"
                    torch.save(diversity_indices, diversity_path)

if __name__ == "__main__":
    process_control_files()