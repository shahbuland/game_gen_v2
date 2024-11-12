from .data_config import SEGMENT_LENGTH, IN_DIR, OUT_DIR

import os
import torch

def calculate_diversity_indices(controls_tensor):
    # controls is [n,d]
    btns = controls_tensor[:,:-2]
    mouse = controls_tensor[:,-2:]

    weights = btns.sum(-1) + (mouse[:,0]**2 + mouse[:,1]**2).sqrt()/140
    _, sorted_indices = torch.sort(weights, descending=True)
    return sorted_indices

def process_control_files(out_dir=OUT_DIR):
    for filename in os.listdir(out_dir):
        if filename.endswith("_controls.pt"):
            # Extract the numeric part of the filename
            file_number = filename[:8]
            
            # Load the controls tensor
            controls_path = os.path.join(out_dir, filename)
            controls_tensor = torch.load(controls_path)
            
            # Calculate diversity indices
            diversity_indices = calculate_diversity_indices(controls_tensor)
            
            # Save the diversity indices
            diversity_path = os.path.join(out_dir, f"{file_number}_diversity_inds.pt")
            torch.save(diversity_indices, diversity_path)

if __name__ == "__main__":
    process_control_files()
