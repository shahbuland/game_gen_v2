from torch.utils.data import Dataset, DataLoader
import os
import torch

class GameData(Dataset):
    """
    Assumes data has been generated through appropriate pipelines into tensor dataset
    """
    def __init__(self, data_root = "game_gen_v2/data/train_data"):
        super().__init__()

        self.root = data_root
        files = [os.path.join(data_root, path) for path in os.listdir(data_root)]
        files = sorted([f for f in files if f.endswith('_vt.pt')])
        if not files:
            raise ValueError(f"No '_vt.pt' files found in {data_root}")
        
        # Extract the largest number from the file names
        max_num = max(int(os.path.basename(f).split('_')[0]) for f in files)
        self.N = max_num

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        path_1 = os.path.join(self.root, f"{idx:08d}_vt.pt")
        path_2 = os.path.join(self.root, f"{idx:08d}_it.pt")

        return torch.load(path_1), torch.load(path_2)

    def get_max_file_number(self):
        return self.max_file_number

def create_loader(batch_size, *args, **kwargs):
    return DataLoader(GameData(), batch_size = batch_size, *args, **kwargs)