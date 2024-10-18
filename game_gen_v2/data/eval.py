import torch
import torch.nn.functional as F

from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from PIL import Image

from .data import create_loader

class Validator:
    def __init__(self, validation_loader, val_batch_size : int, total_size = 10000):
        self.loader = validation_loader
        self.total_size = total_size
        self.b_size = val_batch_size

    @torch.no_grad()
    def __call__(self, model):
        total_loss = 0.
        n_samples = 0
        print("Validating...")
        for batch in tqdm(self.loader, total=self.total_size // self.b_size):
            loss, extra = model(batch)
            total_loss += loss.item()

            n_samples += self.b_size
            if n_samples >= self.total_size:
                break

        return loss.item() / self.total_size