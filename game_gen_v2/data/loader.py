import os
import json
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import random
from pathlib import Path

class FrameDataset(IterableDataset):
    def __init__(
            self, data_dir,
            frame_count=100,
            image_transform = None, ignore_controls = False
        ):
        self.data_dir = Path(data_dir)
        self.frame_count = frame_count
        self.image_transform = image_transform
        self.ignore_controls = ignore_controls
        
        # Find all subdirectories
        self.subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        # Find all unique data object IDs by looking at _video.pt files across all subdirectories
        self.data_ids = []
        for subdir in self.subdirs:
            vt_files = list(subdir.glob("*_video.pt"))
            self.data_ids.extend([f.stem.replace('_video','') for f in vt_files])
        
        # Verify all required files exist
        for id in self.data_ids:
            subdir = self.get_subdir_for_id(id)
            assert (subdir / f"{id}_video.pt").exists(), f"Missing video file for {id}"
            assert (subdir / f"{id}_controls.pt").exists(), f"Missing controls file for {id}"
            assert (subdir / f"{id}_info.json").exists(), f"Missing meta file for {id}"

        # Load metadata and sort data_ids by src_vid_id and src_vid_pos
        id_to_meta = {}
        for id in self.data_ids:
            subdir = self.get_subdir_for_id(id)
            meta_path = subdir / f"{id}_info.json"
            with open(meta_path) as f:
                meta = json.load(f)
                id_to_meta[id] = meta

        # Sort data_ids based on src_vid_id first, then src_vid_pos
        self.data_ids.sort(key=lambda x: (
            id_to_meta[x]["src_vid_id"],
            id_to_meta[x]["src_vid_pos"]
        ))

        # Load video lengths from meta files
        self.vid_lens = {}
        for id in self.data_ids:
            subdir = self.get_subdir_for_id(id)
            meta_path = subdir / f"{id}_info.json"
            with open(meta_path) as f:
                meta = json.load(f)
                self.vid_lens[id] = meta["vid_len"]
        
        self.frames_upto = {}
        cumulative_frames = 0
        for id in self.data_ids:
            self.frames_upto[id] = cumulative_frames 
            cumulative_frames += self.vid_lens[id]

        self.total_samples = 0
        for id in self.vid_lens:
            if id in self.data_ids:  # Only count frames for non-filtered IDs
                self.total_samples += (self.vid_lens[id] - self.frame_count + 1)

    def get_subdir_for_id(self, id):
        """Helper method to get the subdirectory for a given id"""
        subdir_num = int(id) // 1000
        return self.data_dir / f"{subdir_num:03d}"

    def __iter__(self):
        while True:
            data_id = random.choice(self.data_ids)
            seq_len = self.vid_lens[data_id]
            
            if seq_len < self.frame_count:
                continue
                
            # Randomly sample a starting index that allows for frame_count frames
            start_idx = random.randint(0, seq_len - self.frame_count)
            
            subdir = self.get_subdir_for_id(data_id)
            vt_path = subdir / f"{data_id}_video.pt"
            it_path = subdir / f"{data_id}_controls.pt"
            
            yield {
                'vt_path': vt_path,
                'it_path': it_path,
                'div_idx' : start_idx
            }

    def normalize_controls(self, x):
        # x is [n,controls+2] button inputs and mouse inputs as int16 float

        # Mouse statistics from ~2 hours of gameplay with 30 FPS inputs
        # Only mean/std for nonzero values
        non_zero_mouse_x_std = 0.2310
        non_zero_mouse_y_std = 0.0487

        x = x.float()
        x[:,:,-2:] = x[:,:,-2:]/255 # [-1,1] now
        return x
    
    def sample_weighted(self, max_n):
        """
        Returns a number [0, max_n] that is heavily weighted towards earlier numbers.
        Uses an exponential distribution to achieve this weighting.
        """
        import math
        import random

        # Lambda parameter for exponential distribution
        # Higher lambda means more bias towards earlier numbers
        lambda_param = 8

        # Generate a random number from an exponential distribution
        x = random.expovariate(lambda_param)
        
        # Scale, floor, and clamp the number to fit our range
        index = int(min(max(math.floor(x * max_n), 0), max_n-1))
        
        return index

    @classmethod
    def create_loader(cls, batch_size, **dataloader_kwargs):
        return torch.utils.data.DataLoader(
            cls,
            batch_size=batch_size,
            collate_fn=cls.get_collate_fn(),
            **dataloader_kwargs
        )

from functools import partial
def collate_fn(batch, image_transform, frame_count, ignore_controls, normalize_controls):
    """
    Smart loading to make sure we don't load same file twice
    """
    # Control files are assumed to be small enough that it doesn't matter
    vt_paths = [x['vt_path'] for x in batch]
    it_paths = [x['it_path'] for x in batch]
    div_inds = [x['div_idx'] for x in batch]

    res_vid = []
    res_ctrl = []

    # Each file, which idx's is it serving?
    file_map = {}
    for i, path in enumerate(vt_paths):
        if path in file_map:
            file_map[path].append(i)
        else:
            file_map[path] = [i]

    for path in file_map:
        vid = torch.load(path, weights_only = False, map_location = 'cpu')
        ctrl = torch.load(str(path).replace("_video.pt", "_controls.pt"), weights_only = False, map_location = 'cpu')

        vid_len = len(vid)

        for i in file_map[path]:
            idx_into_vid = div_inds[i] # Index into video
            if idx_into_vid >= frame_count:
                slice_end = idx_into_vid
                slice_start = idx_into_vid - frame_count
            else:
                slice_start = idx_into_vid
                slice_end = idx_into_vid + frame_count

            res_vid.append(vid[slice_start:slice_end])
            res_ctrl.append(ctrl[slice_start:slice_end])

    res_vid = torch.stack(res_vid)

    if image_transform is not None:
        res_vid = image_transform(res_vid)

    if frame_count == 1 and len(res_vid.shape) == 5:
        res_vid = res_vid.squeeze(1)  # Remove the frames dimension when only 1 frame

    if ignore_controls:
        return res_vid

    res_ctrl = torch.stack(res_ctrl)
    res_ctrl = normalize_controls(res_ctrl)

    return res_vid, res_ctrl
    
def create_loader(dataset_kwargs, dataloader_kwargs):
    ds = FrameDataset(**dataset_kwargs)
    collate = partial(collate_fn, 
                     image_transform=ds.image_transform, 
                     frame_count=ds.frame_count,
                     ignore_controls=ds.ignore_controls,
                     normalize_controls=ds.normalize_controls)
    return DataLoader(
        ds,
        collate_fn = collate,
        **dataloader_kwargs
    )

if __name__ == "__main__":
    from tinygrad.helpers import Timing
    dataset = FrameDataset("E:/datasets/train_data", frame_count = 8)
    dataloader = dataset.create_loader(batch_size=4, num_workers=0)
    
    # Get first batch
    with Timing("Time to get a batch: "):
        vt_batch, it_batch = next(iter(dataloader))

    print(it_batch.shape)
    ctrl_btn = it_batch[...,:-2]
    ctrl_mouse = it_batch[...,-2:]

    print(ctrl_btn)
    print(ctrl_btn.float())
    #print(ctrl_mouse)
    exit()

    
    print(f"Visual tokens batch shape: {vt_batch.shape}")
    print(f"Image tokens batch shape: {it_batch.shape}")
