"""
Generates segments of data that can be used for final model to be trained on.

Must be called after .embed_walk.py and .make_input_tensors.py

Output data structure:
status.json -> Master file generated at start of dataset creation to determine which videos get put where
00000000_video.pt -> compressed video frames
00000000_controls.pt -> control tensor
00000000_info.json -> metadata info 
"""

import os
import torch
import json

from game_gen_v2.data.controls.loading import load_inputs_tensor

IN_DIR = "game_gen_v2/data/datasets/BlackOpsColdWar"
OUT_DIR = "game_gen_v2/data/train_data"

# Configuration for generated dataset
N_ENTRIES = 100000
N_FRAMES_PER_SAMPLE = 150 # 10 seconds

class FileIndex:
    """
    Index through dataset to return tuples of video tensors and their input tensors.
    Also returns useful info on these files
    """
    def __init__(self, dir = IN_DIR):
        self.files = []
        for root, _, files in os.walk(dir):
            for file in files:
                if file.endswith("_vt.pt"): # Each subdir has one vid tensor pt file (assumed)
                    self.files.append( # We assume an inputs tensor is present
                        (os.path.join(root, file), os.path.join(root, "inputs_it.pt"))
                    )
                
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, ind : int):
        return self.files[ind]
    
    def get_length(self, ind : int, assumed_shape = (4,32,32), assumed_bytes : int = 2):
        """
        Get length of ind-th video in frame count
        Note that this value is approximate, but loading the whole tensor might be hard
        :param assumed_shape: How big is the tensor going to be?
        :param assumed_bytes: How many bytes is each value in tensor going to be? (i.e. fp16 = 2, fp8 = 1, etc.)
        """
        file_size = os.path.getsize(self.files[ind][0])
        num_values = file_size // assumed_bytes
        num_frames = num_values // (assumed_shape[0] * assumed_shape[1] * assumed_shape[2])
        return num_frames

class VideoData:
    """
    Container class for the data for a specific video
    """
    def __init__(self, vt_path, it_path):
        # vid tensor and input tensor
        self.vt = torch.load(vt_path) 
        self.it = torch.load(it_path)[:len(self.vt)]

        self.ind = 0
        self.n_frames = len(self.vt)
        self.chunk_size = N_FRAMES_PER_SAMPLE

    def skip(self, steps, stride):
        """
        Skip forward in both tensors (i.e. to resume somewhere)
        Returns a bool indicating if this skip exhausted the video or not
        """
        self.ind = steps * stride
        if self.ind+self.chunk_size > self.n_frames:
            return True
        return False
    
    def save_next(self, path, filename, stride) -> bool:
        """
        Save the next chunk
        Returns a bool indicating if the video is exhausted or not 

        :param path: Folder dir where we will save files
        :param filename: Base name for both files
        :param stride: How much to move forward after saving this chunk
        """

        vt_path = os.path.join(path, f"{filename}_vt.pt")
        it_path = os.path.join(path, f"{filename}_it.pt")

        vt_chunk = self.vt[self.ind:self.ind+self.chunk_size].contiguous()
        it_chunk = self.it[self.ind:self.ind+self.chunk_size].contiguous()

        torch.save(vt_chunk.contiguous().clone(), vt_path)
        torch.save(it_chunk.contiguous().clone(), it_path)

        self.ind += stride

        if self.ind+self.chunk_size >= self.n_frames:
            return True
        return False

class Logger:
    """
    Logs progress on dataset generation in case a crash happens and we need to resume

    self.info is dict of
    - total_samples: How many total samples we want to produce
    - frames_per_sample: How many frames each sample has
    - stride: Stride taken in videos
    - vid_idx: Which video we're currently processing
    - frame_idx: Step in that video
    """
    def __init__(self, dir = OUT_DIR):
        self.fp = os.path.join(dir, "status.json")
        self.fresh = False
        try:
            with open(self.fp, 'r') as f:
                self.info = json.load(f)
        except:
            self.fresh = True
            self.info = {}

        self.accum = 0
        self.save_interval = 100
        
    def save(self):
        with open(self.fp, 'w') as f:
            json.dump(self.info, f, indent=4)
        
    def accum_save(self):
        self.accum += 1
        if (self.accum % self.save_interval) == 0:
            self.save()
    
    def prepare(self, index, stride):
        frame_lengths = [index.get_length(i) for i in range(len(index))]
        if self.fresh:
            self.info = {
                'total_samples' : N_ENTRIES,
                "frames_per_sample" : N_FRAMES_PER_SAMPLE,
                "stride" : stride,
                "frames_per_videos" : frame_lengths,
                "vid_idx" : 0,
                "frame_step" : 0,
                "samples_so_far" : 0
            }
            self.save()
        return self.info
    
    def step_video(self):
        self.info['vid_idx'] += 1
        self.info['frame_step'] = 0
        
        self.accum_save()

    def step(self):
        self.info['samples_so_far'] += 1
        self.info['frame_step'] += 1
        
        self.accum_save()

    def get_info(self):
        return self.info

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok = True)
    index = FileIndex()
    logger = Logger()

    print(index.files[0][0])
    print(index.get_length(0))

    n_files = len(index)
    total_frames = sum([index.get_length(i) for i in range(n_files)])
    stride = max(1, (total_frames - N_FRAMES_PER_SAMPLE) // (N_ENTRIES - 1))

    info = logger.prepare(index, stride)
    samples_so_far = info['samples_so_far']

    for i in range(info['vid_idx'], n_files):
        info = logger.get_info()
        video = VideoData(*index[i])
        exhausted = video.skip(info['frame_step'], stride)

        while not exhausted:
            filename = f"{samples_so_far:08d}"
            exhausted = video.save_next(OUT_DIR, filename, stride)

            logger.step()
            samples_so_far += 1
        
        logger.step_video()
        if samples_so_far >= N_ENTRIES:
            break

    logger.save()