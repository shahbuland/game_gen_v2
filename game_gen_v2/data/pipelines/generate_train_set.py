"""
Generates segments of data that can be used for final model to be trained on.
Must be called after .embed_walk.py and .make_input_tensors.py

Output data structure:
status.json -> Master file generated at start of dataset creation to determine which videos get put where
00000000_video.pt -> video frames for segment
00000000_controls.pt -> control tensor for segment
00000000_info.json -> metadata info including source video and position
"""

import os
import torch
import json
from tqdm import tqdm
import shutil

from game_gen_v2.data.controls.loading import load_inputs_tensor
from .data_config import SEGMENT_LENGTH, IN_DIR, OUT_DIR, LATENT, ASSUMED_SHAPE

def clear_dataset(data_dir=OUT_DIR):
    """
    Clear the entire dataset directory
    """
    if os.path.exists(data_dir):
        try:
            shutil.rmtree(data_dir)
            print(f"Removed dataset directory: {data_dir}")
        except OSError as e:
            print(f"Error removing dataset directory {data_dir}: {e}")
    else:
        print(f"Dataset directory does not exist: {data_dir}")

class FileIndex:
    """
    Index through dataset to return tuples of video tensors and their input tensors.
    Also returns useful info on these files
    """
    def __init__(self, dir = IN_DIR, assumed_shape = ASSUMED_SHAPE):
        self.files = []
        self.assumed_shape = assumed_shape
        suffix = "_vt.pt" if LATENT else "_raw_vt.pt"
        for root, _, files in os.walk(dir):
            for file in files:
                if file.endswith(suffix): # Each subdir has one vid tensor pt file (assumed)
                    self.files.append( # We assume an inputs tensor is present
                        (os.path.join(root, file), os.path.join(root, "inputs_it.pt"))
                    )
                
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, ind : int):
        return self.files[ind]
    
    def get_length(self, ind : int, assumed_bytes : int = 1):
        """
        Get length of ind-th video in frame count
        Note that this value is approximate, but loading the whole tensor might be hard
        """
        file_size = os.path.getsize(self.files[ind][0])
        num_values = file_size // assumed_bytes
        num_frames = num_values // (self.assumed_shape[0] * self.assumed_shape[1] * self.assumed_shape[2])
        return num_frames

class VideoData:
    """
    Container class for the data for a specific video
    """
    def __init__(self, vt_path, it_path, segment_length = SEGMENT_LENGTH):
        self.vt = torch.load(vt_path)
        self.it = torch.load(it_path)[:len(self.vt)]
        self.n_frames = len(self.vt)
        self.segment_length = segment_length
        self.n_segments = self.n_frames // self.segment_length


    def get_segment(self, seg_idx):
        """Get the seg_idx'th segment from this video"""
        start_idx = seg_idx * self.segment_length
        end_idx = start_idx + self.segment_length
        
        if end_idx > self.n_frames:
            return None
            
        return {
            'video': self.vt[start_idx:end_idx].clone(),
            'controls': self.it[start_idx:end_idx].clone()
        }

class Logger:
    """
    Logs progress on dataset generation in case a crash happens and we need to resume
    """
    def __init__(self, dir = OUT_DIR, segment_length = SEGMENT_LENGTH):
        self.fp = os.path.join(dir, "status.json")
        self.fresh = False
        self.segment_length = segment_length
        try:
            with open(self.fp, 'r') as f:
                self.info = json.load(f)
        except:
            self.fresh = True
            self.info = {}
            
    def save(self):
        with open(self.fp, 'w') as f:
            json.dump(self.info, f, indent=4)
    
    def prepare(self, index):
        frame_lengths = [index.get_length(i) for i in range(len(index))]
        if self.fresh:
            self.info = {
                'segment_length': self.segment_length,
                'frames_per_videos': frame_lengths,
                'vid_idx': 0,
                'seg_idx': 0,
                'segments_created': 0
            }
            self.save()
        return self.info

    def step(self):
        self.info['segments_created'] += 1
        self.info['seg_idx'] += 1
        self.save()
        
    def next_video(self):
        self.info['vid_idx'] += 1
        self.info['seg_idx'] = 0
        self.save()

def generate_train_set(in_dir, out_dir, latent, segment_length, assumed_shape):
    os.makedirs(out_dir, exist_ok=True)
    index = FileIndex(in_dir, assumed_shape)
    logger = Logger(dir=out_dir, segment_length=segment_length)

    info = logger.prepare(index)
    suffix = "_video.pt" if latent else "_raw_video.pt"
    
    segments_created = info['segments_created']
    
    for vid_idx in range(info['vid_idx'], len(index)):
        video_path, control_path = index[vid_idx]
        video = VideoData(video_path, control_path, segment_length)
        
        for seg_idx in range(info['seg_idx'], video.n_segments):
            segment = video.get_segment(seg_idx)
            if segment is None:
                continue
                
            # Save segment files
            filename = f"{segments_created:08d}"
            torch.save(segment['video'], os.path.join(out_dir, f"{filename}_{suffix}"))
            torch.save(segment['controls'], os.path.join(out_dir, f"{filename}_controls.pt"))
            
            # Save segment metadata
            info_data = {
                "src_vid_id": vid_idx,
                "src_vid_pos": seg_idx * segment_length,
                "vid_len": len(segment['video']),
                "src_folder": os.path.dirname(index[vid_idx][0])
            }
            with open(os.path.join(out_dir, f"{filename}_info.json"), 'w') as f:
                json.dump(info_data, f, indent=4)
            
            segments_created += 1
            logger.step()
        
        logger.next_video()
        
    logger.save()

if __name__ == "__main__":
    generate_train_set(IN_DIR, OUT_DIR, LATENT, SEGMENT_LENGTH, ASSUMED_SHAPE)