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

from game_gen_v2.data.controls.loading import load_inputs_tensor
from .data_config import SEGMENT_LENGTH, IN_DIR, OUT_DIR

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
        self.vt = torch.load(vt_path)
        self.it = torch.load(it_path)[:len(self.vt)]
        self.n_frames = len(self.vt)
        self.n_segments = self.n_frames // SEGMENT_LENGTH

    def get_segment(self, seg_idx):
        """Get the seg_idx'th segment from this video"""
        start_idx = seg_idx * SEGMENT_LENGTH
        end_idx = start_idx + SEGMENT_LENGTH
        
        if end_idx > self.n_frames:
            return None
            
        return {
            'video': self.vt[start_idx:end_idx].contiguous(),
            'controls': self.it[start_idx:end_idx].contiguous()
        }

class Logger:
    """
    Logs progress on dataset generation in case a crash happens and we need to resume
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
            
    def save(self):
        with open(self.fp, 'w') as f:
            json.dump(self.info, f, indent=4)
    
    def prepare(self, index):
        frame_lengths = [index.get_length(i) for i in range(len(index))]
        if self.fresh:
            self.info = {
                'segment_length': SEGMENT_LENGTH,
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

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    index = FileIndex()
    logger = Logger()

    info = logger.prepare(index)
    
    segments_created = info['segments_created']
    
    for vid_idx in range(info['vid_idx'], len(index)):
        video = VideoData(*index[vid_idx])
        
        for seg_idx in range(info['seg_idx'], video.n_segments):
            segment = video.get_segment(seg_idx)
            if segment is None:
                continue
                
            # Save segment files
            filename = f"{segments_created:08d}"
            torch.save(segment['video'], os.path.join(OUT_DIR, f"{filename}_video.pt"))
            torch.save(segment['controls'], os.path.join(OUT_DIR, f"{filename}_controls.pt"))
            
            # Save segment metadata
            info_data = {
                "src_vid_id": vid_idx,
                "src_vid_pos": seg_idx,
                "vid_len": len(segment['video']),
                "src_folder": os.path.dirname(index[vid_idx][0])  # Add the source folder path
            }
            with open(os.path.join(OUT_DIR, f"{filename}_info.json"), 'w') as f:
                json.dump(info_data, f, indent=4)
            
            segments_created += 1
            logger.step()
            
        logger.next_video()
        
    logger.save()
