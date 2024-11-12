"""
We read videos with TorchAudio streamer since it can be specified which GPU it uses and it overall has a lot of control
"""

from torchaudio.io import StreamReader
import torch
import numpy as np
import ffmpy

import json
import math
import subprocess
import torch.nn.functional as F

def yuv_2_rgb(videos, normalize = True):
    """
    Map [0,255] YUV [n,c,h,w] video tensor to [0,1] RGB tensor in half precision
    """
    videos = videos.float()
    y = videos[...,0,:,:]
    u = videos[...,1,:,:]
    v = videos[...,2,:,:]

    res = torch.empty_like(videos) # -> RGB
    res[...,0,:,:] = y + (1.370705 * (v - 128))
    res[...,1,:,:] = y - (0.698001 * (v - 128)) - (0.337633 * (u - 128))
    res[...,2,:,:] = y + (1.732446 * (u - 128))

    if normalize:
        res = (res/127.5 - 1).clamp(-1,1).half()
    else:
        res = res.byte()

    return res

class TAStreamVideoReader:
    """
    TorchAudio StreamReader for video reading with CUDA acceleration

    :param frame_skip: Skip every this many frames
    :param chunk_size: How many frames should it return at a time?
    :param out_h: Resolution height of returned frames
    :param out_w: Resolution width of returned frames
    """
    def __init__(self, frame_skip=1, chunk_size=100, out_h = 256, out_w = 256, normalize = True):
        self.frame_skip = frame_skip
        self.chunk_size = chunk_size
        self.stream_reader = None
        self.video_path = None
        self.stream_iterator = None
        self.normalize = normalize

        self.out_h = out_h
        self.out_w = out_w

    def reset(self, video_path):
        self.video_path = video_path
        if self.stream_reader is not None:
            del self.stream_reader
        self.stream_reader = StreamReader(video_path)
        config = {
            'decoder': 'h264_cuvid',
            'hw_accel': 'cuda:0',
            'filter_desc': f'select=not(mod(n\,{self.frame_skip}))',
        }
        self.stream_reader.add_video_stream(self.chunk_size, **config)
        self.stream_iterator = self.stream_reader.stream()

    def __len__(self):
        probe = ffmpy.FFprobe(
            inputs={self.video_path: None},
            global_options=['-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams']
        )
        stdout, stderr = probe.run(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        metadata = json.loads(stdout.decode('utf-8'))
        
        # Find the video stream
        video_stream = next((stream for stream in metadata['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream:
            total_frames = int(video_stream['nb_frames'])
            frames_after_skip = total_frames // self.frame_skip
            return math.ceil(frames_after_skip / self.chunk_size)
        else:
            return 0

    def read(self, n_frames):
        assert n_frames == self.chunk_size

        try:
            next_chunk = next(self.stream_iterator)[0]
            res = yuv_2_rgb(next_chunk, normalize=self.normalize)
            
            interp_fn = lambda x: F.interpolate(x, size = (self.out_h, self.out_w), mode = 'bilinear', align_corners = False)
            if not self.normalize:
                res = interp_fn(res.float()).byte()
            else:
                res = interp_fn(res)
            return res
        except StopIteration:
            return None