"""
Processing control CSVs
"""

from .mappings import get_keycode

import pandas as pd
import torch
import math
from tqdm import tqdm

# Default keys to look at
DEFAULT_KEYS = ["SPACE", "W", "A", "S", "D", "R", "E", "G", "F", "Q", "CONTROL", "SHIFT"]

def filter_alike_button_events(events):
    """
    :param events: Events across a singular frame

    Collates button events that occured within a single frame to prioritize the ones that occurred most recently.
    """
    key_set = set()
    indices_to_delete = []

    # Iterate through events backwards
    for i, event in reversed(list(enumerate(events))):
        if event['event_type'] in ['KEYBOARD', 'MOUSE_BUTTON']:
            key_idx = event['event_args'][0] 
            
            if key_idx not in key_set:
                key_set.add(key_idx)
            else:
                # If the key is already in the set, mark this index for deletion
                indices_to_delete.append(i)
    
    # Delete marked indices from the events list
    for index in sorted(indices_to_delete, reverse=True):
        del events[index]

    return events

def sum_mouse_moves(events):
    """
    :param events: Events in a singular frame

    Get average mouse displacement during frame
    """
    if not events:
        return []

    sum_x, sum_y = 0, 0

    for event in events:
        x, y = event['event_args'][0], event['event_args'][1]
        sum_x += x
        sum_y += y

    sum_x = int(min(sum_x, 255)) # uint8
    sum_y = int(min(sum_y, 255))
    sum_x = max(-255, sum_x)
    sum_y = max(-255, sum_y)

    return [{'event_type': 'MOUSE_MOVE', 'event_args': [sum_x, sum_y]}]

def read_events(df, fps, num_frames = None, num_seconds = None, keys_of_interest = DEFAULT_KEYS):
    """
    Given loaded dataframe (see .loading.py), reads events assuming given FPS
    I.e. if FPS = 15, then gets frame_time of 1/fps, collates any events that occured within time of a single frame
    """

    # Get time of single frame
    frame_time = 1./fps
    if num_seconds is not None:
        num_frames = fps * num_seconds

    # Assume loaded with load_inputs_data 
    # Get relevant times we will iterate over
    start_time = df.iloc[0]['timestamp']
    crnt_time = start_time
    end_time = df.iloc[-1]['timestamp']
    if num_frames is not None:
        end_time = min(end_time, start_time + (num_frames-1) * frame_time)

    event_idx = 1 # Skip start event

    frame_events = []

    # Collect events
    total_steps = math.floor((end_time-start_time)/frame_time)

    for _ in tqdm(range(total_steps)):
        batched_events = []
        while df.iloc[event_idx]['timestamp'] <= crnt_time:
            row = df.iloc[event_idx]
            batched_events.append(row)
            event_idx += 1
        
        final_events = []
        if batched_events:
            # Filter keyboard and mouse button events
            button_events = [event for event in batched_events if event['event_type'] in ['KEYBOARD', 'MOUSE_BUTTON']]
            if button_events:
                filtered_button_events = filter_alike_button_events(button_events)
                final_events += filtered_button_events

            # Filter mouse move events
            mouse_move_events = [event for event in batched_events if event['event_type'] == 'MOUSE_MOVE']
            if mouse_move_events: 
                summed_mouse_move_events = sum_mouse_moves(mouse_move_events)
                final_events += summed_mouse_move_events
        frame_events.append(final_events)
        crnt_time += frame_time

    #print(f"{len(frame_events)} events collected.")
    #print("BUILDING TENSOR...")
    
    # Keys and LMB/RMB as 0|1 and mouse dx and dy
    res = torch.zeros(len(frame_events), len(keys_of_interest)+4, dtype=torch.int16)
    for i in tqdm(range(len(frame_events))):
        for event in frame_events[i]:
            if event['event_type'] == 'KEYBOARD':
                #print(event['event_args'][0])
                #print(get_keycode(event['event_args'][0]))
                key = get_keycode(event['event_args'][0])
                if key in keys_of_interest:
                    key_idx = keys_of_interest.index(key)
                    if not event['event_args'][1]: # key up
                        res[i:, key_idx] = 0
                    elif event['event_args'][1]: # key down
                        res[i:, key_idx] = 1
            elif event['event_type'] == 'MOUSE_BUTTON':
                if event['event_args'][0] == 0: # LMB
                    res[i:, -4] = 1 if event['event_args'][2] else 0 # 1 if down else 0
                elif event['event_args'][0] == 1: # RMB
                    res[i:, -3] = 1 if event['event_args'][-1] else 0
            elif event['event_type'] == 'MOUSE_MOVE':
                res[i, -2] = event['event_args'][0] # dx
                res[i, -1] = event['event_args'][1] # dy
    
    #print(res.shape)
    return res