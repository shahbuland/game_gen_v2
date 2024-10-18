"""
Processing control CSVs
"""

from .mappings import get_keycode
from .events import (
    timestamp,
    etype,
    get_key_idx,
    get_mouse_idx,
    is_down_press,
    is_keyboard,
    is_mouse_button
)

import pandas as pd
import torch
import math
from tqdm import tqdm

# Default keys to look at
DEFAULT_KEYS = ["SPACE", "W", "A", "S", "D", "R", "E", "G", "F", "Q", "CONTROL", "SHIFT"]

def filter_alike_button_events(events):
    """
    :param events: Events across a singular frame

    Collates button events that occurred within a single frame to prioritize the ones that occurred most recently.
    In order to ensure key down events are captured, we create new event types "KEY_TAP" and "MOUSE_TAP"
    that are created if a key down event is followed by a key up event for keyboard and mouse respectively.
    """
    key_set = set()
    mouse_set = set()

    key_down_set = set()
    mouse_down_set = set()
    
    indices_to_delete = []

    # Iterate through events backwards
    for i, event in reversed(list(enumerate(events))):
        # Only care about mouse or keyboard events
        # In both cases, add down events to X_down_set
        # Add most recent key alone to X_set
        # Things that aren't recent are marked for deletion
        if is_keyboard(event):
            key_idx = get_key_idx(event)
            if is_down_press(event):
                key_down_set.add(key_idx)
            if key_idx not in key_set:
                key_set.add(key_idx)
            else:
                indices_to_delete.append(i)

        if is_mouse_button(event):
            mouse_idx = get_mouse_idx(event)
            if is_down_press(event):
                mouse_down_set.add(mouse_idx)
            if mouse_idx not in mouse_set:
                mouse_set.add(mouse_idx)
            else:
                indices_to_delete.append(i)
    
    # Deletion
    for index in sorted(indices_to_delete, reverse = True):
        del events[index]
    
    # Key tap events, if key down occured *at all* it should be a X_tap event type
    for event in events:
        if not is_down_press(event): # key up
            if is_keyboard(event):
                if get_key_idx(event) in key_down_set:
                    event['event_type'] = 'KEY_TAP'
            if is_mouse_button(event):
                if get_mouse_idx(event) in mouse_down_set:
                    event['event_type'] = 'MOUSE_TAP'

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
            # Taps take priority as they should supercede down events
            if etype(event) == 'KEY_TAP':
                key = get_keycode(get_key_idx(event))
                if key in keys_of_interest:
                    key_idx = keys_of_interest.index(key)
                    res[i, key_idx] = 1  # Set to 1 only for this frame
                    if i < (len(frame_events)-1): res[i+1:, key_idx] = 0 # An up was also logged, so set rest to false
            elif etype(event) == "MOUSE_TAP":
                if get_mouse_idx(event) == 1: # LMB
                    res[i,-4] = 1
                    if i < (len(frame_events)-1): res[i+1:,-4] = 0
                if get_mouse_idx(event) == 2: # RMB
                    res[i,-3] = 1
                    if i < (len(frame_events)-1): res[i+1:,-3] = 0
            elif etype(event) == 'KEYBOARD':
                key = get_keycode(get_key_idx(event))
                if key in keys_of_interest:
                    key_idx = keys_of_interest.index(key)
                    if is_down_press(event):  # key down
                        res[i:, key_idx] = 1
                    else:  # key up
                        res[i:, key_idx] = 0
            elif etype(event) == 'MOUSE_BUTTON':
                if get_mouse_idx(event) == 1:  # LMB
                    res[i:, -4] = 1 if is_down_press(event) else 0  # 1 if down else 0
                elif get_mouse_idx(event) == 2:  # RMB
                    res[i:, -3] = 1 if is_down_press(event) else 0
            elif event['event_type'] == 'MOUSE_MOVE':
                res[i, -2] = event['event_args'][0]  # dx
                res[i, -1] = event['event_args'][1]  # dy
    #print(res.shape)
    return res