"""
Loading controls from CSVs
"""

import pandas as pd
import ast

from .processing import read_events

def load_inputs_data(data_path):
    """
    Loads the input data into a CSV file with the following basic processing:
    1. Drops everything before start record event
    2. Drops everything after end recording event
    3. Updates all timestamps to become time since recording start
    """
    df = pd.read_csv(data_path)
    # Apply ast.literal_eval to 'event_args' column
    df['event_args'] = df['event_args'].apply(ast.literal_eval)

    # Find the index of the "START" event
    start_index = df[df['event_type'] == "START"].index[0]
    # Get the timestamp of the "START" event
    start_timestamp = df.iloc[start_index]['timestamp']
    
    # Subtract the start timestamp from all events
    df['timestamp'] = df['timestamp'] - start_timestamp
    
    # Find the index of the "END" event
    end_index = df[df['event_type'] == "END"].index[0]
    
    # Slice the DataFrame to keep only rows between "START" and "END" events, inclusive
    df = df.iloc[start_index:end_index+1]
    
    # Reset the index of the DataFrame
    df = df.reset_index(drop=True)
    return df

def load_inputs_tensor(data_path, fps, keybinds):
    """
    Create a tensor of inputs at given fps

    :param data_path: Path to inputs csv file
    :param fps: FPS to read inputs at
    :param keybinds: List of keys (see mappings.py names) that we want to include in data
    """
    df = load_inputs_data(data_path)
    events_tensor = read_events(df, fps, keys_of_interest=keybinds)
    return events_tensor