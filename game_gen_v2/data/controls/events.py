"""
Helper functions to simplify accessing things about events
"""

def timestamp(event):
    return event['timestamp']

def etype(event):
    return event['event_type']

def is_keyboard(event):
    return event['event_type'] == "KEYBOARD"

def is_mouse_button(event):
    return event['event_type'] == "MOUSE_BUTTON"

def get_key_idx(event):
    assert etype(event) in ["KEYBOARD", "KEY_TAP"]
    return event['event_args'][0]

def get_mouse_idx(event):
    assert etype(event) in ["MOUSE_BUTTON", "MOUSE_TAP"]
    return event['event_args'][0]

def is_down_press(event):
    assert etype(event) in ["KEYBOARD", "MOUSE_BUTTON"]
    return bool(event['event_args'][1])

