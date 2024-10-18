"""
Mapping integers from my reading setup to actual keybinds
"""

ASCII_TO_KEY = {
    27: "ESCAPE",
    96: "BACKQUOTE",
    49: "1",
    50: "2",
    51: "3",
    52: "4",
    53: "5",
    54: "6",
    55: "7",
    56: "8",
    57: "9",
    48: "0",
    45: "MINUS",
    61: "EQUALS",
    8: "BACKSPACE",
    9: "TAB",
    16: "LSHIFT",
    17: "LCTRL",
    18: "LALT",
    # CAPS_LOCK doesn't have a standard ASCII code
    65: "A",
    66: "B",
    67: "C",
    68: "D",
    69: "E",
    70: "F",
    71: "G",
    72: "H",
    73: "I",
    74: "J",
    75: "K",
    76: "L",
    77: "M",
    78: "N",
    79: "O",
    80: "P",
    81: "Q",
    82: "R",
    83: "S",
    84: "T",
    85: "U",
    86: "V",
    87: "W",
    88: "X",
    89: "Y",
    90: "Z",
    91: "OPEN_BRACKET",
    93: "CLOSE_BRACKET",
    92: "BACK_SLASH",
    59: "SEMICOLON",
    39: "QUOTE",
    13: "ENTER",
    44: "COMMA",
    46: "PERIOD",
    47: "SLASH",
    32: "SPACE",
    # Many special keys don't have standard ASCII codes, so we'll omit them
}

def get_keycode(keycode_int):
    return ASCII_TO_KEY.get(keycode_int, f"Unknown key: {keycode_int}")


if __name__ == "__main__":
    print("Enter a keycode integer (or 'q' to quit):")
    while True:
        user_input = input("Keycode: ")
        if user_input.lower() == 'q':
            break
        try:
            keycode_int = int(user_input)
            key_str = get_keycode(keycode_int)
            print(f"Keycode {keycode_int} maps to: {key_str}")
        except ValueError:
            print("Invalid input. Please enter an integer or 'q' to quit.")
    print("Exiting...")

