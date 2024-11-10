# For video data + fps on input controls
VAE_BATCH_SIZE = 64
VAE_PATH = "madebyollin/taesdxl"
OUT_H = 256
OUT_W = 256
FPS_IN = 60
FPS_OUT = 30
FRAME_SKIP = FPS_IN // FPS_OUT
SEGMENT_LENGTH = 5000 # Length of segments in output dataset
# Keybinds to keep data for
KEYBINDS = ["SPACE", "W", "A", "S", "D", "R", "E", "G", "F", "Q", "CONTROL", "SHIFT"]

# Directories
IN_DIR = "D:/datasets/BlackOpsColdWar" # where all the .mp4 + .csv data is
OUT_DIR = "./datasets/train_data" # where to save segments