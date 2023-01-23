
BYTETRACK_PATH = 'runs/track/exp13-bytetrack-msmt-weights/tracks/video_020.txt'
STRONGSORT_PATH = 'runs/track/exp16-strongsort-good-reid/tracks/video_020.txt'
TOTAL_FRAMES = 2091

def get_amt_of_frames(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return len(lines)

print(get_amt_of_frames(BYTETRACK_PATH), get_amt_of_frames(STRONGSORT_PATH))
