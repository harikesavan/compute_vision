from typing import Tuple, List, NamedTuple, Union
import numpy as np
import cv2
import torch
from collections import namedtuple
import pandas as pd
import os
from glob import glob

capture = cv2.VideoCapture("./video_020.mp4")
FRAME_WIDTH, FRAME_HEIGHT = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

def _tlwh_to_xyxy(bbox_tlwh, frame_width, frame_height):
    x, y, w, h = bbox_tlwh
    x1 = max(int(x), 0)
    x2 = min(int(x+w), frame_width - 1)
    y1 = max(int(y), 0)
    y2 = min(int(y+h), frame_height - 1)
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)
    return x_center, y_center

def convert_txts(paths: List[str]) -> None:
    StrongSORTTrack = namedtuple("Track", "frame target_nr x y w h")
    PersonTrack = namedtuple('PersonTrack', 'Person_id x y')
    for path in paths:
        with open(path, 'r') as f:
            lines = [StrongSORTTrack(*[int(x) for x in x.split(' ') if len(x) > 0][:6]) for x in f.read().strip().split('\n')]
            persons_rectangles = [(track.target_nr, _tlwh_to_xyxy([track.x, track.y, track.w, track.h], FRAME_WIDTH, FRAME_HEIGHT)) for track in lines]
            result_lines = [f"Person_id: {p[0]} x: {p[1][0]} y: {p[1][1]}\n" for p in persons_rectangles]

        try:
            os.makedirs('output/')
        except FileExistsError:
            pass

        with open(f"output/{os.path.basename(path)}", "w") as f:
            f.writelines(result_lines)


if __name__ == '__main__':
    fpaths = glob('../Yolov5_StrongSORT_OSNet/runs/track/exp/tracks/*.txt')
    convert_txts(fpaths)
