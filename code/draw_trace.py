from typing import Tuple, List, NamedTuple, Union
import numpy as np
import cv2
from collections import namedtuple
import pandas as pd

track = namedtuple("Track", "frame target_nr x y w h")
def stack_trace(centerpoints: List[Tuple[int]],
                init_frame: np.ndarray,
                pixel_threshold: int = 35) -> List[Tuple[int]]:
    p1, p2 = 0, 1
    trace = init_frame
    while p2 < len(centerpoints):
        if (np.abs(np.asarray(centerpoints[p1]) - np.asarray(centerpoints[p2])) <= pixel_threshold).all():
            trace = cv2.line(trace, centerpoints[p1], centerpoints[p2], (0, 0, 255), thickness=2)
        p2 += 1
        p1 += 1
    return trace

def _tlwh_to_xyxy(bbox_tlwh, frame_width, frame_height):
    x, y, w, h = bbox_tlwh
    x1 = max(int(x), 0)
    x2 = min(int(x+w), frame_width - 1)
    y1 = max(int(y), 0)
    y2 = min(int(y+h), frame_height - 1)
    return x1, y1, x2, y2

def main(capture: cv2.VideoCapture,
         writer: cv2.VideoWriter,
         frame_count: int,
         frame_tuples_dict: dict,
         frame_wh: List[int],
         center_history: List):

    # Read the next frame
    success, frame = capture.read()

    # Check if the frame was read successfully
    if not success:
        return 'stop'

    print(frame_count)
    try:
        # contours = get_contours(frame)
        if frame_count in frame_tuples_dict.keys():
            frame_tuple = frame_tuples_dict[frame_count]
            print('drawing trace', count, frame_tuple.frame)
            xywh = [int(frame_tuple.x), int(frame_tuple.y), int(frame_tuple.w), int(frame_tuple.h)]
            x1, y1, x2, y2 = _tlwh_to_xyxy(xywh, *frame_wh)
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            # The average center point can be represented as a tuple of the mean x and y coordinates
            center_point = (x_center, y_center)
            # rect_centerpoint = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            trace = frame
            if len(center_history) != 0:
                trace = stack_trace(center_history, trace)
                writer.write(trace)
            center_history.append(center_point)
        else:
            trace = stack_trace(center_history, frame)
            writer.write(trace)

    except Exception as e:
        trace = stack_trace(center_history, frame)
        writer.write(trace)
        print(f'Exception: {e}. Writing original frame instead.')
        return 'continue'


if __name__ == '__main__':

    with open('./video_020.txt', 'r') as f:
        lines = [track(*[int(x) for x in x.split(' ') if len(x) > 0][:6]) for x in f.read().strip().split('\n')]
        lines_dict = {k.frame:k for k in lines}

    trace_dframe = pd.DataFrame(lines, columns=track._fields)
    capture = cv2.VideoCapture("./video_020.mp4")
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frame_height, frame_width)
    # # Define the codec and create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter("./MOG_Tracker_strongSORT.avi", fourcc, 30.0, (frame_width, frame_height))
    # trajectory_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    count = 0
    center_history = []
    while capture.isOpened():
        s = main(capture, writer, count, lines_dict, [frame_width, frame_height], center_history)
        count += 1
        if s == 'stop':
            break

    trace_dframe.to_csv('./strongSORT_track.txt', index=False)
    capture.release()
    writer.release()
    cv2.destroyAllWindows()
