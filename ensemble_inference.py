from typing import List
import numpy as np
import cv2
import torch
import yolov5


def get_contours(frame):
    fgbg = cv2.createBackgroundSubtractorMOG2(history=10)
    fgmask = fgbg.apply(frame)
    _, mask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 13:
            x, y, w, h = cv2.boundingRect(cnt)
            return x, y, w, h


def get_yolo(frame):
    model = yolov5.load('yolov5/best.pt')
    outputs = model(frame)
    for detection in outputs.pred:
        x1, y1, x2, y2 = detection[:, :4][0]
        scores = detection[:, 4]
        class_label = detection[:, 5]
        return int(x1), int(y1), int(x2), int(y2)

capture = cv2.VideoCapture("data/processed/video_002.mp4")
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object for the output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter("MOG_Tracker.avi", fourcc, 30.0, (frame_width, frame_height))
trajectory_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
center = []
while capture.isOpened():
    # Read the next frame
    success, frame = capture.read()

    # Check if the frame was read successfully
    if not success:
        break

    contours = get_contours(frame)
    yolo = get_yolo(frame)

    x_sum = 0
    y_sum = 0

    # Iterate over the bounding boxes and sum the center coordinates
    print(contours)
    print(yolo)
    for box in [contours,yolo]:
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        x_sum += x_center
        y_sum += y_center

    # Calculate the mean center coordinates
    x_mean = x_sum / len(yolo + contours)
    y_mean = y_sum / len(yolo + contours)

    # The average center point can be represented as a tuple of the mean x and y coordinates
    center_point = (x_mean, y_mean)
    center.append(center_point)

capture.release()
print(center)