from typing import List
import numpy as np
import cv2
import torch
import yolov5


class KalmanFilter:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, id: int = None) -> None:
        self.state = np.array([[x1], [y1], [x2], [y2]])
        self.transition_matrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.process_noise_covariance = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.measurement_noise_covariance = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.identity_matrix = np.eye(4)
        self.measurement_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kalman_gain = np.zeros((4, 4))
        self.error_covariance = np.zeros((4, 4))
        self.id = id

    def update(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.state = self.transition_matrix @ self.state
        self.error_covariance = (
                                        self.transition_matrix @ self.error_covariance) @ self.transition_matrix.T + self.process_noise_covariance
        self.kalman_gain = (self.error_covariance @ self.measurement_matrix.T) @ np.linalg.inv(
            self.measurement_matrix @ self.error_covariance @ self.measurement_matrix.T + self.measurement_noise_covariance)
        self.state = self.state + self.kalman_gain @ (
                np.array([[x1], [y1], [x2], [y2]]) - self.measurement_matrix @ self.state)
        self.error_covariance = (
                                        self.identity_matrix - self.kalman_gain @ self.measurement_matrix) @ self.error_covariance


def yolov5_kalman_filter(video_path: str, model: torch.nn.Module, kalman_filters: List[KalmanFilter]) -> None:
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        outputs = model(frame)
        for detection in outputs.pred:
            try:
                x1, y1, x2, y2 = detection[:, :4][0]
            except:
                print()
            scores = detection[:, 4]
            class_label = detection[:, 5]
            found = False
            for kf in kalman_filters:
                if kf.id == class_label:
                    kf.update(x1, y1, x2, y2)
                    x1, y1, x2, y2 = kf.state
                    found = True
                    break
            if not found:
                kf = KalmanFilter(x1, y1, x2, y2, id=class_label)
                kalman_filters.append(kf)
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Draw the label on the image
            # frame = cv2.putText(frame, class_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
            #                    2)


        # Close the video file
        video_capture.release()
        cv2.destroyAllWindows()

# Load a YOLOv5 model
model = yolov5.load('yolov5/best.pt')

# Post-process the detections from the YOLOv5 model using Kalman filters
yolov5_kalman_filter("data/processed/video_000.mp4", model, [])
