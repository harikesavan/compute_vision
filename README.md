# cv_lab_aos

This project will deal with the task of muliple-object tracking based on occluded aerial thermal images.

More information will follow!


### YoloV5 Training + Validating

All experiments were done using code from https://github.com/ultralytics/yolov5. 
To start training and validation we used the command line script from that repository:

    python train.py --data custom_data.yaml --epochs 300 --weights yolov5n.pt
    --cfg yolov5n.yaml --batch-size 16 --img-size 640 --save-txt

after model has been validated, we receive the experiment output containing the model metrics, image samples, detection coordinates and sample videos,
which are stored in ./code/yolov5/runs folder.

### StrongSORT Tracking

StrongSORT tracking was done using the following GitHub repository: https://github.com/mikel-brostrom/yolov8_tracking. 
It works the same way as YoloV5 repo we used, by sending commands in the CLI we perform tracking using our pretrained YoloV5 network weights.
Example usage: 

    python track.py --yolo-weights ./weights/best_weights_v5n.pt --tracking-method strongsort 
    --conf-thres 0.3 --iou-thres 0.05 --img-size 640 --save-txt --save-vid

### Converting text files output of tracker

Tracking code outputs a .txt file for each video it tracked,
therefore we can use this to get the center points for each tracked video (using ./code/convert_txt.py) 
and draw a full path of the tracked objects (using ./code/draw_trace.py).
