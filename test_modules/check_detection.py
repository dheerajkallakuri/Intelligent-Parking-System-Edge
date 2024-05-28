mport torch
import numpy as np
import cv2
import pickle
from time import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
classes = model.names

# model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
# classes = model.module.names if hasattr(model, 'module') else model.names

# Open video file or stream
input_path = "parkingvideo.mp4"
cap = cv2.VideoCapture(input_path)

# Load camera calibration data
cameraMatrix, dist = pickle.load(open("calibration.pkl", "rb"))

def undistort_frame(frame):
    h,  w = frame.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
    # Undistort
    dst = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Remove distortion from frame
    frame = undistort_frame(frame)
    # frame = cv2.resize(frame, (640,640))
    frame1=[frame]
    results = model(frame1)
    print(results)

    car_boxes = results.xyxyn[0].cpu().numpy()

    # Filter out boxes corresponding to cars
    car_boxes = car_boxes[(car_boxes[:, 5] == 0)]
    # car_boxes = car_boxes[(car_boxes[:, 5] == 2) | (car_boxes[:, 5] == 7) | (car_boxes[:, 5] == 5)]
    car_boxes = car_boxes[car_boxes[:, 4] >= 0.3]
    
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    # Draw bounding boxes on frame
    for box in car_boxes:
        x_min, y_min, x_max, y_max, confidence, class_label = box
        label = f'{classes[int(class_label)]} {confidence:.2f}'
        x_min, y_min, x_max, y_max = int(box[0]*x_shape), int(box[1]*y_shape), int(box[2]*x_shape), int(box[3]*y_shape)
        # cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), (0, 255, 0), 2)
        cv2.putText(frame, label, (x_min,y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cx=int(x_min+x_max)//2
        cy=int(y_min+y_max)//2
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # Display the frame with bounding boxes
    cv2.imshow('Object Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release resources
cap.release()
cv2.destroyAllWindows()