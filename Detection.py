import cv2
import torch
import numpy as np
import pickle
from shapely.geometry import Point, Polygon
import json
import time
import csv
from pymongo import MongoClient
from pymongo import UpdateOne

import concurrent.futures

connection_string = "DB_LINK" # paste mongodb  connection string
client = MongoClient(connection_string)

# Replace "your_database" and "your_collection" with the desired database and collection names
database = client.IPS
collection = database["parking-space"]
distinct_ids = collection.distinct("id")

# Load YOLO model

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
class_list = model.names

model.cuda()


pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)

def publishMongoResults(parkingSpots):
    bulk_ops = []

    for data in parkingSpots:
        print(data)
        data_dict = json.loads(data)  # Convert JSON string to dictionary
        filter_criteria = {"id": data_dict["id"]}
        update_document = {"$set":data_dict}
        bulk_ops.append(UpdateOne(filter_criteria, update_document, upsert = True))

    collection.bulk_write(bulk_ops)


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        # print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.resizeWindow('RGB', 1920, 1080)
cv2.setMouseCallback('RGB', RGB)

# Load image or video
# input_path = "outputv1.mp4"
# cap = cv2.VideoCapture(input_path)
cap = cv2.VideoCapture(42)

# Load the Parking Spaces that you marked
regions = "regions.p"
with open(regions, 'rb') as f:
    parking_boxes = pickle.load(f)

# Function to check if the a point is within a Polygon
def parkingIoU(point, polygon_coordinates):
    point = Point(point)
    polygon = Polygon(polygon_coordinates)
    return point.within(polygon)


class ParkingSpace:
    _counter = 0  # Class variable to generate unique IDs

    def __init__(self, coords):
        '''
            Keep a global counter
            Create a plygon from vertices coordinates
            Find center of the polygon
            assign unique id as name
        Arguments: Parking space coordinates space coordinates
        Returns: Nothing
        '''
        # Increment the counter and assign a unique ID
        ParkingSpace._counter += 1
        self.id = ParkingSpace._counter

        # Set the provided coordinates
        self.vertices = np.array(coords,np.int32).tolist()
        polygon = Polygon(self.vertices)

        # Find the center of the polygon
        polygon_center = polygon.centroid.xy
        self.center_x, self.center_y = polygon_center[0][0], polygon_center[1][0]
        self.occupancy_stat = False

        # Set the parking space name as "p" followed by the counter
        self.name = f"p{self.id}"


psObjs=[]
#  2
# For all parking spaces in the ground truth
#   Crete ParkingSpace Object

for ps in parking_boxes:
    psObj = ParkingSpace(ps)
    psObjs.append(psObj)

# 3
# Read frames from the cam video
# Conert frames to RGB
# Perform Object detection on each frame
# From the detection results filter out cars, car boxes 
# Find the centers of car detection boxes
# Draw a circles around thoes centres
# Loop through all the Ground truth parking spaces and for each ground truth box check if any detected box lies in that ground truth box
_count = 0
time_od = []

#Updating GPS Coordinates
csv_file = 'Final GPS.csv'
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header if exists
    for row in csv_reader:
        for ps in psObjs:
            if row[0].replace(" ", "") == ps.name.replace(" ", ""):
                ps.lat = float(row[1])
                ps.long = float(row[2])
                ps.dis = float(row[3])
                break
#calibartion data import
cameraMatrix, dist = pickle.load(open("calibration.pkl", "rb"))
cameraMatrix = pickle.load(open("cameraMatrix.pkl", "rb"))
dist = pickle.load(open("dist.pkl", "rb"))

#removing distortion
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
    frame = undistort_frame(frame)

    if not ret:
        break
    
    # Convert BGR image to RGB (PIL format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame1=[frame_rgb]

    # Perform inference
    results = model(frame1)

    # Extract car bounding boxes
    car_boxes = results.xyxyn[0].cpu().numpy()

    # Filter out boxes corresponding to cars (you may need to adjust class labels)
    car_boxes = car_boxes[car_boxes[:, 5] == 0]
    # car_boxes = car_boxes[(car_boxes[:, 5] == 2) | (car_boxes[:, 5] == 7) | (car_boxes[:, 5] == 5)]
    car_boxes = car_boxes[car_boxes[:, 4] >= 0.3]

    cxBig=[]
    cyBig=[]

    occupiedSpaces=[]

    # Calculating center of car detection box
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for box in car_boxes:
        x_min, y_min, x_max, y_max, confidence, class_label = box
        label = f'{class_list[int(class_label)]} {confidence:.2f}'
        x_min, y_min, x_max, y_max = int(box[0]*x_shape), int(box[1]*y_shape), int(box[2]*x_shape), int(box[3]*y_shape)
        #cv2.putText(frame, label, (x_min,y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cx=int(x_min+x_max)//2
        cy=int(y_min+y_max)//2
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cxBig.append(cx)
        cyBig.append(cy)

    # Updating the status of parkingSpace using IoU
    for ps in psObjs:
        if len(cxBig) > 0 and len(cyBig) > 0:
            for i in range(len(cxBig)):
                point_inside=(cxBig[i],cyBig[i])
                polygon_coordinates=np.array(ps.vertices,np.int32)
                check_stat=parkingIoU(point_inside, polygon_coordinates)
                if check_stat==1:
                   ps.occupancy_stat=True
                   break
                else:
                    ps.occupancy_stat=False

    # Database Integration with no-sql
    json_objects = [json.dumps(space.__dict__) for space in psObjs]

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pool.submit(publishMongoResults, json_objects)
    
    # Drawing lines on parking-spaces
    for ps in psObjs:
        yc=ps.center_y+15
        if ps.occupancy_stat==True:
            cv2.polylines(frame,[np.array(ps.vertices,np.int32)],True,(0,0,255),1)
            cv2.putText(frame,ps.name,(int(ps.center_x),int(yc)),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        else:
            cv2.polylines(frame,[np.array(ps.vertices,np.int32)],True,(0,255,0),1)
            cv2.putText(frame,ps.name,(int(ps.center_x),int(yc)),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    
    # Display the frame with bounding boxes
    cv2.imshow('RGB', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break
    
# Release resources
pool.shutdown()
cap.release()
cv2.destroyAllWindows()
average_time_od = np.average(np.array(time_od))
print(f"Average Object Detection Time: {average_time_od}")
