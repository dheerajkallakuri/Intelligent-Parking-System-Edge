import os
import numpy as np
import cv2
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector
from matplotlib.collections import PatchCollection


points = []
prev_points = []
patches = []
total_points = []
breaker = False

class SelectFromCollection(object):
    def __init__(self, ax):
        self.canvas = ax.figure.canvas
        self.poly = PolygonSelector(ax, self.onselect)
        self.ind = []

    def onselect(self, verts):
        global points
        points = verts
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.canvas.draw_idle()

def break_loop(event):
    global breaker
    global globSelect
    global savePath
    if event.key == 'b':
        globSelect.disconnect()
        if os.path.exists(savePath):
            os.remove(savePath)

        print("data saved in "+ savePath + " file")    
        with open(savePath, 'wb') as f:
            pickle.dump(total_points, f, protocol=pickle.HIGHEST_PROTOCOL)
            exit()

def onkeypress(event):
    global points, prev_points, total_points
    if event.key == 'n': 
        pts = np.array(points, dtype=np.int32)   
        if points != prev_points and len(set(points)) == 4:
            print("Points : "+str(pts))
            patches.append(Polygon(pts))
            total_points.append(pts)
            prev_points = points

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

if __name__ == '__main__':
    

    global globSelect
    global savePath

    savePath = "regions.p"
    video_path= "parkingvideo.mp4"

    cap = cv2.VideoCapture(42)
    # cap = cv2.VideoCapture("outputv1.mp4")

    # Get the width and height of frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

    # Record video for 5 seconds
    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 5:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything and save the video
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n> Select a region in the figure by enclosing them within a quadrilateral.")
    print("> Press the 'f' key to go full screen.")
    print("> Press the 'esc' key to discard current quadrilateral.")
    print("> Try holding the 'shift' key to move all of the vertices.")
    print("> Try holding the 'ctrl' key to move a single vertex.")
    print("> After marking a quadrilateral press 'n' to save current quadrilateral and then press 'q' to start marking a new quadrilateral")
    print("> When you are done press 'b' to Exit the program\n")

    video_capture = cv2.VideoCapture(video_path)
    cnt=0
    rgb_image = None
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break
        if cnt == 5:
            rgb_image = frame[:, :, ::-1]
        cnt += 1
    video_capture.release()
  
    while True:
        fig, ax = plt.subplots()
        image = rgb_image
        frame = undistort_frame(image)
        ax.imshow(frame)
    
        p = PatchCollection(patches, alpha=0.7)
        p.set_array(10*np.ones(len(patches)))
        ax.add_collection(p)
            
        globSelect = SelectFromCollection(ax)
        bbox = plt.connect('key_press_event', onkeypress)
        break_event = plt.connect('key_press_event', break_loop)
        plt.show()
        globSelect.disconnect()
