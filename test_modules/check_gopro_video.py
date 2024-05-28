# import the opencv library 
import cv2 
import pickle

# define a video capture object 
#ls -al /dev/video* (to check id of the webcam)
#sudo systemctl start gopro_webcam.service
vid = cv2.VideoCapture(42) 

width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width)
print(height)

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

while(True): 
	
	# Capture the video frame 
	# by frame 
	ret, frame = vid.read() 
	frame = undistort_frame(frame)

	# Display the resulting frame 
	cv2.imshow('frame', frame) 
	
	# the 'q' button is set as the 
	# quitting button you may use any 
	# desired button of your choice 
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows
cv2.destroyAllWindows()