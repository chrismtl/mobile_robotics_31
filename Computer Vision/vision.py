import os
import cv2 as cv
import numpy as np
import aruco
from scipy.spatial.transform import Rotation as R
import keyboard

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === CONSTANTS ===
MARKER_SIZE_ROBOT = 0.1     # Side length of the ArUco marker in meters 
FIRST_FRAME = 50            # First frame to analyse

def vision_start():
    # Start the video stream
    capture = cv.VideoCapture(0)
    
    # Drop the first 50 frame
    for _ in range(FIRST_FRAME):
        capture.read()
    
    # Scan Robot Aruco
    ret, frame = capture.read()
    markers = aruco.get_arucos(frame, MARKER_SIZE_ROBOT)
    cv.imshow('frame',frame)
    
    # Scan Map Aruco
    
    cv.waitKey(1)
    capture.release()
    cv.destroyAllWindows()
   
if __name__ == '__main__':
    os.system('cls')
    vision_start()