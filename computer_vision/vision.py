import os
import cv2 as cv
import numpy as np
from .aruco import *
from scipy.spatial.transform import Rotation as R
from .geometry import *
from .constants import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))
#os.system('cls')

def rescaleFrame(frame, scale=0.3):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

class Map:
    def __init__(self):
        # Initialilze camera video capture
        self.capture = cv.VideoCapture(0)
        # Drop the first x frames
        for _ in range(FIRST_FRAME):
            self.capture.read()
        
        # Define class attributes
        success, self.raw_frame = self.capture.read()
        self.frame = self.raw_frame.copy()
        self.robot = np.zeros((1,3))
        self.destination = np.zeros(3)
        self.found_robot = False
        self.found_destination = False
        self.obstacles = []
        
        # Get aruco in the corners
        self.map_corners = get_corner_arucos(self.frame)
        
        # Load the camera parameters from the saved file
        cv_file = cv.FileStorage(
            CAMERA_CALIBRATION_FILE, cv.FILE_STORAGE_READ) 
        self.camera_matrix = cv_file.getNode('K').mat()
        self.dist_coeffs = cv_file.getNode('D').mat()
        cv_file.release()
    
    def info(self):
        print("===== MAP INFO =====")
        if self.found_robot:
            print("ROBOT:")
            print(f"Position: {self.robot[0]}")
            print(f"Orientation: {self.robot[1]}\n")
        else:
            print("ROBOT: Not found")
        if self.found_destination:
            print("DESTINATION:")
            print(f"Position: {self.destination[0]}")
        else:
            print("DESTINATION: Not found")
        if len(self.obstacles):
            print("OBSTACLES:",len(self.obstacles))
        else:
            print("OBSTACLES: Not found")

    def flatten_scene(self,frame):
        inner_corners = np.array([
            self.map_corners['top_left'],
            self.map_corners['top_right'],
            self.map_corners['bottom_right'],
            self.map_corners['bottom_left']
        ], dtype="float32")
        
        destination_corners = np.array([
            [0,0],
            [WIDTH-1,0],
            [WIDTH-1, HEIGHT-1],
            [0, HEIGHT-1]
        ], dtype="float32")
        
        M = cv.getPerspectiveTransform(inner_corners,destination_corners)
        
        return cv.warpPerspective(frame,M,(WIDTH,HEIGHT))
    
    def update(self):
        success, self.raw_frame = self.capture.read()
        #self.frame = cv.undistort(self.frame, self.camera_matrix, self.dist_coeffs)
        # Only keep the region inside the 4 aruco codes
        self.map_corners = get_corner_arucos(self.raw_frame)
        if len(self.map_corners)==4:
            self.frame = self.flatten_scene(self.raw_frame)
        # Reset obstacles
        self.obstacles = []
        if not success:
            print("Warning: Failed to capture new frame")
    
    def set_frame(self,frame):
        self.frame = frame
    
    def find_robot(self):
        aruco_markers = get_rt_arucos(self.frame, MARKER_SIZE_ROBOT, self.camera_matrix, self.dist_coeffs)
        
        if len(aruco_markers):
            if 1 in aruco_markers.keys():
                print("Robot found")
                self.robot = aruco_markers[0]
                self.found_robot = True
            if 2 in aruco_markers.keys():
                print("Destination found")
                self.destination = aruco_markers[1]
                self.found_destination = True
    
    def detect_obstacles(self):
        # Create local copy of the frame that we will use for treatment
        frame = self.frame.copy()
        # Detect edges
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Blur the image
        frame = cv.GaussianBlur(frame, (5, 5),0)
        frame = cv.Canny(frame, 50, 150)
        # Detect contours
        contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Show edge detection
        cv.imshow("Edges", frame)
        
        # Approximate the contour to a polygon and extract corners
        
        for contour in contours:
            epsilon = 0.025 * cv.arcLength(contour, True)  # 2% of the perimeter
            approx_corners = cv.approxPolyDP(contour, epsilon, True)
        
            if len(approx_corners) >= 3:  # Ensure it's a valid polygon
                corners = approx_corners.reshape(-1, 2) # Extract the corners
                corners = geom.remove_close_points(corners) # Remove duplicates
                corners = geom.augmented_polygon(corners) # Compute augmented obstacle
                
                self.obstacles.append(corners)  # Add the obstacles to our obstacle list

        self.obstacles = [contour for contour in self.obstacles if MIN_AREA <= cv.contourArea(contour) <= MAX_AREA]

    def show(self):
        # Draw robot
        if self.found_robot:
            cv.drawContours(self.frame, self.robot[2], -1, (255,0,0), LINE_THICKNESS)
        # Draw the robot orientation
        
        # Draw destination
        if self.found_destination:
            cv.drawContours(self.frame, self.destination[2], -1, (0,255,0), LINE_THICKNESS)
        
        # Draw obstacle
        for obstacle in self.obstacles:
            if DISPLAY_OBSTACLES=="POLYGON":
                cv.polylines(self.frame, [obstacle], isClosed=True, color=(0, 0, 255), thickness=LINE_THICKNESS)
            elif DISPLAY_OBSTACLES=="POINTS":
                for point in obstacle:
                    cv.circle(self.frame,point,3,(255,0,0),LINE_THICKNESS)
            else:
                print("Error: Wrong DISPLAY_OBSTACLES value !")
            
        cv.imshow('Flatten',self.frame)
        cv.imshow('Raw',self.raw_frame)
        
    def visibility_graph():
        return []
        
    def output_navigation(self):
        # Create visibility graph
        return []
    
    def get_obstacles(self):
        return self.obstacles

    def vision_stop(self):
        self.capture.release()
    
    