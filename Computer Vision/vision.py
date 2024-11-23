import os
import cv2 as cv
import numpy as np
import aruco
from scipy.spatial.transform import Rotation as R
import geometry as geom

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.system('cls')

# === CONSTANTS ===
MARKER_SIZE_ROBOT = 0.1     # Side length of the ArUco marker in meters 
FIRST_FRAME = 50            # First frame to analyse
ROBOT_RADIUS = 50

def rescaleFrame(frame, scale=0.75):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

class Map:
    def __init__(self):
        self.capture = cv.VideoCapture(0)
        for _ in range(FIRST_FRAME):
            self.capture.read()
            
        success, self.frame = self.capture.read()
        self.robot = np.zeros((1,3))
        self.destination = np.zeros(3)
        self.found_robot = False
        self.found_destination = False
        self.obstacles = []

    def info(self):
        print("===== MAP INFO =====")
        if self.found_robot:
            print("ROBOT:")
            print(f"Position: {self.robot[0]}")
            print(f"Orientation: {self.robot[1]}\n")
        if self.found_destination:
            print("DESTINATION:")
            print(f"Position: {self.destination[0]}")

    def update(self):
        success, self.frame = self.capture.read()
    
    def scan_arucos(self):
        aruco_markers = aruco.get_arucos(self.frame, MARKER_SIZE_ROBOT)
        
        if len(aruco_markers):
            # Debug
            if 0 in aruco_markers.keys():
                print("Robot found")
                self.robot = aruco_markers[0]
                self.found_robot = True
            if 1 in aruco_markers.keys():
                print("Destination found")
                self.destination = aruco_markers[1]
                self.found_destination = True
    
    def detect_obstacles(self):
        # Create local copy of the frame that we will use for treatment
        frame = self.frame.copy()
        
        # Mask out the robot and the destination
        if self.found_robot:
            mask_robot = [geom.augmented_polygon(self.robot[2][0], 6)]
            cv.fillPoly(frame, mask_robot, (255, 255, 255))
            
        if self.found_destination:
            mask_destination = [geom.augmented_polygon(self.destination[2][0], 6)]
            cv.fillPoly(frame, mask_destination, (255, 255, 255))
        
        # Detect edges
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.Canny(frame, 50, 150)
        
        # Detect contours
        contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        # Show edge detection
        #cv.imshow("Edges", frame)

        # Approximate the contour to a polygon and extract corners
        for contour in contours:
            epsilon = 0.025 * cv.arcLength(contour, True)  # 2% of the perimeter
            approx_corners = cv.approxPolyDP(contour, epsilon, True)
        
            if len(approx_corners) >= 3:  # Ensure it's a valid polygon
                corners = approx_corners.reshape(-1, 2) # Extract the corners
                corners = geom.remove_close_points(corners) # Remove duplicates
                corners = geom.augmented_polygon(corners,ROBOT_RADIUS) # Compute augmented obstacle
                self.obstacles.append(corners)  # Add the obstacles to our obstacle list


    def draw(self):
        # Draw robot
        if self.found_robot:
            print("Printing robot")
            cv.drawContours(self.frame, self.robot[2], -1, (255,0,0), 5)
        # Draw the robot orientation
        
        # Draw destination
        if self.found_destination:
            cv.drawContours(self.frame, self.destination[2], -1, (0,255,0), 5)
        
        # Draw obstacle
        for obstacle in self.obstacles:
            cv.polylines(self.frame, [obstacle], isClosed=True, color=(0, 0, 255), thickness=5)

        cv.imshow('Camera',self.frame)
        
    def visibility_graph():
        return []
        
    def output_navigation(self):
        # Create visibility graph
        
        return []

    def vision_stop(self):
        self.capture.release()
   
if __name__ == '__main__':
    os.system('cls')
       
    map = Map()
    
    while True:
        map.update()
        map.scan_arucos()
        map.detect_obstacles()   
        map.draw()
        if cv.waitKey(10) != -1:
            print("== Key pressed ===")
            break
        
    map.info()
    map.vision_stop()
    cv.destroyAllWindows()
    
    