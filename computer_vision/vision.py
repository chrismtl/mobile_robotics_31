import cv2 as cv
import numpy as np
from .aruco import *
from .geometry import *
from constants import *

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
        self.success = True
        self.raw_frame = None
        self.frame = None
        self.robot = 3*[None]
        self.destination = 3*[None]
        self.map_corners = {}
        self.found_corners = False
        self.found_robot = False
        self.found_destination = False
        self.obstacles = []
        self.edges = None
        self.obstacles_lines = []
        self.target_lines = []
        self.pose_est = 3*[None]
        
        # Load the camera parameters from the saved file
        try:
            cv_file = cv.FileStorage(
                CAMERA_CALIBRATION_FILE, cv.FILE_STORAGE_READ) 
            self.camera_matrix = cv_file.getNode('K').mat()
            self.dist_coeffs = cv_file.getNode('D').mat()
            cv_file.release()
        except Exception as e:
            if P_VISION:
                print("ERROR: Could not read camera calibration file")
                print(f"---\n{e}\n---")
            self.success = False
    
    def info(self):
        # Print method to display essential info of the current Map instance
        print(P_INFO)
        print(f"* FOUND CORNERS: {self.found_corners}\n")
        
        print(f"* FOUND ROBOT: {self.found_robot}")
        if self.found_robot:
            print(f"Position: {self.robot[0:2]}")
            print(f"Orientation: {self.robot[2]}\n")
            
        print(f"* FOUND DESTINATION: {self.found_destination}")
        if self.found_destination:
            print(f"Position: {self.destination[0:2]}\n")
            
        print("* OBSTACLES:",len(self.obstacles))
        print(P_END)
    
    def snap(self):
        # Capture a frame from the camera and if possible crop the Region of Interest
        self.success, self.raw_frame = self.capture.read()
        if not(self.success) and P_VISION:
            print("ERROR: Could not read image from camera")
        if self.found_corners:
            self.flatten_scene()
    
    def find_corners(self):
        # Get the four aruco in the corners of the captured frame
        self.map_corners = get_corner_arucos(self.raw_frame)

        # Check if it found the 4 corners
        if self.map_corners["valid"]:
            self.found_corners = True
            self.flatten_scene()
        else:
            self.found_corners = False
            self.frame = self.raw_frame.copy()
            if P_VISION:
                print(f"WARNING: Detected aruco corners \n{self.map_corners['ids']}")

    def flatten_scene(self):
        # From the four detected aruco corners, crop out and flatten our region of interest using opencv
        inner_corners = np.array([
            self.map_corners['top_left'],
            self.map_corners['top_right'],
            self.map_corners['bottom_right'],
            self.map_corners['bottom_left']
        ], dtype="float32")

        destination_corners = np.array([
            [0,0],
            [SCREEN_WIDTH-1,0],
            [SCREEN_WIDTH-1, SCREEN_HEIGHT-1],
            [0, SCREEN_HEIGHT-1]
        ], dtype="float32")
        
        try:
            M = cv.getPerspectiveTransform(inner_corners,destination_corners)
            self.frame = cv.warpPerspective(self.raw_frame,M,(SCREEN_WIDTH,SCREEN_HEIGHT))
        except Exception as e:
            if P_VISION: print("ERROR: warpPerspective")
            self.success = False
    
    def update(self, setup=False):
        # Capture a new frame and update the class attributes
        self.snap()
        if setup:
            self.find_corners()
            if self.found_corners:
                self.find_thymio_destination()
                self.pose_est = self.robot.copy()
                self.detect_global_obstacles()
        else:
            self.find_thymio_destination()

        self.show()
    
    def find_thymio_destination(self):
        # Scan the aruco tag of the robot and the destination
        aruco_markers = get_rt_arucos(self.frame, MARKER_SIZE_ROBOT, self.camera_matrix, self.dist_coeffs)
        
        if not None in aruco_markers[AT_ROBOT]:
            self.robot = aruco_markers[AT_ROBOT]
            self.found_robot = True
        else:
            if P_VISION: print("WARNING: Robot not found")
            self.found_robot = False
        
        if not None in aruco_markers[AT_DESTINATION]:
            self.destination = aruco_markers[AT_DESTINATION]
            self.found_destination = True
        elif P_VISION:
            print("WARNING: Destination not found")
    
    def detect_global_obstacles(self):
        # Clear previous obstacles
        self.obstacles = []
        # Create local copy of the frame that we will use for treatment
        frame = self.frame.copy()
        # Convert to grayscale
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Blur the image
        frame = cv.GaussianBlur(frame, (5, 5),0)
        # Detect edges
        frame = cv.Canny(frame, 50, 150)
        # Detect contours
        contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # Store edge detection
        self.edges = frame
        # Remove area out of range obstacles
        contours = [contour for contour in contours if MIN_AREA <= cv.contourArea(contour) <= MAX_AREA]
        # Approximate the contour to a polygon and extract corners
        for contour in contours:
            epsilon = 0.02 * cv.arcLength(contour, True)  # 2% of the perimeter
            approx_corners = cv.approxPolyDP(contour, epsilon, True)

            if len(approx_corners) >= 3:  # Ensure it's a valid polygon
                corners = approx_corners.reshape(-1, 2) # Extract the corners
                corners = remove_close_points(corners) # Remove duplicates
                corners = augment_corners(corners) # Compute augmented obstacle
                valid_obstacle = not on_points(corners,
                                                [self.found_robot,self.found_destination],
                                                [self.robot[0:2],self.destination[0:2]],
                                                [MIN_DIST_TO_ROBOT, MIN_DIST_TO_DESTINATION])
                if valid_obstacle:   # Check if it is not the robot or on the destination
                    self.obstacles.append(corners)  # Add the obstacles to our obstacle list

    def show(self):
        # Show the different vision windows
        # Draw frame
        frame = self.frame.copy()
        raw_frame = self.raw_frame.copy()
        if self.found_corners:
            #Draw reference frame
            cv.line(frame, (0,0), (50,0), X_AXIS_COLOR, 6)
            cv.line(frame, (0,0), (0,50), Y_AXIS_COLOR, 6)

            # Draw robot
            robot_x = self.robot[0]
            robot_y = self.robot[1]
            robot_angle = self.robot[2]
            robot_color = ROBOT_COLOR
            if not self.found_robot:
                robot_x = self.pose_est[0]
                robot_y = self.pose_est[1]
                robot_angle = self.pose_est[2]
                #robot_color = KALMAN_COLOR
            
            end_x = int(robot_x + 75 * np.cos(robot_angle))
            end_y = int(robot_y - 75 * -np.sin(robot_angle))
            end_point = (end_x, end_y)
            cv.arrowedLine(frame, (int(self.robot[0]),int(self.robot[1])), end_point, ROBOT_ARROW_COLOR, D_ARROW_LINE_WIDTH, tipLength=0.2)
            cv.circle(frame, self.robot[0:2],D_ROBOT_CIRCLE_RADIUS,robot_color,-1)

            # Draw robot's estimated pose
            end_x_est = int(self.pose_est[0] + 75 * np.cos(self.pose_est[2]))
            end_y_est = int(self.pose_est[1] - 75 * -np.sin(self.pose_est[2]))
            end_point_est = (end_x_est, end_y_est)
            
            cv.circle(frame, (int(self.pose_est[0]),int(self.pose_est[1])),D_ROBOT_CIRCLE_RADIUS,(255,0,0),-1)
            cv.arrowedLine(frame, (int(self.pose_est[0]),int(self.pose_est[1])), end_point_est, (138,43,226), D_ARROW_LINE_WIDTH, tipLength=0.2)               
            
            # Draw destination
            if self.found_destination:
                tl,tr,br,bl = self.destination[2]
                cv.line(frame, tl, br, DESTINATION_COLOR, DESTINATION_CROSS_WIDTH)
                cv.line(frame, tr, bl, DESTINATION_COLOR, DESTINATION_CROSS_WIDTH)
            
            # Draw obstacle
            for obstacle in self.obstacles:
                if DISPLAY_OBSTACLES=="POLYGON":
                    cv.polylines(frame, [obstacle], isClosed=True, color=(0, 0, 255), thickness=LINE_THICKNESS)
                elif DISPLAY_OBSTACLES=="POINTS":
                    for point in obstacle:
                        cv.circle(frame,point,3,(255,0,0),LINE_THICKNESS)
                elif P_VISION:
                    print("Error: Wrong DISPLAY_OBSTACLES value !")
            
            if SHOW_ALL_PATHS:
                #Draw lines between obstacles
                for obstacles_line in self.obstacles_lines:
                    cv.line(frame, obstacles_line[0:2], obstacles_line[2:4], (0,0,0), 1)

            #Draw shortest path
            for i in range(1, len(self.target_lines)):
                cv.line(frame, self.target_lines[i-1], self.target_lines[i], PATH_COLOR, 3)

            # Draw raw frame plus green region of interest
            cv.line(raw_frame, self.map_corners['top_left'], self.map_corners['top_right'], (0, 255, 0), 10)
            cv.line(raw_frame, self.map_corners['top_right'], self.map_corners['bottom_right'], (0, 255, 0), 10)
            cv.line(raw_frame, self.map_corners['bottom_right'], self.map_corners['bottom_left'], (0, 255, 0), 10)
            cv.line(raw_frame, self.map_corners['bottom_left'], self.map_corners['top_left'], (0, 255, 0), 10)

            cv.imshow('Frame', frame)
            cv.imshow('Edges', self.edges)

        cv.imshow('Raw frame', raw_frame)
        
    
    def set_raw_frame(self,frame):
        self.raw_frame = frame
    
    def set_frame(self,frame):
        self.frame = frame
    
    def get_obstacles(self):
        return self.obstacles

    def __del__(self):
        if P_VISION: print(P_STOP)
        self.capture.release()
        cv.destroyAllWindows()
    
    