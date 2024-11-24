import cv2 as cv
import numpy as np
from . import geometry as geom
from operator import itemgetter
from .constants import *

# CONSTANTS
ARUCO_DICTIONARY = cv.aruco.DICT_4X4_50

def get_arucos(frame, marker_size, camera_matrix, dist_coeffs):
    
    # Load the ArUco dictionary
    aruco_dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    aruco_parameters =  cv.aruco.DetectorParameters()
   
    # Detect ArUco markers in the video frame
    (corners, ids, rejected) = cv.aruco.detectMarkers(
        frame, aruco_dictionary, parameters=aruco_parameters)
    # Check that at least one ArUco marker was detected
    if len(corners) > 0:
        # Draw a square around detected markers in the video frame
        if CV_DRAW: cv.aruco.drawDetectedMarkers(frame, corners, ids)
        # Get the rotation and translation vectors
        rvecs, tvecs, obj_points = cv.aruco.estimatePoseSingleMarkers(
            corners,
            marker_size,
            camera_matrix,
            dist_coeffs)
        # Flatten the ArUco IDs list
        ids = ids.flatten()
        # Loop over the detected ArUco corners
        i = 0
        aruco_markers = {}
        for (marker_corner, marker_id) in zip(corners, ids):
            # Draw marker axes
            if CV_DRAW: cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)
            
            # Extract the marker corners
            treated_corners = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = treated_corners
                
            # Convert the (x,y) coordinate pairs to integers
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))            
               
            # Calculate the center of the ArUco marker
            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
                        
            # Store the rotation information
            rotation_matrix = np.eye(4)
            rotation_matrix = cv.Rodrigues(np.array(rvecs[i][0]))[0]
            (rx,ry,rz) = geom.get_rotations_chat(rotation_matrix)
            aruco_markers[marker_id] = [int(marker_id),
                                        [center_x,center_y],
                                        [rx,ry,rz],
                                        np.round(marker_corner).astype(np.int32)]
            
            i+=1
            
        return aruco_markers
    return {}

def get_corner_arucos(frame):
    # Load the ArUco dictionary
    aruco_dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    aruco_parameters =  cv.aruco.DetectorParameters()
   
    # Detect ArUco markers in the video frame
    (corners, ids, rejected) = cv.aruco.detectMarkers(
        frame, aruco_dictionary, parameters=aruco_parameters)
    
    cv.aruco.drawDetectedMarkers(frame, corners, ids)
    # Check that all four ArUco markers were detected
    if len(ids)==4:
        # Flatten the ArUco IDs list
        ids = ids.flatten()
        # Loop over the detected ArUco corners
        aruco_markers = {}
        for (marker_corner, marker_id) in zip(corners, ids):
            # Extract the marker corners
            (top_left, top_right, bottom_right, bottom_left) = marker_corner.reshape((4, 2))
            
            # Convert the (x,y) coordinate pairs to integers
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            
            if top_left[0] < 300 and top_left[1] < 300:
                aruco_markers['top_left'] = bottom_right
            if top_left[0] > 300 and top_left[1] < 300:
                aruco_markers['top_right'] = bottom_right
            if top_left[0] > 300 and top_left[1] > 300:
                aruco_markers['bottom_right'] = bottom_right
            if top_left[0] < 300 and top_left[1] > 300:
                aruco_markers['bottom_left'] = bottom_right

        return aruco_markers
    print("Error: could not find the 4 Aruco corners")
    return {}

    