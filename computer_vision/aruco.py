import cv2 as cv
import numpy as np
from .geometry import *
from constants import *

# CONSTANTS
ARUCO_DICTIONARY_CORNER = cv.aruco.DICT_4X4_50

def get_rt_arucos(frame, marker_size, camera_matrix, dist_coeffs):
    """
    Scan the Aruco Tags of the Robot and the destination
    
    Args:
        frame: image snapshot of the camera
        marker_size (int): Size of the aruco marker in meters
        camera_matrix: Camera calibration matrix
        dist_coeffs: Camera calibration coefficients

    Returns:
        if markers found:
            aruco_markers = { ROBOT_TAG_ID: [center_x, center_y, angle],
                              DESTINATION_TAG_ID: [center_x, center_y, corners]}
        else:
            aruco_markers = { ROBOT_TAG_ID: [center_x, center_y, angle],
                              DESTINATION_TAG_ID: [center_x, center_y, corners]}
    """
    # Load the ArUco dictionary
    aruco_dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_CORNER)
    aruco_parameters =  cv.aruco.DetectorParameters()
   
    # Detect ArUco markers in the video frame
    (corners, ids, rejected) = cv.aruco.detectMarkers(
        frame, aruco_dictionary, parameters=aruco_parameters)
    
    # Set default ids to None
    center_4 = [None]*3
    center_5 = [None]*3
    
    # Check that at least one ArUco marker was detected
    if ids is not None:
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
        for (marker_corner, marker_id) in zip(corners, ids):
            # Draw marker axes
            if CV_DRAW: cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.04)
            
            # Extract the marker corners
            treated_corners = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = treated_corners
                
            # Convert the (x,y) coordinate pairs to integers
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            treated_corners = (top_left, top_right, bottom_right, bottom_left)
                        
            # Calculate the center of the ArUco marker
            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
                        
            # Store the rotation information
            rotation_matrix = np.eye(4)
            rotation_matrix = cv.Rodrigues(np.array(rvecs[i][0]))[0]
            (rx,ry,rz) = get_rotations(rotation_matrix)
            
            if marker_id==AT_ROBOT:
                center_4 = [center_x,center_y,np.radians(rz)]
            if marker_id==AT_DESTINATION:
                center_5 = [center_x,center_y,treated_corners]
            
            i+=1
        
    return {4:center_4,
            5:center_5}

def get_corner_arucos(frame):
    """
    Find the four aruco tags of the map corners

    Args:
        frame : image camera snapshot

    Returns:
        if markers found:
            aruco_markers = { valid: True,
                              *corner*: *bottom_right corner pixel coordinates*}
        else:
            aruco_markers = { valid: False,
                              ids: *ids found*}

    """
    # Load the ArUco dictionary
    aruco_dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_CORNER)
    aruco_parameters =  cv.aruco.DetectorParameters()
   
    # Detect ArUco markers in the video frame
    (corners, ids, rejected) = cv.aruco.detectMarkers(
        frame, aruco_dictionary, parameters=aruco_parameters)
    
    # Check that all four ArUco markers were detected
    if ids is not None:
        if (AT_TOP_LEFT in ids) and (AT_BOTTOM_LEFT in ids) and (AT_BOTTOM_RIGHT in ids) and (AT_TOP_RIGHT in ids):
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
                
                if marker_id==AT_TOP_LEFT:
                    aruco_markers['top_left'] = bottom_right
                if marker_id==AT_BOTTOM_LEFT:
                    aruco_markers['bottom_left'] = top_right
                if marker_id==AT_BOTTOM_RIGHT:
                    aruco_markers['bottom_right'] = top_left
                if marker_id==AT_TOP_RIGHT:
                    aruco_markers['top_right'] = bottom_left
            aruco_markers['valid'] = True
            return aruco_markers
    return {'valid':False,
            'ids':ids}

    