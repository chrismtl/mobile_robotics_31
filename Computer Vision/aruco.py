import cv2 as cv
import numpy as np

# Calibration parameters yaml file
camera_calibration_parameters_filename = 'calibration_chessboard.yaml'
  
def get_arucos(frame, marker_size):
    # Load the camera parameters from the saved file
    cv_file = cv.FileStorage(
        camera_calibration_parameters_filename, cv.FILE_STORAGE_READ) 
    camera_matrix = cv_file.getNode('K').mat()
    dist_coeffs = cv_file.getNode('D').mat()
    cv_file.release()
    
    aruco_dictionary = cv.aruco.DICT_7X7_50
    # Load the ArUco dictionary
    aruco_dictionary = cv.aruco.getPredefinedDictionary(aruco_dictionary)
    aruco_parameters =  cv.aruco.DetectorParameters()
   
    # Detect ArUco markers in the video frame
    (corners, ids, rejected) = cv.aruco.detectMarkers(
        frame, aruco_dictionary, parameters=aruco_parameters)
        
    # Check that at least one ArUco marker was detected
    if len(corners) > 0:
        # Draw a square around detected markers in the video frame
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
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
            cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)
            
            # Extract the marker corners
            corners = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners
                
            # Convert the (x,y) coordinate pairs to integers
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
               
            # Calculate the center of the ArUco marker
            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
            aruco_markers[marker_id] = [center_x,
                                        center_y,
                                        top_left,
                                        top_right,
                                        bottom_right,
                                        bottom_left]
            i+=1
        return aruco_markers
    return {}

    