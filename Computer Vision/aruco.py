import cv2 as cv
import numpy as np

aruco_dictionary = cv.aruco.DICT_7X7_50
  
def get_arucos(frame):
    """
    Main method of the program.
    """
    # Load the ArUco dictionary
    print("ARUCO: '{}'".format(aruco_dictionary))
    aruco_dictionary = cv.aruco.getPredefinedDictionary(aruco_dictionary)
    aruco_parameters =  cv.aruco.DetectorParameters()
   
    # Detect ArUco markers in the video frame
    (corners, ids, rejected) = cv.aruco.detectMarkers(
        frame, aruco_dictionary, parameters=aruco_parameters)
        
    # Check that at least one ArUco marker was detected
    if len(corners) > 0:
        # Flatten the ArUco IDs list
        ids = ids.flatten()
        
        # Loop over the detected ArUco corners
        for (marker_corner, marker_id) in zip(corners, ids):
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
    return [corners,ids]

    