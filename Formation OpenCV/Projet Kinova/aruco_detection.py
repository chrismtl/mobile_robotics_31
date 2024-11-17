import cv2 as cv
import numpy as np

desired_aruco_dictionary = "DICT_7X7_50"
 
# The different ArUco dictionaries built into the OpenCV library. 
ARUCO_DICT = {
  "DICT_4X4_50": cv.aruco.DICT_4X4_50,
  "DICT_4X4_100": cv.aruco.DICT_4X4_100,
  "DICT_4X4_250": cv.aruco.DICT_4X4_250,
  "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
  "DICT_5X5_50": cv.aruco.DICT_5X5_50,
  "DICT_5X5_100": cv.aruco.DICT_5X5_100,
  "DICT_5X5_250": cv.aruco.DICT_5X5_250,
  "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
  "DICT_6X6_50": cv.aruco.DICT_6X6_50,
  "DICT_6X6_100": cv.aruco.DICT_6X6_100,
  "DICT_6X6_250": cv.aruco.DICT_6X6_250,
  "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
  "DICT_7X7_50": cv.aruco.DICT_7X7_50,
  "DICT_7X7_100": cv.aruco.DICT_7X7_100,
  "DICT_7X7_250": cv.aruco.DICT_7X7_250,
  "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
  "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL
}
  
def main():
  """
  Main method of the program.
  """
  # Load the ArUco dictionary
  print("[INFO] detecting '{}' markers...".format(
    desired_aruco_dictionary))
  aruco_dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary])
  aruco_parameters =  cv.aruco.DetectorParameters()
  detector = cv.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
   
  # Start the video stream
  capture = cv.VideoCapture(1)
   
  while(True):
  
    # Capture frame-by-frame
    # This method returns True/False as well
    # as the video frame.
    ret, frame = capture.read()  
     
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
         
        # Draw the bounding box of the ArUco detection
        cv.line(frame, top_left, top_right, (0, 255, 0), 2)
        cv.line(frame, top_right, bottom_right, (0, 255, 0), 2)
        cv.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
        cv.line(frame, bottom_left, top_left, (0, 255, 0), 2)
         
        # Calculate and draw the center of the ArUco marker
        center_x = int((top_left[0] + bottom_right[0]) / 2.0)
        center_y = int((top_left[1] + bottom_right[1]) / 2.0)
        cv.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
         
        # Draw the ArUco marker ID on the video frame
        # The ID is always located at the top_left of the ArUco marker
        cv.putText(frame, str(marker_id), 
          (top_left[0], top_left[1] - 15),
          cv.FONT_HERSHEY_SIMPLEX,
          0.5, (0, 255, 0), 2)
  
    # Display the resulting frame
    cv.imshow('frame',frame)
          
    # If "q" is pressed on the keyboard, 
    # exit this loop
    if cv.waitKey(1) & 0xFF == ord('q'):
      break
  
  # Close down the video stream
  capture.release()
  cv.destroyAllWindows()
   
if __name__ == '__main__':
  print(__doc__)
  main()