import cv2
import numpy as np
import time

# Resizing frame or image
def rescaleFrame(image, height_scale=0.75, width_scale=0.8):
    # For images, videos and live videos
    height = int(image.shape[0] * height_scale)
    width = int(image.shape[1] * width_scale)
    dimensions = (width, height)

    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

def color_segmentation(image):
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define ranges for red color in HSV
    lower_red1 = np.array([0, 50, 250])
    upper_red1 = np.array([10, 100, 255])
    lower_red2 = np.array([170, 50, 250])
    upper_red2 = np.array([180, 100, 255])

    # Define ranges for blue color in HSV
    lower_blue1 = np.array([100, 20, 225])
    upper_blue1 = np.array([110, 30, 255])
    lower_blue2 = np.array([100, 100, 225])
    upper_blue2 = np.array([110, 130, 255])

    # Define ranges for green color in HSV
    lower_green = np.array([40, 100, 150])
    upper_green = np.array([75, 255, 255])

    # Define ranges for yellow color in HSV
    lower_yellow = np.array([109, 0, 225])
    upper_yellow = np.array([111, 7, 255])
    
    # Segment colors
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask1 = cv2.inRange(hsv_image, lower_blue1, upper_blue1)
    blue_mask2 = cv2.inRange(hsv_image, lower_blue2, upper_blue2)
    blue_mask = cv2.bitwise_or(blue_mask1, blue_mask2)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Clean up the masks using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    detected_objects = []
    masks = [(red_mask, 'red'), (blue_mask, 'blue'), (yellow_mask, 'yellow'), (green_mask, 'green')]
    
    for mask, color in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                centroid_x = int(moments['m10'] / moments['m00'])
                centroid_y = int(moments['m01'] / moments['m00'])
                detected_objects.append((color, (centroid_x, centroid_y)))
                cv2.circle(image, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
                cv2.putText(image, color, (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return detected_objects

if __name__ == "__main__":
    image = cv2.imread("Photos/Green Background/bloc_4.jpeg")
    if image is None:
        print("Could not open or find the image!")
    else:
        objects = color_segmentation(image)
        for obj in objects:
            print(f"Detected {obj[0]} shape at {obj[1]}")
        image_resized = rescaleFrame(image)
        cv2.imshow("Detected Shapes", image_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
