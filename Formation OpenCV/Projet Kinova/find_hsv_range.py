import cv2
import numpy as np

# Resizing frame or image
def rescaleFrame(image, height_scale=0.75, width_scale=0.8):
    # For images, videos and live videos
    height = int(image.shape[0] * height_scale)
    width = int(image.shape[1] * width_scale)
    dimensions = (width, height)

    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    

# Function to get HSV values from a point clicked on the image
def get_hsv_values(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv_image[y, x]
        print(f"HSV value at ({x}, {y}): {pixel}")

# Load the image and convert to HSV
image = cv2.imread("Photos/Green Background/bloc_4.jpeg")
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Display the image and set the callback function
image_resized = rescaleFrame(image)
cv2.imshow('Image', image_resized)
cv2.setMouseCallback('Image', get_hsv_values)
cv2.waitKey(0)
cv2.destroyAllWindows()