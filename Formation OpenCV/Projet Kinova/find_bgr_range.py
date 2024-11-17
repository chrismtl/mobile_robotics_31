import cv2
import numpy as np

# Resizing frame or image
def rescaleFrame(image, height_scale=0.75, width_scale=0.8):
    # For images, videos and live videos
    height = int(image.shape[0] * height_scale)
    width = int(image.shape[1] * width_scale)
    dimensions = (width, height)

    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

def adjust_saturation(image, scale):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], scale)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Function to get HSV values from a point clicked on the image
def get_bgr_values(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_saturated[y, x]
        print(f"BGR value at ({x}, {y}): {pixel}")

# Load the image
image = cv2.imread("Photos/Green Background/bloc_4.jpeg")

# Display the image and set the callback function
image_resized = rescaleFrame(image)
image_saturated = adjust_saturation(image_resized, 2) #increase the saturation by 50%
cv2.imshow('Image', image_saturated)
cv2.setMouseCallback('Image', get_bgr_values)
cv2.waitKey(0)
cv2.destroyAllWindows()