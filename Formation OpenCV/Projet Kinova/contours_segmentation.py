import cv2 as cv 
import numpy as np

# Resizing frame or image
def rescaleFrame(image, height_scale=0.75, width_scale=0.8):
    # For images, videos and live videos
    height = int(image.shape[0] * height_scale)
    width = int(image.shape[1] * width_scale)
    dimensions = (width, height)

    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

# Function to check if a point is inside the area
def is_point_in_area(area, x, y):
    if area[y, x] != 0:
        return True
    else:
        return False

# Function to calculate centroids with sensitivity parameter
def calculate_centroids(image, min_area_threshold, target_area):
    contours, hierarchies = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) #possible to pass thresh instead of canny as first argument
    #cv.RETR_LIST returns all the contours of the image, cv.RETR_EXTERNAL returns all the external contours and cv.RETR_TREE return all the hierarchal (?) contours
    #cv.CHAIN_APPROX_NONE returns all the contours, cv.CHAIN_APPROX_SIMPLE compresses all the contours returned (ex : a line is compressed into two points). cv.CHAIN_APPROX_SIMPLE is more time-efficient

    centroids = []
    count = 0
    for contour in contours:
        # Calculate area of contour
        area = cv.contourArea(contour)
        if area > min_area_threshold:
            # Calculate centroid using moments
            M = cv.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if is_point_in_area(target_area, cx, cy):
                    centroids.append((cx, cy))
                    count += 1
                    cv.drawContours(img_resized, contour, -1, (0, 0, 255), 2) #drawing the contours found in a blank image

    print(f'{count} contour(s) found !')
    return centroids

# Reading Images
img = cv.imread('Photos/Brown Background/bloc_5.jpeg') #read the image

img_resized = rescaleFrame(img)

# Blur 
blur = cv.GaussianBlur(img_resized, (3,3), cv.BORDER_DEFAULT) #reduce image noise

# Grayscale
gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

# Masking
blank = np.zeros(gray.shape[:2], dtype='uint8') #important : mask needs to be the same size of the image
mask = cv.rectangle(blank, (517, 93), (1153, 650), 255, -1)
# masked_image = cv.bitwise_and(gray, gray, mask=mask)
# cv.imshow('Masked Image', masked_image)

# Edge Cascade
canny = cv.Canny(blur, 125, 175) #strictness of what is a contour

# Contours and Calculate centroids
min_area_threshold = 20  # Adjust this sensitivity parameter based on your needs
centroids = calculate_centroids(canny, min_area_threshold, mask)

for i, (cx, cy) in enumerate(centroids):
    print(f"Centroids at ({cx}, {cy})")
    cv.circle(img_resized, (cx, cy), 5, (0, 255, 0), thickness = -1)

cv.imshow('Contours Detected', img_resized)

cv.waitKey(0)