import cv2 as cv
import matplotlib.pyplot as plt

# White balancing
def white_balance(image):
    # Create a SimpleWB object and balance the white
    white_balancer = cv.xphoto.createSimpleWB()
    result = white_balancer.balanceWhite(image)
    return result

# Reading Images
img = cv.imread('Photos/Green Background/bloc_4.jpeg') #read the image
cv.imshow('Blocs', img)

white_balanced_image = white_balance(img)
cv.imshow('White Balancing', white_balanced_image)

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)
lab_wb = cv.cvtColor(white_balanced_image, cv.COLOR_BGR2LAB)
cv.imshow('LAB White Balanced', lab_wb)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

# plt.imshow(rgb)
# plt.show()

cv.waitKey(0)