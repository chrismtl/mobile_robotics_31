import cv2 as cv 
import numpy as np

# Reading Images
img = cv.imread('Photos/Green Background/bloc_3.jpeg') #read the image

cv.imshow('Cats', img)

blank = np.zeros(img.shape, dtype='uint8')

# Gray 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Blur 
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)

# Edge Cascade
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

# Thresholding
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)

# Contours
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) #possible to pass thresh instead of canny as first argument
#cv.RETR_LIST returns all the contours of the image, cv.RETR_EXTERNAL returns all the external contours and cv.RETR_TREE return all the hierarchal (?) contours
#cv.CHAIN_APPROX_NONE returns all the contours, cv.CHAIN_APPROX_SIMPLE compresses all the contours returned (ex : a line is compressed into two points)
print(f'{len(contours)} contour(s) found !')
cv.drawContours(blank, contours, -1, (0, 0, 255), 1) #drawing the contours found in a blank image
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)