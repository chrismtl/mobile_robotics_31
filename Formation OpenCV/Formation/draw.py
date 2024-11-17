import cv2 as cv
import numpy as np

# Reading Images
blank = np.zeros((500, 500, 3), dtype='uint8')
# cv.imshow('Blank', blank)

# 1. Paint the image of a certain color
blank[:] = 0, 255, 255 #B, G, R
# cv.imshow('White', blank)

# 2. Draw a rectangle 
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,0,255), thickness=-1) #draw a red rectangle in the image named blank, from (0,0) to (250,250)
# cv.imshow('Rectangle', blank)

# 3. Draw a circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (255,0,255), thickness=-1) #thickness=-1 -> completely colored
# cv.imshow('Circle', blank)

# 4. Draw a line
cv.line(blank, (100,250), (300,400), (255,0,0), thickness=3)
# cv.imshow('Line', blank)

# 5. Write a text
cv.putText(blank, 'Hello', (375,400), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
cv.imshow('Text', blank)

cv.waitKey(0) #wait for a key to be pressed during an infinite amount of time (0)