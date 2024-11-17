import cv2 as cv
import numpy as np

# Reading Images
img = cv.imread('Photos/Green Background/bloc_3.jpeg') #read the image
cv.imshow('Blocs', img)

blank = np.zeros(img.shape[:2], dtype='uint8') #important : mask needs to be the same size of the image

mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)

masked_image = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Image', masked_image)

print(mask.shape)

cv.waitKey(0)