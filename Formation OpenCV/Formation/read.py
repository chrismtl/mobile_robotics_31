import cv2 as cv

# Reading Images
# img = cv.imread('Photos/Green Background/bloc_2.jpeg') #read the image
# cv.imshow('Bloc 2', img) #show the image, named img, in a new window named "Bloc 1"

# Reading Videos
capture = cv.VideoCapture(1) #read a video (int, for camera connected to the PC, or videos path)

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'): #if 'd' is pressed, break out of the while loop
        break

capture.release()
cv.destroyAllWindows()

# cv.waitKey(0) #wait for a key to be pressed during an infinite amount of time (0)