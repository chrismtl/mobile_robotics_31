import cv2 as cv

# Resizing frame or image
def rescaleFrame(frame, scale=0.75):
    # For images, videos and live videos
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Reading Videos
capture = cv.VideoCapture(0) #read a video (int, for camera connected to the PC, or videos path)

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame)

    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'): #if 'd' is pressed, break out of the while loop
        break

capture.release()
cv.destroyAllWindows()