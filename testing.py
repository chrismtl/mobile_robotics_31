from computer_vision.vision import *

capture = cv.VideoCapture(0)

def test_camera():
    while True:
        _, frame = capture.read()

        cv.imshow('Video', frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break

#test_camera()

map = Map()

while True:
    map.update()
    map.show()
    if cv.waitKey(1) != -1:
        map.vision_stop()
        break
   
map.info()
cv.destroyAllWindows()