from computer_vision.vision import *


def test_camera():
    capture = cv.VideoCapture(0)
    while True:
        _, frame = capture.read()

        cv.imshow('Video', frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break

def test_augmented_obstacles():
    map = Map()
    test_frame  = cv.imread("computer_vision/test_map.png")
    map.set_frame(test_frame)
    map.vision_start()
    map.show()
    cv.waitKey(0)
    map.vision_stop()
   

if __name__ == "__main__":
    test_camera()