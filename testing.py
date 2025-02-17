from computer_vision.vision import *
import os
# Change path of the execution ot the path of file's directory 
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def test_camera():
    capture = cv.VideoCapture(0)
    while True:
        _, frame = capture.read()

        cv.imshow('Video', frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            capture.release()
            break

def test_onimage():
    map = Map()
    test_frame  = cv.imread("computer_vision/test_map.jpg")
    map.set_raw_frame(test_frame)
    map.set_frame(test_frame)
    map.detect_global_obstacles()
    map.show()
    cv.waitKey(0)
    map.vision_stop()
   
def test_pre_update():
    map = Map()
    while True:
        map.pre_update()
        if cv.waitKey(20) & 0xFF==ord('q'):
            break

if __name__ == "__main__":
    test_onimage()