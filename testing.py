from computer_vision.vision import *

map = Map()

while True:
    map.update()
    map.show()
    if cv.waitKey(1) != -1:
        map.vision_stop()
        break
   
map.info()
cv.destroyAllWindows()