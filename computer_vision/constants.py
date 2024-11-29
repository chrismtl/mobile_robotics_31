# Debugs
DEBUG = False
CV_DRAW = True

# Files
CAMERA_CALIBRATION_FILE = 'computer_vision\calibration_chessboard.yaml'

# Vision
LINE_THICKNESS = 1
FIRST_FRAME = 30            # First frame to analyse
MARKER_SIZE_ROBOT = 0.037    # Side length of the ArUco marker in meters 

# Obstacles
MIN_OBSTACLE_SEGMENT_LENGTH = 10
MIN_AREA = 10
MAX_AREA = 180000
DISPLAY_OBSTACLES = "POLYGON"

# Metrics
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
MAP_HEIGHT_CM = 62.5
MAP_WIDTH_CM = 97.2
ROBOT_RADIUS_CM = 8
PIXEL_PER_CM = SCREEN_WIDTH/MAP_WIDTH_CM
ROBOT_RADIUS_PIXEL = ROBOT_RADIUS_CM * PIXEL_PER_CM

print(f"Pixels per cm: {PIXEL_PER_CM}")
print(f"Robot pixel radius: {ROBOT_RADIUS_PIXEL}")