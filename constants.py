"""
Dynamical Constant file containing every chosen constants for our project.
"""
# FILES
CAMERA_CALIBRATION_FILE = 'computer_vision\\calibration_chessboard.yaml'

# ARUCO TAGS
AT_TOP_LEFT     = 0
AT_BOTTOM_LEFT  = 1
AT_BOTTOM_RIGHT = 2
AT_TOP_RIGHT    = 3
AT_ROBOT        = 4
AT_DESTINATION  = 5

# VISION
EPSILON = 5
LINE_THICKNESS = 1
FIRST_FRAME = 100            # First frame to analyse
MARKER_SIZE_ROBOT = 0.037    # Side length of the ArUco marker in meters 
FIRST_FRAME = 15             # First frame to analyse
ROBOT_RADIUS = 200

# METRICS
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
MAP_HEIGHT_CM = 62.5
MAP_WIDTH_CM = 97.2
ROBOT_RADIUS_CM = 12
PIXEL_PER_CM = SCREEN_WIDTH/MAP_WIDTH_CM
ROBOT_RADIUS_PIXEL = int(ROBOT_RADIUS_CM * PIXEL_PER_CM)
TARGET_RADIUS_PIXEL = int(100 * MARKER_SIZE_ROBOT * PIXEL_PER_CM)

# GRAPHISMS
SHOW_ALL_PATHS = True
D_ARROW_LINE_WIDTH = 6
DESTINATION_CROSS_WIDTH = 6
D_ROBOT_CIRCLE_RADIUS = int(ROBOT_RADIUS_PIXEL/3)
D_DESTINATION_CIRCLE_RADIUS = TARGET_RADIUS_PIXEL

# COLORS
X_AXIS_COLOR        = (0,0,255)
Y_AXIS_COLOR        = (0,255,0)
ROBOT_COLOR         = (0,255,0)
KALMAN_COLOR        = (255,0,0)
ROBOT_ARROW_COLOR   = (1,50,32)
DESTINATION_COLOR   = (0,0,255)
PATH_COLOR          = (43,255,255)

# OBSTACLES
MIN_OBSTACLE_SEGMENT_LENGTH = 10
MIN_AREA = 20
MAX_AREA = 180000
DISPLAY_OBSTACLES = "POINTS"
MIN_SECURITY = 2
MIN_DIST_TO_ROBOT = MIN_SECURITY*ROBOT_RADIUS_PIXEL
MIN_DIST_TO_DESTINATION = MIN_SECURITY*TARGET_RADIUS_PIXEL

# ROBOT CONSTANTS
SPEED_COEFF = 0.04 # conversion factor to get thymio's speed in cm/s
WHEEL_RADIUS = 1#4 * PIXEL_PER_CM # wheel radius [px]
WHEEL_AXLE_LENGTH = 9.5 * PIXEL_PER_CM # wheels axle length [px]

# Measurement covariances
# cov_thymio_px = 0.1
# cov_thymio_py = 0.004
VAR_THYMIO_PX = 0.0007 # [px^2]
VAR_THYMIO_PY = 0.16 # [px^2]
VAR_THYMIO_THETA = 0.002 # [rad^2]
VAR_THYMIO_VL = 39.84 * (SPEED_COEFF**2) * (PIXEL_PER_CM**2) # [px^2/s^2]
VAR_THYMIO_VR = 23.33 * (SPEED_COEFF**2) * (PIXEL_PER_CM**2) # [px^2/s^2]
VAR_THYMIO_V = (0.5**2)*(VAR_THYMIO_VL+VAR_THYMIO_VR) # [px^2/s^2]
VAR_THYMIO_OMEGA = ((0.5/WHEEL_AXLE_LENGTH)**2)*(VAR_THYMIO_VL-VAR_THYMIO_VR)

# DEBUGS
DEBUG = False
CV_DRAW = False
P_VISION = False
P_SETUP = "=====[   SETUP      ]==============="
P_START = "=====[   START      ]==============="
P_STOP  = "=====[   STOP       ]==============="
P_INFO  = "=====[   MAP INFO   ]==============="
P_END   = "====================================\n"