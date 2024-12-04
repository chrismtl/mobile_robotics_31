import numpy as np

T_s = 0.1
A = np.array([[1, 0, T_s, 0], [0, 1, 0, T_s], [0, 0, 1, 0], [0, 0, 0, 1]])
B = np.array([[T_s, 0], [0, T_s], [1, 0], [0, 1]])
C = np.eye(4)

# Covariance matrices
q_px = 2
q_py = 2
q_vx = 3 #0.0008 #measured
q_vy = 3 #0.0008 #measured
Q = np.diag([q_px, q_py, q_vx, q_vy])

cov_thymio_vx = 22.6
cov_thymio_vy = 22.6
cov_thymio_px = 0.1
cov_thymio_py = 0.004

speed_coeff = 0.04 #conversion factor to get thymio's speed in cm/s
SCREEN_WIDTH = 640
MAP_WIDTH_CM = 97.2
px_conv_x = SCREEN_WIDTH/MAP_WIDTH_CM #px/cm
SCREEN_HEIGHT = 480
MAP_HEIGHT_CM = 62.5
px_conv_y = SCREEN_HEIGHT/MAP_HEIGHT_CM #px/cm

def kalman_filter(y, u_old, mu_predict_old, cov_predict_old, robot_found):
    """
    Estimates the current state of the robot using input odometry and camera data and the previous state
    
    param y: state measurements
    param u_old: previous speed sent to the motors
    param mu_predict_old: previous state estimation
    param cov_predict_old: previous state covariance
    param robot_found: indicator to check if the view is blocked
    
    return mu_est: next state estimation
    return cov_est: next state covariance

    note: strongly inspired from the Kalman filer algorithm provided in the solutions of the exercises session 7
    """

    # Predicition through the a previous state estimate
    mu_predict = np.dot(A, mu_predict_old) + np.dot(B, u_old)
    #print("Mu prediction:", mu_predict)
    
    # Estimated covariance of the state from the previous state covariance
    cov_predict = np.dot(A, np.dot(cov_predict_old, A.T)) + Q

    # Innovation parameter
    i = y - np.dot(C, mu_predict)
    
    # Prediction of the measurement covariance
    r_vx = cov_thymio_vx * speed_coeff**2 * px_conv_x**2 # covariance in px^2/s^2
    r_vy = cov_thymio_vy * speed_coeff**2 * px_conv_y**2 # covariance in px^2/s^2
    if not robot_found: # measurement for the position isn't reliable
        r_px = 10000000 
        r_py = 10000000
    else:
        r_px = cov_thymio_px
        r_py = cov_thymio_py
    R = np.diag([r_px, r_py, r_vx, r_vy])
    S = np.dot(C, np.dot(cov_predict, C.T)) + R
             
    # Kalman gain
    K = np.dot(cov_predict, np.dot(C.T, np.linalg.inv(S)))
    
    # Next state estimate and covariance
    mu_est = mu_predict + np.dot(K,i)
    cov_est = cov_predict - np.dot(K,np.dot(C, cov_predict))
     
    return mu_est, cov_est

def check_angle(kalman_angle , robot_angle):
    """
    Compares the estimated robot angle with the measured angle 

    param angle: robot angle from kalman
    param robot_angle: measured robot angle by the camera
    
    return est_angle: estimated robot angle

    note: strongly inspired from the Kalman filer algorithm provided in the solutions of the exercises session 7
    """
    diff_angle = abs(kalman_angle-robot_angle)

    if diff_angle <= np.pi/2:
        est_angle = kalman_angle
    else:
        est_angle = robot_angle

    return est_angle