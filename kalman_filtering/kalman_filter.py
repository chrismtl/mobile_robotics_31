import numpy as np
from constants import *

# x = [px, py, theta]
# u = [vl, vr]

# Dynamic of system
A = np.eye(3)

# Dynamic of observator
C = np.eye(3)

def kalman_filter(y, u_old, mu_predict_old, cov_predict_old, robot_found, dt):
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

    # Dynamic of command
    B = (WHEEL_RADIUS*dt/2)*np.array([[np.cos(mu_predict_old[2]), np.cos(mu_predict_old[2])], [np.sin(mu_predict_old[2]), np.sin(mu_predict_old[2])], [1/WHEEL_AXLE_LENGTH, -1/WHEEL_AXLE_LENGTH]])

    # Control Law partial derivatives
    df1_dv = (WHEEL_RADIUS*dt/2)*np.cos(mu_predict_old[2])
    df2_dv = (WHEEL_RADIUS*dt/2)*np.sin(mu_predict_old[2])
    df3_dv = (WHEEL_RADIUS*dt/(2*WHEEL_AXLE_LENGTH))

    # Prediction covariance noise
    cst = 100000000
    q_var_px = cst#(df1_dv**2)*(VAR_THYMIO_VL + VAR_THYMIO_VR)
    q_var_py = cst#(df2_dv**2)*(VAR_THYMIO_VL + VAR_THYMIO_VR)
    q_var_theta = cst#(df3_dv**2)*(VAR_THYMIO_VL + VAR_THYMIO_VR)
    q_cov_px_py = 0*(df1_dv * df2_dv)*(VAR_THYMIO_VL + VAR_THYMIO_VR) 
    q_cov_px_theta = 0*(df1_dv * df3_dv)*(VAR_THYMIO_VR - VAR_THYMIO_VL)
    q_cov_py_theta = 0*(df2_dv * df3_dv)*(VAR_THYMIO_VR - VAR_THYMIO_VL)  
    Q = np.array([[      q_var_px,    q_cov_px_py, q_cov_px_theta], 
                  [   q_cov_px_py,       q_var_py, q_cov_py_theta],
                  [q_cov_px_theta, q_cov_py_theta,    q_var_theta]]) #0.01 trop bas
    
    # Measurement covariance noise
    if not robot_found: # measurement for the position isn't reliable
        r_var_px = np.inf
        r_var_py = np.inf
        r_var_theta = np.inf
    else:
        r_var_px = VAR_THYMIO_PX
        r_var_py = VAR_THYMIO_PY
        r_var_theta = VAR_THYMIO_THETA
    R = np.diag([r_var_px, r_var_py, r_var_theta])

    # Predicition through the a previous state estimate
    u_old = u_old * SPEED_COEFF * PIXEL_PER_CM # [px/s]
    mu_predict = np.dot(A, mu_predict_old) + np.dot(B, u_old)
    
    # Estimated covariance of the state from the previous state covariance
    cov_predict = np.dot(A, np.dot(cov_predict_old, A.T)) + Q

    # Innovation parameter
    i = y - np.dot(C, mu_predict)
    
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