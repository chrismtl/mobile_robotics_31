import numpy as np
from constants import *
from .odometry import *

# x = [px, py, theta]
# u = [vl, vr]

# Dynamic of system
A = np.eye(3)

# Dynamic of observator
C = np.eye(3)

def kalman_filter(y, vel_old, mu_est_old, cov_est_old, robot_found, dt):
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

    # Control Law partial derivatives
    df1_dv = dt*np.cos(mu_est_old[2])
    df2_dv = dt*np.sin(mu_est_old[2])
    df3_domega = dt

    # Prediction covariance noise
    # cst = 0.5
    q_var_px = (df1_dv**2)*(VAR_THYMIO_V) #0.27
    q_var_py = (df2_dv**2)*(VAR_THYMIO_V) #0.27
    q_var_theta = (df3_domega**2)*(VAR_THYMIO_OMEGA) #0.000124
    print("Prediction covariances:", q_var_px, q_var_py, q_var_theta)
    q_cov_px_py = 0#(df1_dv * df2_dv)*(VAR_THYMIO_VL + VAR_THYMIO_VR) 
    q_cov_px_theta = 0#(df1_dv * df3_dv)*(VAR_THYMIO_VR - VAR_THYMIO_VL)
    q_cov_py_theta = 0#(df2_dv * df3_dv)*(VAR_THYMIO_VR - VAR_THYMIO_VL)  
    Q = np.array([[      q_var_px,    q_cov_px_py, q_cov_px_theta], 
                  [   q_cov_px_py,       q_var_py, q_cov_py_theta],
                  [q_cov_px_theta, q_cov_py_theta,    q_var_theta]])
    
    # Measurement covariance noise
    if not robot_found: # measurement for the position isn't reliable
        r_var_px = np.inf
        r_var_py = np.inf
        r_var_theta = np.inf
        px, py, theta = odometry(vel_old[0], vel_old[1], mu_est_old[0], mu_est_old[1], mu_est_old[2], dt)
        y = np.array([px, py, theta])
    else:
        r_var_px = VAR_THYMIO_PX
        r_var_py = VAR_THYMIO_PY
        r_var_theta = VAR_THYMIO_THETA
    R = np.diag([r_var_px, r_var_py, r_var_theta])

    # Dynamic of command
    #B = (WHEEL_RADIUS*dt/2)*np.array([[np.cos(mu_predict_old[2]), np.cos(mu_predict_old[2])], [np.sin(mu_predict_old[2]), np.sin(mu_predict_old[2])], [1/WHEEL_AXLE_LENGTH, -1/WHEEL_AXLE_LENGTH]])
    B = np.array([[np.cos(mu_est_old[2])*dt, 0], [np.sin(mu_est_old[2])*dt, 0], [0, -dt]])
    
    # Predicition through the a previous state estimate
    u_old = np.zeros(2)
    u_old[0] = ((0.5*(vel_old[0]+vel_old[1]))*SPEED_COEFF)*PIXEL_PER_CM # [px/s]
    u_old[1] = (((0.5/WHEEL_AXLE_LENGTH)*(vel_old[1]-vel_old[0]))*SPEED_COEFF)*PIXEL_PER_CM # [rad/s]
    mu_predict = np.dot(A, mu_est_old) + np.dot(B, u_old)   
    
    # Estimated covariance of the state from the previous state covariance
    cov_predict = np.dot(A, np.dot(cov_est_old, A.T)) + Q

    # Innovation parameter
    i = y - np.dot(C, mu_predict)
    
    S = np.dot(C, np.dot(cov_predict, C.T)) + R
             
    # Kalman gain
    K = np.dot(cov_predict, np.dot(C.T, np.linalg.inv(S)))
    
    # Next state estimate and covariance
    mu_est = mu_predict + np.dot(K,i)
    cov_est = cov_predict - np.dot(K,np.dot(C, cov_predict))
     
    return mu_est, cov_est