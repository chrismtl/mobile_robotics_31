import numpy as np

T_s = 0.1
A = np.array([[1, 0, T_s, 0], [0, 1, 0, T_s], [0, 0, 1, 0], [0, 0, 0, 1]])
B = np.array([[T_s, 0], [0, T_s], [1, 0], [0, 1]])
C = np.eye(4)

# Covariance matrices
q_px = 0.1
q_py = 0.1
q_vx = 0.1
q_vy = 0.1
Q = np.diag([q_px, q_py, q_vx, q_vy])

def kalman_filter(y, u_old, mu_predict_old, cov_predict_old, view_blocked=False):
    """
    Estimates the current state using input sensor data and the previous state
    
    param speed: measured speed (Thymio units)
    param ground_prev: previous value of measured ground sensor
    param ground: measured ground sensor
    param pos_last_trans: position of the last transition detected by the ground sensor
    param x_est_prev: previous state a posteriori estimation
    param P_est_prev: previous state a posteriori covariance
    
    return pos_last_trans: updated if a transition has been detected
    return x_est: new a posteriori state estimation
    return P_est: new a posteriori state covariance
    """
    
    ## Prediciton through the a priori estimate
    # estimated mean of the state
    mu_predict = np.dot(A, mu_predict_old) + np.dot(B, u_old)
    
    # Estimated covariance of the state
    cov_predict = np.dot(A, np.dot(cov_predict_old, A.T)) + Q
    # cov_est_a_priori = cov_est_a_priori + Q if type(Q) != type(None) else P_est_a_priori

    # innovation / measurement residual
    i = y - np.dot(C, mu_predict)
    
    # measurement prediction covariance
    r_vx = 0.1
    r_vy = 0.1
    if view_blocked: # measurement for the position isn't reliable
        r_px = 10000000 
        r_py = 10000000
    else:
        r_px = 0.1
        r_py = 0.1
    R = np.diag([r_px, r_py, r_vx, r_vy])
    S = np.dot(C, np.dot(cov_predict, C.T)) + R
             
    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = np.dot(cov_predict, np.dot(C.T, np.linalg.inv(S)))
    
    # a posteriori estimate
    mu_est = mu_predict + np.dot(K,i)
    cov_est = cov_predict - np.dot(K,np.dot(C, cov_predict))
     
    return mu_est, cov_est