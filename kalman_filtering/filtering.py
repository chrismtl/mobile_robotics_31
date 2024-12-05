from filterpy.kalman import KalmanFilter
import numpy as np
import cv2
import math


## General parameters
dimension_x = 3 # State dimension
dimension_z = 3 # Measurement dimension

motor_scale = 43.52 # [Motor_space/(rad/s)] Scale of motor speeds in motor space to rad/s
R = 0.021 # [m] The radius of the Thymio's wheels
d = 0.095 # [m] The wheelbase of the Thymio
dt = 0.137 # [s] Time delta between steps

# Creating the filter
f = KalmanFilter(dim_x=dimension_x, dim_z=dimension_z) # state and measurement variables are x, y and theta

## Filter parameters
# State transition matrix
f.F = np.eye(3)        
# Measurement function
f.H = np.eye(3)
# Initial covariance matrix
f.P = np.eye(3) * 100
# Measurement noise
camera_variances = [2.13554018e-01, 2.93571267e-01, 6.02748876e-05]
f.R = np.diag(camera_variances)
# Process noise
process_variances = [3.8751605996417765e-01, 3.8751605996417765e-01, 2.9656863710880975e-03]
f.Q = np.diag(process_variances)


def run_filter(speed_right, speed_left, pose_old):
    # Converting the motors to rad/s
    speed_right = speed_right / motor_scale
    speed_left = speed_left / motor_scale
    
    # Defining control input and control transition matrix
    u = np.array([[speed_right],
                  [speed_left]])
    B = np.array([[np.cos(pose_old[2])*(dt/2), np.cos(pose_old[2])*(dt/2)],
                    [np.sin(pose_old[2])*(dt/2), np.sin(pose_old[2])*(dt/2)],
                    [(-dt/d), (dt/d)]]) * R
    
    # Predict step of kalman filter with control input
    f.predict(u = u, B = B)
    # Only update if we have new camera measurements

    # Defining the estimate in camera coordinates
    estimate = np.array([f.x[0,0], f.x[1,0], f.x[2,0]])
    
    return estimate # Return the kalman filtered state