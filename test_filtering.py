# %%
from filterpy.kalman import KalmanFilter
import numpy as np
import cv2
import math
from constants import *


# %%

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

# %%

def run_filter( u_old, mu_predict_old, robot_found, dt, map)
                #speed_right, speed_left, prev_angle, vis):

    global f

    #camera_scale = vis.scale # [camera_coordinates/m]
    #PIXEL_PER_CM *100
    # Converting the motors to rad/s
    speed_right = u_old[1] / WHEEL_RADIUS
    speed_left = u_old[0] / WHEEL_RADIUS
    
    # Defining control input and control transition matrix
    u = np.array([[speed_right],
                  [speed_left]])
    B = np.array([[np.cos(mu_predict_old[2])*(dt/2), np.cos(mu_predict_old[2])*(dt/2)],
                    [np.sin(mu_predict_old[2])*(dt/2), np.sin(mu_predict_old[2])*(dt/2)],
                    [(-dt/WHEEL_AXLE_LENGTH), (dt/WHEEL_AXLE_LENGTH)]]) * (WHEEL_RADIUS*dt/2)
    
    # Getting camera measurements and conveting to [m]
    measurement = np.array([mu_predict_old[0]/camera_scale, mu_predict_old[1]/camera_scale, mu_predict_old[2]])
    
    # Predict step of kalman filter with control input
    f.predict(u = u, B = B)
    # Only update if we have new camera measurements
    if robot_found:
        f.update(measurement)

    # Defining the estimate in camera coordinates
    estimate = np.array([f.x[0,0] * PIXEL_PER_CM *100, f.x[1,0] * PIXEL_PER_CM *100, f.x[2,0]])

    # Plotting an ellipse to show evolution of uncertainty along x and y
    cv2.ellipse(map.copy, (int(estimate[0]), int(estimate[1])), (int(200 * f.P[0,0]), int(200 * f.P[1,1])), math.degrees(f.x[2,0]),0,360,(255,0,0),3)
    # Plotting a line to show the angle estimate
    cv2.line(map.copy, (int(estimate[0]), int(estimate[1])), (int(estimate[0] + 100*np.cos(estimate[2])), int(estimate[1] + 100*np.sin(estimate[2]))), (255,0,0), 3)
    
    return estimate # Return the kalman filtered state