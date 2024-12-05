import numpy as np
from constants import *

def update_var(v_left, v_right, x_prev, y_prev, theta_prev, dt):
    """
    Compute the updated position and orientation of the Thymio robot based on its wheel speeds, 
    geometry, and the elapsed time.

    Args:
        v_left (float): Speed of the left wheel in mm/s.
        v_right (float): Speed of the right wheel in mm/s.
        x_prev (int): Previous x-coordinate of the robot in mm.
        y_prev (int): Previous y-coordinate of the robot in mm.
        theta_prev (float): Previous orientation of the robot in radians.
        dt (float): Time elapsed since the last update in seconds.

    Returns:
        tuple: (x, y, theta)
            - x (int): Updated x-coordinate in mm.
            - y (int): Updated y-coordinate in mm.
            - theta (float): Updated orientation in radians.
    """

    # Compute the linear and angular speed
    v = ((0.5*(v_left+v_right))*SPEED_COEFF)*PIXEL_PER_CM # [px/s]
    omega = (((0.5/WHEEL_AXLE_LENGTH)*(v_right-v_left))*SPEED_COEFF)*PIXEL_PER_CM # [rad/s]

    # Update position x y and theta
    x = int(x_prev + v * dt * np.cos(theta_prev)) 
    y = int(y_prev + v * dt * np.sin(theta_prev))
    theta = theta_prev - omega * dt 
    
    return x, y, theta