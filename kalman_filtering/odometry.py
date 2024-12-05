import numpy as np
from constants import *

def odometry(v_left, v_right, x_prev, y_prev, theta_prev, dt):
    """
    Compute the new position of the Thymio based on the wheel speeds, the time passed, the Thymio's geometry and the previous position.

    :param x_prev: int (previous x position in mm)
    :param y_prev: int (previous y position in mm)
    :param theta_prev: float (previous orientation in rad)

    :return: tuple (new x position in mm, new y position in mm, new orientation in rad, linear velocity in mm/s, angular velocity in rad/s)
    """
    # Compute the linear velocity (in mm/s) and angular velocity (in rad/s) of the Thymio
    v = ((0.5*(v_left+v_right))*SPEED_COEFF)*PIXEL_PER_CM # [px/s]
    omega = (((0.5/WHEEL_AXLE_LENGTH)*(v_left-v_right))*SPEED_COEFF)*PIXEL_PER_CM # [rad/s]
    #print("Motors speed", [v_left, v_right])
    #print("Robot angular velocity", omega)

    # Update position x y (in mm)
    x = int(x_prev + v * dt * np.cos(theta_prev)) 
    y = int(y_prev + v * dt * np.sin(theta_prev))

    # Update orientation theta (in rad)
    theta = theta_prev + omega * dt # Minus sign due to the y-axis orientation in the camera frame, pointing down instead of up
    
    return x, y, theta, v, omega