import numpy as np
from constants import *

async def stop_thymio(node): # The center
    is_pressed = node['button.center']
    if is_pressed == 1:
        print("Center button pressed. Stopping the algorithm.")
        return False
    else:
        return True
    
def get_thymio_velocity(node):  
    motors_speed = np.array([node["motor.left.speed"], node["motor.right.speed"]])
    return motors_speed

# Control motors speed
def motors(l_speed=100, r_speed=100, verbose=False):
    """
    Sets the motor speeds of the Thymio 
    param l_speed: left motor speed
    param r_speed: right motor speed
    param verbose: whether to print status messages or not
    """
    # Printing the speeds if requested
    if verbose:
        print("\t\t Setting speed : ", l_speed, r_speed)
    return {
        "motor.left.target": [l_speed],
        "motor.right.target": [r_speed],
    }