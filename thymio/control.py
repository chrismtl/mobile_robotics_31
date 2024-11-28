import numpy as np

async def stop_thymio(node): # The center
    is_pressed = node['button.center']
    if is_pressed == 1:
        print("Center button pressed. Stopping the algorithm.")
        return False
    else:
        return True
    
def velocity_with_odometry(node, angle):
    motors_speed = np.array([node["motor.left.speed"], node["motor.right.speed"]])
    vel = np.average(motors_speed)
    vel_x = vel * np.cos(angle)
    vel_y = vel * np.sin(angle)
    vel_meas = np.array([vel_x, vel_y])
    return vel_meas

# Control motors speed
def motors(l_speed=500, r_speed=500, verbose=False):
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