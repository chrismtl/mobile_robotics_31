import numpy as np

async def stop_thymio(node): # The center
    is_pressed = node['button.center']
    if is_pressed == 1:
        print("Center button pressed. Stopping the algorithm.")
        return False
    else:
        return True
    
def velocity_with_odometry(node, angle):
    speed_coeff = 0.04 #conversion factor to get thymio's speed in cm/s
    SCREEN_WIDTH = 640
    MAP_WIDTH_CM = 97.2
    px_conv_x = SCREEN_WIDTH/MAP_WIDTH_CM #px/cm
    SCREEN_HEIGHT = 480
    MAP_HEIGHT_CM = 62.5
    px_conv_y = SCREEN_HEIGHT/MAP_HEIGHT_CM #px/cm
    motors_speed = np.array([node["motor.left.speed"], node["motor.right.speed"]])
    print("Raw vel:", motors_speed[0], motors_speed[1])
    vel = np.average(motors_speed)
    vel_x = vel * np.cos(angle) * speed_coeff * px_conv_x
    vel_y = vel * np.sin(angle) * speed_coeff * px_conv_y
    vel_meas = np.array([vel_x, vel_y])
    return vel_meas

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