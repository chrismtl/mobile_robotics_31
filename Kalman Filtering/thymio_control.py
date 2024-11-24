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