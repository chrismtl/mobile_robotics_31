import tdmclient.notebook
from Navigation_utils import *
from tdmclient import ClientAsync, aw
#await tdmclient.notebook.start()

def collision_avoidance(nodes_slopes, pose_est, error_est, old_obstacle, segment):
    speed_turn_left = [-25,50]       #Actuation for turning left
    speed_turn_right = [50,-25]      #Actuation for turning right
    actuation = [0,0]
    #obstThrL = 10      # Low threshold for which the robot exits the avoidance obstacle
    obstThrH = 20      # High threshold for which the robot enters the avoidance obstacle
    obstacle = 0         # 0 = no obstacle, 1 = obstacle avoidance 
    close_obst = 0      # number of sensors that detect an obstacle

    path_reached = 0    # 0 = not reached, 1 = reached
    obj_right = 0      # 1 = object detected on the right, 0 = object detected on the left
    
    await node.wait_for_variables()
    prox = node['prox.horizontal'] # Lire les valeurs des capteurs de proximité

    if old_obstacle == 0:       # The robot is not in the avoidance mode
        for i in range (5):     # Let's find if there is an obstacle in front of the robot
            if prox[i] > obstThrH:
                obstacle = 1
        if obstacle == 0:
            return actuation, obstacle, segment

    #Let's check if we reached a segment of the global path
    for i in range (0, nodes_slopes.shape[0]-1):
        alpha = nodes_slopes[i,2]
        beta = nodes_slopes[i,3]
        x_path = [nodes_slopes[i,0], nodes_slopes[i+1,0]]       
        y_path = [nodes_slopes[i,1], nodes_slopes[i+1,1]]
        x_path.sort()
        y_path.sort()
        if ((x_path[0] - error_est[0] <= pose_est[0] <= x_path[1] + error_est[0]) 
            and (y_path[0] - error_est[1] <= pose_est[1] <= y_path[1] + error_est[1])):
            pose_y_min = pose_est[1] - error_est[1]
            pose_y_max = pose_est[1] + error_est[1]

            fx_min = alpha * (pose_est[0] - error_est[0]) + beta
            fx_max = alpha * (pose_est[0] + error_est[0]) + beta

            if alpha < 0:                   # If alpha is negative, we have to inverse the boundaries
                fx_min, fx_max = fx_max, fx_min

            if (pose_y_min <= fx_max <= pose_y_max) or (pose_y_min <= fx_min <= pose_y_max):
                path_reached = 1
                segment = i
            else:
                path_reached = 0

    if old_obstacle == 1:
        for i in range (5):     # find if the obstacle is on the left or on the right
            if prox[i] > obstThrH:
                close_obst = close_obst + 1
                if i>2:
                    obj_right = 1
                    obstacle = 1
                else:
                    obj_right = 0
                    obstacle = 1
        if close_obst == 0:
            if path_reached == 1:
                obstacle = 0
                return actuation, obstacle, segment
    
    # Now, let's actuate the robot in order to surround the obstacle
    #leds_top = [30,30,30]
    w_l = [40,  20, -20, -20, -40,  30, -10]
    w_r = [-40, -20, -20,  20,  40, -10,  30]
    x = [0,0,0,0,0,0,0]
    sensor_translation = 100

    if (close_obst == 0):
        if (obj_right == 1):
            actuation[0] = speed_turn_right[0]
            actuation[1] = speed_turn_right[1]
        else:
            actuation[0] = speed_turn_left[0]
            actuation[1] = speed_turn_left[1]
    else:
        for i in range (6):
            x[i] = prox[i] // sensor_translation
            actuation[0] = actuation[0] + x[i] * w_l[i]
            actuation[1] = actuation[1] + x[i] * w_r[i]

        return actuation, obstacle, segment