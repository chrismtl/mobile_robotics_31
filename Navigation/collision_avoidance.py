import tdmclient.notebook
import numpy as np
from tdmclient import ClientAsync, aw
from thymio.control import motors
#await tdmclient.notebook.start()

async def collision_avoidance(nodes_slopes, segment, pose_est, error_est, node):
    speed0 = np.array([30,30])       #Actuation for going straight
    speed_turn_left = [-25,60]       #Actuation for turning left
    speed_turn_right = [60,-25]      #Actuation for turning right
    actuation = np.zeros(2)
    obstThrH = 20                    # High threshold for which the robot enters the avoidance obstacle
    obstacle = 0                     # 0 = no obstacle, 1 = obstacle avoidance 
    avoidance_mode = 0
    close_obst = 0                   # number of sensors that detect an obstacle
    path_reached = 0                 # 0 = not reached, 1 = reached
    obj_right = 0                    # 1 = object detected on the right, 0 = object detected on the left
    w_l = [10,  5, 8, -5, -10, 7, -2]
    w_r = [-10, -5, -8,  5,  10, -5,  15]
    

    # We need to get the prox values from the robot
    await node.wait_for_variables()
    prox = node['prox.horizontal']


    # Let's find if there is an obstacle in front of the robot
    for i in range(5):    
        if prox[i] > obstThrH:
            obstacle = 1
            avoidance_mode = 1
            print('Avoidance mode activated')
    if avoidance_mode == 0:
        print('No avoidance mode')
        return actuation, obstacle, segment
    
    while(avoidance_mode == 1):
        await node.wait_for_variables()
        prox = node['prox.horizontal']

        # Let's find if there is an obstacle in front of the robot
        obstacle = 0
        close_obst = 0
        for i in range(5):     
            if prox[i] > obstThrH:
                close_obst = close_obst + 1
                if i>2:
                        obj_right = 1
                        obstacle = 1
                else:
                        obj_right = 0
                        obstacle = 1
        
        if close_obst == 0:
            obstacle = 0

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
                    obstacle = 0
                    print ('Path_reached')
                    return actuation, obstacle, segment
                else:
                    path_reached = 0

        '''if obstacle == 1:
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
                    print ('PB3')
                    return actuation, obstacle, segment'''
        
        # Now, let's actuate the robot in order to surround the obstacle
        #leds_top = [30,30,30]
    
        x = [0,0,0,0,0,0,0]
        sensor_translation = 150

        if (obstacle == 0):
            actuation = speed0
            if (obj_right == 1):
                actuation[0] = speed_turn_right[0]
                actuation[1] = speed_turn_right[1]
            else:
                actuation[0] = speed_turn_left[0]
                actuation[1] = speed_turn_left[1]
        else:
            actuation = speed0
            for i in range (6):
                x[i] = prox[i] // sensor_translation
                actuation[0] = actuation[0] + x[i] * w_l[i]
                actuation[1] = actuation[1] + x[i] * w_r[i]
        print('PB4')
        await node.set_variables(motors(int(actuation[0]), int(actuation[1])))