import tdmclient.notebook
import numpy as np
from tdmclient import ClientAsync, aw
from thymio.control import motors
#await tdmclient.notebook.start()

async def collision_avoidance(path, node, pos, avoidance_mode, segment_index, obj_right, obstacle_pos):
    error = [10,25]
    error_f = 18
    speed_turn_left = [125,300]       #Actuation for turning left
    speed_turn_right = [300,125]      #Actuation for turning right
    actuation = np.array([0,0])
    obstThrH = 20      # High threshold for which the robot enters the avoidance obstacle
    obstacle = 0         # 0 = no obstacle, 1 = obstacle avoidance 

    await node.wait_for_variables()
    prox = node['prox.horizontal'] # Lire les valeurs des capteurs de proximité
   
    for i in range (5):     # find if the obstacle is on the left or on the right
        if prox[i] > obstThrH:
            obstacle = 1
            if avoidance_mode == 0:
                #print('Obstacle !!!!')
                obstacle_pos = pos
                avoidance_mode = 1
                if i in [2,3,4]:
                    obj_right = 1
                    ##print('Objet a droite détecté')
                if i in [0,1]:
                    obj_right = 0  
                    ##print('Objet a gauche détecté')

    #if (obstacle == 0) and (avoidance_mode == 0):
        ##print('Pas d obstacle')

    #Let's check if we reached a segment_index of the global path
    pos = np.array([pos[0], pos[1]])  
    obstacle_pos = np.array([obstacle_pos[0], obstacle_pos[1]])
    distance_to_obstacle = np.linalg.norm(pos - obstacle_pos)
    ##print('Distance to obstacle: ', distance_to_obstacle)

    for i in range (0, path.shape[0]-1):
        alpha = path[i,2]
        beta = path[i,3]
        x_path = [path[i,0]-error[0], path[i+1,0]+error[0]]
        y_path = [path[i,1]-error[1], path[i+1,1]+error[1]]
        x_path.sort()
        y_path.sort()
        #print('Test 1')
        #print('x range', x_path)
        #print('y range', y_path)
        #print('pos', pos)
        if ((x_path[0] <= pos[0] <= x_path[1]) and (y_path[0] <= pos[1] <= y_path[1])):
            #print('Zone du segment numéro', i)
            pose_y_min = pos[1] - error[1]
            pose_y_max = pos[1] + error[1]

            fx_min = alpha * (pos[0] - error[0]) + beta
            fx_max = alpha * (pos[0] + error[0]) + beta

            if alpha < 0:                               # If alpha is negative, we have to inverse the boundaries
                fx_min, fx_max = fx_max, fx_min
            
            fx_min = fx_min
            fx_max = fx_max

            #print('Test 2')
            #print('pose_y_min', pose_y_min)
            #print('pose_y_max', pose_y_max) 
            #print('fx_min', fx_min)
            #print('fx_max', fx_max)

            if ((pose_y_min <= fx_max <= pose_y_max) or (pose_y_min <= fx_min <= pose_y_max))and(distance_to_obstacle>30):
                if (avoidance_mode == 0):
                    #print('Je suis sur le segment numéro', i)
                    segment_index = i
                else:
                    avoidance_mode = 0
                    segment_index = i
                    obstacle = 0
                    #print('J ai rejoint le segment numéro', i)
                

    # Now, let's actuate the robot in order to surround the obstacle
    # Leds_top = [30,30,30]

    w_l = [40,  20, -20, -20, -40,  30, -10]
    w_r = [-40, -20, -20,  20,  40, -10,  30]
    w_l_test = [50,  25, -20, 0, 0,  30, -10]
    w_r_test = [0, 0, -20,  25,  50, -10,  30]
    x = [0,0,0,0,0,0,0]
    sensor_translation = 350

    if (avoidance_mode == 1):
        if obstacle == 0:
            if (obj_right == 1):
                actuation[0] = speed_turn_right[0]
                actuation[1] = speed_turn_right[1]
                #print("L'objet est a droite")
            else:
                actuation[0] = speed_turn_left[0]
                actuation[1] = speed_turn_left[1]
                #print("L'objet est a gauche")
        else:
            for i in range (6):
                x[i] = prox[i] // sensor_translation
                actuation[0] = actuation[0] + x[i] * w_l_test[i]
                actuation[1] = actuation[1] + x[i] * w_r_test[i]
    #print("Le segment retourné est le numéro", segment_index)
    return actuation, avoidance_mode, segment_index, obj_right, obstacle_pos