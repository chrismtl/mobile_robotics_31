import tdmclient.notebook
import numpy as np
from tdmclient import ClientAsync, aw
from thymio.control import motors
#await tdmclient.notebook.start()

def object_detection(prox, avoidance_mode, obstacle, obj_right, obstacle_pos, pos):
    '''
    This function checks if there is an obstacle in front of the robot.
    If there is one, it returns the position of the obstacle and the side of the obstacle.

    inputs:
            prox: the proximity sensors' values
            avoidance_mode: 0 if the robot is not avoiding an obstacle, 1 if it is
            obstacle: 0 if there is no obstacle, 1 if there is one
            obj_right: 1 if the obstacle is on the right, 0 if it is on the left
            obstacle_pos: the position of the start of the obstacle
            pos: the robot's position

    outputs:
            avoidance_mode: 0 if the robot is not avoiding an obstacle, 1 if it is
            obstacle: 0 if there is no obstacle, 1 if there is one
            obj_right: 1 if the obstacle is on the right, 0 if it is on the left
            obstacle_pos: the position of the start of the obstacle
    '''
    obstThrH = 20           # High threshold for which the robot enters the avoidance obstacle
    
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
    return avoidance_mode, obstacle, obj_right, obstacle_pos


def segment_check(path, pos, avoidance_mode, segment_index, obstacle, obstacle_pos):
    ''' 
    This function checks if the robot is on a segment of the global path
    and if it is, it returns the index of the segment.

    inputs:
            path: matrix of the global path
            pos: the robot's position
            avoidance_mode: 0 if the robot is not avoiding an obstacle, 1 if it is
            segment_index: the index of the segment, from the path, the robot is currently on
            obstacle: 0 if there is no obstacle, 1 if there is one
    
    outputs:
            segment_index: the index of the segment, from the path, the robot is currently on
            avoidance_mode: 0 if the robot is not avoiding an obstacle, 1 if it is
            obstacle: 0 if there is no obstacle, 1 if there is one
    '''

    error = [10,25]                   # Error on the position of the robot
    distance_to_obstacle = np.linalg.norm(pos - obstacle_pos)
    print('distance_to_obstacle', distance_to_obstacle)
    for i in range (0, path.shape[0]-1):
        alpha = path[i,2]
        beta = path[i,3]
        x_path = [path[i,0]-error[0], path[i+1,0]+error[0]]
        y_path = [path[i,1]-error[1], path[i+1,1]+error[1]]
        x_path.sort()
        y_path.sort()

        if ((x_path[0] <= pos[0] <= x_path[1]) and (y_path[0] <= pos[1] <= y_path[1])):
            pose_y_min = pos[1] - error[1]
            pose_y_max = pos[1] + error[1]

            fx_min = alpha * (pos[0] - error[0]) + beta
            fx_max = alpha * (pos[0] + error[0]) + beta

            if alpha < 0:                               # If alpha is negative, we have to inverse the boundaries
                fx_min, fx_max = fx_max, fx_min
            
            fx_min = fx_min
            fx_max = fx_max

            if ((pose_y_min <= fx_max <= pose_y_max) or (pose_y_min <= fx_min <= pose_y_max))and(distance_to_obstacle>30):
                if (avoidance_mode == 0):
                    segment_index = i
                else:
                    avoidance_mode = 0
                    segment_index = i
                    obstacle = 0
    return segment_index, avoidance_mode, obstacle


def actuation_robot(prox, avoidance_mode, obj_right, obstacle, actuation):
    '''
    This function returns the actuation commands to control the wheels' motors
    in order to avoid the obstacle.

    inputs: 
            prox: the proximity sensors' values
            avoidance_mode: 0 if the robot is not avoiding an obstacle, 1 if it is
            obj_right: 1 if the obstacle is on the right, 0 if it is on the left
            obstacle: 0 if there is no obstacle, 1 if there is one
            actuation: the actuation commands to control the wheels' motors

    outputs:
            actuation: the actuation commands to control the wheels' motors 
    '''
    w_l_test = [50,  25, -20, 0, 0,  30, -10]
    w_r_test = [0, 0, -20,  25,  50, -10,  30]
    speed_turn_left = [125,300]       #Actuation for turning left
    speed_turn_right = [300,125]      #Actuation for turning right
    x = [0,0,0,0,0,0,0]
    sensor_translation = 350

    if (avoidance_mode == 1):
        if obstacle == 0:
            if (obj_right == 1):
                actuation[0] = speed_turn_right[0]
                actuation[1] = speed_turn_right[1]
            else:
                actuation[0] = speed_turn_left[0]
                actuation[1] = speed_turn_left[1]
        else:
            for i in range (6):
                x[i] = prox[i] // sensor_translation
                actuation[0] = actuation[0] + x[i] * w_l_test[i]
                actuation[1] = actuation[1] + x[i] * w_r_test[i]
    return actuation

async def collision_avoidance(path, node, pos, avoidance_mode, segment_index, obj_right, obstacle_pos, destination):
    '''
    This function was made to avoid unplanned obstacles that may appear on the robot's path.
    It uses the robot's sensors in order to detect the obstacles and then return actuation commands
    to the main.

    inputs:
            path: matrix of the global path
            node: contains the robot's variables
            avoidance_mode: 0 if the robot is not avoiding an obstacle, 1 if it is
            segment_index: the index of the segment, from the path, the robot is currently on
            obj_right: 1 if the obstacle is on the right, 0 if it is on the left
            obstacle_pos: the position of the start of the obstacle

    outputs:
            actuation: the actuation commands to control the wheels' motors
            avoidance_mode: 0 if the robot is not avoiding an obstacle, 1 if it is
            segment_index: the index of the segment, from the path, the robot is currently on
            obj_right: 1 if the obstacle is on the right, 0 if it is on the left
            obstacle_pos: the position of the start of the obstacle
    '''
    # Initialization of the variables
    actuation = np.array([0,0])
    pos = np.array([pos[0], pos[1]])  
    obstacle_pos = np.array([obstacle_pos[0], obstacle_pos[1]])
    obstacle = 0                      # 0 = no obstacle, 1 = obstacle avoidance 

    # Let's read the proximity sensors
    await node.wait_for_variables()
    prox = node['prox.horizontal'] 
   
    # Let's check if there is an obstacle in front of the robot
    avoidance_mode, obstacle, obj_right, obstacle_pos =  object_detection(prox, avoidance_mode, obstacle, obj_right, obstacle_pos, pos)

    #Let's check if we reached a segment_index of the global path
    segment_index, avoidance_mode, obstacle =  segment_check(path, pos, avoidance_mode, segment_index, obstacle, obstacle_pos)
    
    # Now, let's actuate the robot in order to surround the obstacle
    actuation = actuation_robot(prox, avoidance_mode, obj_right, obstacle, actuation)

    # We want to be able to move the destination without entering into avoidance mode
    if (np.linalg.norm(pos - destination) < 100):
        avoidance_mode = 0

    return actuation, avoidance_mode, segment_index, obj_right, obstacle_pos