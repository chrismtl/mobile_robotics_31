import tdmclient.notebook
import numpy as np
from tdmclient import ClientAsync, aw
from thymio.control import motors
#await tdmclient.notebook.start()

async def collision_avoidance(nodes_slopes, segment, pose_est, node):
    error_est = [10,10]
    speed0 = np.array([80,80])       #Actuation for going straight
    actuation = np.zeros(2)
    obstThrH = 20                    # High threshold for which the robot enters the avoidance obstacle
    obstacle = 0                     # 0 = no obstacle, 1 = obstacle avoidance 
    w_l = [10,  45, 24, -9, -2, 14, -2]
    w_r = [-2, -9, -4, 45,  10, -14,  2]
    
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
                segment = i
                obstacle = 0
                print ('Path_reached')
                return actuation, obstacle, segment

    # We need to get the prox values from the robot
    await node.wait_for_variables()
    prox = node['prox.horizontal']

    # Let's find if there is an obstacle in front of the robot
    for i in range(5):    
        if prox[i] > obstThrH:
            obstacle = 1
            print('Obstacle !!!!')
    if obstacle == 0:
        print('Le chemin est chill')
        return actuation, obstacle, segment
    
    x = [0,0,0,0,0,0,0]
    sensor_translation = 50

    actuation = speed0
    for i in range (6):
        x[i] = (prox[i] - (1000)) // sensor_translation
        if x[i]>0:
            actuation[0] = actuation[0] + x[i] * w_l[i]
            actuation[1] = actuation[1] + x[i] * w_r[i]
            
    actuation[actuation>200] = 200
    actuation[actuation<-200] = -200
       
    print('On evite l objet')
    print('Vitesse moteur gauche: ', actuation[0])
    print('Vitesse moteur droite: ', actuation[1])
    return actuation, obstacle