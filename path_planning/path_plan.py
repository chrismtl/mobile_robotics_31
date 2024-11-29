# %% [markdown]
# # Path Planning

# %%

# %%
import numpy as np
from shapely.geometry import LineString, Polygon,Point
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from heapq import heappush, heappop
from numpy import array
from numpy.linalg import norm
import math
from math import atan


# %%
def heuristic(p1, p2):
    # Implement the Manhattan distance heuristic
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# %%
def path_functions(shortest_path):
    """
    Find the coefficient of the path slopes ( y = alpha*x + beta)

    input: 
        shortest_path: M X 2 (M = # of nodes in the shortest path), A list 
            of corner indices representing the shortest path from start to goal.

    output: 
        nodes_slopes = M x 5 (M = number of nodes in the shortest path), where 
        each row represents the node's coordinates along with the 
        alpha and beta coefficients, the last colomn represents the direction 
        of the slope.
    """

    M = shortest_path.shape[0]
    nodes_slopes = np.zeros((M, 5))
    for i in range(M-1):
        nodes_slopes[i,0] = shortest_path[i,0]
        nodes_slopes[i,1] = shortest_path[i,1]
        #check if the slope is not vertical
        if i != (M-1):
            if shortest_path[i+1,0] == shortest_path[i,0]:
                #means the segment is vertical
                nodes_slopes[i,2] = 0
                nodes_slopes[i,3] = 0
                nodes_slopes[i,4] = 0

            else :
                #find alpha
                nodes_slopes[i,2] = (shortest_path[i+1,1]-shortest_path[i,1])/(shortest_path[i+1,0]-shortest_path[i,0])
                #find beta
                nodes_slopes[i,3] = shortest_path[i+1,1]-shortest_path[i+1,0]*nodes_slopes[i,2]
                #find direction
                if shortest_path[i+1,0] > shortest_path[i,0] :
                    nodes_slopes[i,4] = 1
                else:
                    nodes_slopes[i,4] = -1
            
    return nodes_slopes


# %%
def a_star_search(points,ex_path):
    """
    A* 

    input:
        start
        end 
        points: N x 2 array (N = (# of corners) ), where:
            - The remaining rows contain the coordinates of the extended corners.
        ex_path: N X N, is equal to 1 if two corners are directly connected 
              by a line that does not cross any obstacle, and 0 otherwise.

    output: 
        shortest_path: M X 2 (M = # of corners in the shortest path), A list 
            of corner indices representing the shortest path from start to goal.
    """

    
    # Initialize the open set as a priority queue and add the start node
    open_set = []
    start_index = 0
    start = points[0,:]
    goal_index = 1
    goal = points[1,:]

    N = points.shape[0]
    distance_matrix = np.zeros((N, N))
    for i in range(N):
         for j in range(N):
              #calculate the distance between two corners
              distance_matrix[i,j] = norm(points[i,:]-points[j,])
    

    heappush(open_set, (heuristic(start, goal), 0, start_index))  # (f_cost, g_cost, position)

    # Initialize the came_from dictionary
    came_from = {}
    # Initialize g_costs dictionary with default value of infinity and set g_costs[start] = 0
    g_costs = {start_index: 0}
    # Initialize the explored set
    explored = set()
    operation_count = 0

    while open_set:
        # Pop the node with the lowest f_cost from the open set
        current_f_cost, current_g_cost, current_index = heappop(open_set)
        # Add the current node to the explored set
        explored.add(current_index)

        # For directly reconstruct path
        if current_index == goal_index:
            break

        # Get the neighbors of the current node 
        index_vector = ex_path[current_index,:]
        neighbors_index = np.nonzero(index_vector)[0]

        for index in neighbors_index:
             if  index not in explored:
                    # Calculate tentative_g_cost
                    tentative_g_cost = current_g_cost + distance_matrix[current_index,index]

                    # If this path to neighbor is better than any previous one
                    if index not in g_costs or tentative_g_cost < g_costs[index]:
                        # Update came_from, g_costs, and f_cost
                        came_from[index] = current_index
                        g_costs[index] = tentative_g_cost
                        f_cost = tentative_g_cost + heuristic(points[index,:], goal)
                        
                        # Add neighbor to open set
                        heappush(open_set, (f_cost, tentative_g_cost, index))
                        operation_count += 1

    # Reconstruct path
    if current_index == goal_index:
        path = []
        while current_index in came_from:
            path.append(points[current_index,:])
            current_index = came_from[current_index]
        path.append(start)
        np_path = np.array(path)
        return np_path[::-1]
    else:
        # If we reach here, no path was found
        return None
    

# %%
def path_direction(coordinates, nodes_slopes, segment_index):
    """
    Find the right motor speed so the robot follows the rigth seglent

    input: 
        coordinates: 1 x 3 array, where:
            - The first row represents the mean of x, y and theta coordinates.
        nodes_slopes = M x 5 (M = number of nodes in the shortest path), where 
        each row represents the node's coordinates along with the 
        alpha and beta coefficients, the last colomn represents the direction 
        of the slope.

    output: 
        speed: 1 X 2 , contains the speed value for the left and right motor respectively. 
        segment_index: scalar indicationg the index of the segment the robot is on.
        end: scalar, if equal to 1 we're at the final destination
    """

    M = nodes_slopes.shape[0]
    speed = np.zeros(2)
    y_mean = coordinates[0,1]
    x_mean = coordinates[0,1]
    theta_mean = -1*coordinates[0,2]
    #theta_mean = math.radians(theta_mean) #si angle en degres

    tolerance_norm = 1
    Param1 = 100 #depend de l'unité
    Param2 = 50 
    end = 0

    #check if we're close to the end of the segment
    distance_segm = ((nodes_slopes[segment_index+1,1]-y_mean)**2 + (nodes_slopes[segment_index+1,0]-x_mean)**2)**0.5

    if distance_segm < tolerance_norm:
        #means we're at the end of the segment
        segment_index += 1

        #check if we're at the final distination
        if segment_index == (M-1):
            end = 1
            speed[:] = [0,0]
            return speed,segment_index,end


    #find the angle of the slope and set the speed
    angle_err = angle_error(x_mean,y_mean, theta_mean, nodes_slopes[segment_index+1,0], nodes_slopes[segment_index+1,1])
    speed[0] = distance_segm*Param1 + angle_err*Param2
    speed[1] = distance_segm*Param1 - angle_err*Param2


# %%
def angle_error(x_rob,y_rob, theta_rob, x_fin, y_fin):
    """
    Find on which slope the robot is so we can activate the motors 

    input: 
        x_rob : x coordinates of the robot
        y_rob : y coordinates of the robot
        theta_rob : theta coordinates of the robot
        x_fin : x coordinates of the end of the segment
        y_fin : y coordinates of the end of the segment

    output: 
        angle_err: scalar in radian corresponding to the difference between the orientation
                   of the robot and the slope on which it should be.
    """

    angle_slope = np.arctan2(y_fin-y_rob, x_fin-x_rob)
    angle_diff = theta_rob - angle_slope
    if abs(angle_diff) < math.pi:
        angle_err = angle_diff
        return angle_err
    else: 
        value = 2*math.pi-abs(angle_diff)
        sign = -1* math.copysign(1,angle_diff)
        angle_err = sign*value
        return angle_err
    
# err = angle_error(0,0, (-1*math.pi)/2, 0, 1)
# print(err)

# %%

def compute_visibility_matrix(start,end,obstacles):
    """
    Compute a visibility matrix

    input:
        start point
        end poitn
        obstacles: List of obstacles, each as a list of extended 
                   corner coordinates [[(x1, y1), ...], ...]
    
    output:
        Visibility_mat : (N x N) Visibility_mat(i,j) is equal to 1 if a path exist between the ith corner
                            and the jth corner, if not  it's equal to 0.
        corners
    """
    # Convert start and end points to single-point obstacles
    start_obstacle = [start]
    end_obstacle = [end]

    # Add start and end points to the obstacles list
    obstacles = [start_obstacle, end_obstacle] + obstacles

    # Flatten the list of obstacles to get all corners
    corners = [corner for obstacle in obstacles for corner in obstacle]
    corners = np.array(corners)
    N = len(corners)
    matrix = np.ones((N, N), dtype=int)

    # Convert obstacles to polygons

    obstacle_polygons = []
    for obs in obstacles:
        if len(obs) >= 3:  # A valid polygon requires at least 3 points
            obstacle_polygons.append(Polygon(obs))
    obstacle_polygons = np.array(obstacle_polygons)

    # Map corners to their respective obstacle indices
    corner_to_obstacle = {}
    for obs_idx, obstacle in enumerate(obstacles):
        for corner in obstacle:
            corner_to_obstacle[tuple(corner)] = (obs_idx, obstacle)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            line = LineString([corners[i], corners[j]])
            # Check if corners[i] and corners[j] belong to the same obstacle
            same_obstacle = (
                tuple(corners[i]) in corner_to_obstacle 
                and tuple(corners[j]) in corner_to_obstacle 
                and corner_to_obstacle[tuple(corners[i])][0] == corner_to_obstacle[tuple(corners[j])][0]
            )

            # If they are from the same obstacle
            if same_obstacle:
                # Retrieve the obstacle corners
                obstacle_corners = corner_to_obstacle[tuple(corners[i])][1]
                # Check if they are adjacent corners (only adjacent corners are visible)
                if not are_adjacent_corners(corners[i], corners[j], obstacle_corners):
                    matrix[i, j] = 0

            else:
                for poly in obstacle_polygons:
                    if line.intersects(poly) and not line.touches(poly.boundary):
                        matrix[i, j] = 0
                        break


    return matrix, corners

def are_adjacent_corners(corner1, corner2, obstacle_corners):
    """
    Check if two corners are adjacent in the obstacle.
    
    :param corner1: First corner (x, y)
    :param corner2: Second corner (x, y)
    :param obstacle_corners: List of corners of the obstacle
    :return: True if the corners are adjacent, False otherwise
    """
    n = len(obstacle_corners)
    for i in range(n):
        if (np.array_equal(obstacle_corners[i], corner1) and 
           (np.array_equal(obstacle_corners[(i + 1) % n], corner2) or 
           np.array_equal(obstacle_corners[(i - 1) % n], corner2))):
            return True
    return False

# %%

def possible_lignes(ex_path, corners):

    N = ex_path.shape[0]
    lignes = []
    for i in range(N):
        for j in range(N):
            if ex_path[i,j] == 1:
                lignes.append(np.concatenate((corners[i, :], corners[j, :])))
    
    lignes = np.array(lignes)
    return lignes




# %% test

# obstacles = [
#     [(0, 0), (1, 0), (1, 1), (0, 1)],  
#     [(2, 2), (3, 2), (3, 3), (2, 3)] ,
# ]
# start= (0,0)
# end = (4,4)
# matrix,corners = compute_visibility_matrix(start,end,obstacles)
# opt_path = a_star_search(start,end,corners,matrix)
# print(opt_path)


 

# %% test 

# points = np.array([[0, 0], [16, 0], [4, 0] , [8, 8], [4, 4]])
# ex_path =  np.array([[1,0 ,1, 1, 1], [0,1 ,0, 1, 1], [1,0 ,1, 1, 1], [1,1 ,1, 1, 1], [1,1 ,1, 1, 1]])
# path= a_star_search(points,ex_path)
# nodes = path_functions(path)
# coordinates = np.array([[2, 0, 30], [0.2, 0.2,0]])
# segment_index = 0
# speed,segment_index2,end = path_direction(coordinates, nodes, segment_index)


# corners = [(1, 1), (2, 3), (4, 4), (5, 2)]  
# obstacles = [[(3, 1), (3, 2), (4, 2), (4, 1)]] 

# visibility_matrix = compute_visibility_matrix(corners, obstacles)
# print(visibility_matrix)




