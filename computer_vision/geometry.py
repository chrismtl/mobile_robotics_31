import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from .constants import *
import cv2 as cv

def remove_close_points(corners, threshold=MIN_OBSTACLE_SEGMENT_LENGTH):
    # Create a new list
    clean_corners = corners.copy()
    
    # Find duplicates
    duplicates = []
    for i in range(len(clean_corners)):
        for j in range(len(clean_corners)):
            if i==j: continue
            dist = np.linalg.norm(clean_corners[i] - clean_corners[j])
            if (dist<threshold) and (not((i in duplicates) or (j in duplicates))):
                duplicates.append(j)
                
    # Remove duplicates
    duplicates = sorted(duplicates)
    mask = np.ones(len(clean_corners), dtype=bool)
    mask[duplicates] = False
    clean_corners = clean_corners[mask]

    return clean_corners

def find_peak(corners, i):
    # Compute left and right corners
    corner = np.array(corners[i])
    left_corner  = np.array(corners[(i-1)%len(corners)])
    right_corner = np.array(corners[(i+1)%len(corners)])
    # Compute left and right vectors
    left_vector = (corner - left_corner)
    right_vector = (corner - right_corner)
    # Normalize left and right vectors
    left_vector = left_vector/np.linalg.norm(left_vector)
    right_vector = right_vector/np.linalg.norm(right_vector)
    # Sum them to get the corner vector
    corner_vector = left_vector + right_vector
    # Scale the vector to have a norm of ROBOT_RADIUS_PIXEL
    scaler = ROBOT_RADIUS_PIXEL/np.linalg.norm(corner_vector)
    corner_vector = scaler*corner_vector
    # Return the vector added to the corner
    return corner + corner_vector
    
def augment_corners(corners):
    augmented_corners = []

    # Iterate through each pair of consecutive corners
    for i in range(len(corners)):
        p1 = corners[i]
        p2 = corners[(i + 1) % len(corners)]
        
        segment_vector = p2 - p1
        perpendicular_vector = np.array([-segment_vector[1], segment_vector[0]])
        if np.linalg.norm(perpendicular_vector) < EPSILON: return corners
        perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
        
        # Calculate offset points
        offset_peak = find_peak(corners,i)
        offset_p1 = (p1 + perpendicular_vector * ROBOT_RADIUS_PIXEL).astype(np.int32)
        offset_p2 = (p2 + perpendicular_vector * ROBOT_RADIUS_PIXEL).astype(np.int32)

        augmented_corners.append(offset_peak)
        augmented_corners.append(offset_p1)
        augmented_corners.append(offset_p2)

    return np.round(augmented_corners).astype(np.int32)

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
        
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
        
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
        
    return roll_x, pitch_y, yaw_z # in radians

def get_rotations(rotation_matrix):
    r = R.from_matrix(rotation_matrix[0:3, 0:3])
    quat = r.as_quat()

    # Quaternion format     
    transform_rotation_x = quat[0] 
    transform_rotation_y = quat[1] 
    transform_rotation_z = quat[2] 
    transform_rotation_w = quat[3] 

    # Euler angle format in radians
    roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, 
                                                transform_rotation_y, 
                                                transform_rotation_z, 
                                                transform_rotation_w)
    
    roll_x = round(math.degrees(roll_x), 2)
    pitch_y = round(math.degrees(pitch_y), 2)
    yaw_z = round(math.degrees(yaw_z), 2)

    return (roll_x,pitch_y,yaw_z)

def get_rotations_chat(matrix):
    sy = math.sqrt(matrix[0, 0]**2 + matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(matrix[2, 1], matrix[2, 2])
        y = math.atan2(-matrix[2, 0], sy)
        z = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        x = math.atan2(-matrix[1, 2], matrix[1, 1])
        y = math.atan2(-matrix[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])