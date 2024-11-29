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

def find_intersection(p1, p2, p3, p4):
    # Line 1: p1 -> p2
    # Line 2: p3 -> p4
    A1 = p2[1] - p1[1]
    B1 = p1[0] - p2[0]
    C1 = A1 * p1[0] + B1 * p1[1]

    A2 = p4[1] - p3[1]
    B2 = p3[0] - p4[0]
    C2 = A2 * p3[0] + B2 * p3[1]

    determinant = A1 * B2 - A2 * B1

    if determinant == 0:
        return None

    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant

    return np.array([x, y], dtype=np.float32)

def augment_corners(frame, corners):
    augmented_corners = []

    # Iterate through each pair of consecutive corners
    for i in range(len(corners)):
        p1 = corners[i]
        p2 = corners[(i + 1) % len(corners)]  # Loop back to the start for the last segment
        p3 = corners[(i + 2) % len(corners)]  # Next segment's starting point
        
        # Calculate the first segment's offset
        segment_vector_1 = p2 - p1
        perpendicular_vector_1 = np.array([-segment_vector_1[1], segment_vector_1[0]])
        perpendicular_vector_1 = perpendicular_vector_1 / np.linalg.norm(perpendicular_vector_1)
        offset_p1 = p1 + perpendicular_vector_1 * ROBOT_RADIUS_PIXEL
        offset_p2 = p2 + perpendicular_vector_1 * ROBOT_RADIUS_PIXEL
        
        # Calculate the second segment's offset
        segment_vector_2 = p3 - p2
        perpendicular_vector_2 = np.array([-segment_vector_2[1], segment_vector_2[0]])
        perpendicular_vector_2 = perpendicular_vector_2 / np.linalg.norm(perpendicular_vector_2)
        offset_p3 = p2 + perpendicular_vector_2 * ROBOT_RADIUS_PIXEL
        offset_p4 = p3 + perpendicular_vector_2 * ROBOT_RADIUS_PIXEL
        
        # Find the intersection point of the two offset lines
        intersection = find_intersection(offset_p1, offset_p2, offset_p3, offset_p4)
        if intersection is not None:
            augmented_corners.append(intersection)


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