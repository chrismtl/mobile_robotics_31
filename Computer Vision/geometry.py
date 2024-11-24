import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def remove_close_points(corners, threshold=10):
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

def augmented_polygon(corners, offset):
    center_x, center_y = np.mean(corners, axis=0)
    augmented_corners = []
    for corner in corners:
        dist_y = corner[0] - center_x
        dist_x = corner[1] - center_y
        dist_center = np.sqrt(dist_x**2 + dist_y**2)
        if dist_center!=0: alpha = offset/dist_center
        else: alpha = 0
        delta_x = alpha * dist_x
        delta_y = alpha * dist_y
        
        augmented_corners.append([corner[0] + delta_x, corner[1] + delta_y])
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