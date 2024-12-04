import numpy as np
from scipy.spatial.transform import Rotation as R
from constants import *

def remove_close_points(corners, threshold=MIN_OBSTACLE_SEGMENT_LENGTH):
    """
    Removes points in a list that are closer than a given threshold

    Args:
        corners (list): list of points
        threshold (float, optional): Minimal distance between points. Defaults to MIN_OBSTACLE_SEGMENT_LENGTH.

    Returns:
        clean_corners (list): list with the close points removed
    """
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

def on_points(corners, detected, points, limits):
    """
    From a list of points (corners) tells if a point is on a target point (closer than given limit).

    Args:
        corners (list,list,int): Treated points
        detected (list,bool): gives info on whether the target is detected or not
        points (list,list,int): target points
        limits (list,float): minimal distance between a treated point and a target point

    Returns:
        True: the point is on one of the targets
        False: the point is not on one of the target
    """
    if len(points)==len(limits)==len(detected):
        for point,limit,detect in zip(points,limits,detected):
            if not detect: continue
            dist_to_point = np.linalg.norm(np.array(point)-np.mean(corners,axis=0).astype(int))
            if dist_to_point<limit:
                return True
        return False
    if P_VISION: print("ERROR: Size of arguments on on_points")
    return False

def find_peak(corners, i):
    """
    From a given set of points (polygon) gives the augmented point of corner i,
    defined by the offset point such that the distance between the offset point and
    the corner is the robot radius. (used to augment the obstacles)

    Args:
        corners (list,list,int): list of 2D points
        i (int): index of the point in corners to treat

    Returns:
        offset point
    """
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
    """
    From a list of points (corners) return a new list of augmented points,
    which are all extended by a distance of *ROBOT_RADIUS_PIXEL*

    Args:
        corners (list,int): list of points

    Returns:
        augmented_corners: list of augmented points
    """
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
    roll_x = np.arctan2(t0, t1)
        
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)
        
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
        
    return roll_x, pitch_y, yaw_z # in radians

def get_rotations(rotation_matrix):
    """
    Compute the euler angle for a given rotation matrix

    Args:
        rotation_matrix (matrix)

    Returns:
        (roll_x,pitch_y,yaw_z): Euler angles
    """
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
    
    roll_x = round(np.degrees(roll_x), 2)
    pitch_y = round(np.degrees(pitch_y), 2)
    yaw_z = round(np.degrees(yaw_z), 2)

    return (roll_x,pitch_y,yaw_z)