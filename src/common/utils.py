import numpy as np

def get_dynamic_steering_scale(lidar_data):
    """ lidar_data: [left90, left60, left30, front, right30, right60, right90]
    Here, higher values mean the path is clearer (farther from obstacles). """
    
    left = np.max(lidar_data[:3])    # Best left-side clearance
    front = lidar_data[3]            # Front clearance
    right = np.max(lidar_data[4:])   # Best right-side clearance

    # If the front has enough clearance, steer slightly toward the side that has more room
    safe_distance = 0.5
    if front > safe_distance:
        # If right side is more open, turn a bit right; if left side is more open, turn a bit left
        # Note: If right is bigger, that side is clearer, so maybe steer toward it
        steering_scale = 0.1 * (1 if right > left else -1)
    elif left > safe_distance:
        # Front not safe, but left is safer than right
        steering_scale = -0.2  # Turn left slightly (since left is clearer)
    elif right > safe_distance:
        # Front not safe, left not safe, but right is safer
        steering_scale = 0.2   # Turn right slightly
    else:
        # No good clear direction, maybe just turn slightly in one direction
        steering_scale = 0.1

    return steering_scale
