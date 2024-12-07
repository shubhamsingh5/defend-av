import numpy as np

def get_dynamic_steering_scale(lidar_data, safe_distance=1.5):
    """
    Calculate a dynamic steering scale based on LiDAR data.
    
    Parameters:
    - lidar_data (np.ndarray): Array of LiDAR distances.
    - safe_distance (float): Threshold for safe distance. Default is 1.5 meters.
    
    Returns:
    - float: Steering scale for dynamic adjustment.
    """
    # Clip LiDAR data to a maximum range
    lidar_distances = np.clip(lidar_data, 0, 10.0)  # Assuming 10 meters max range

    # Divide LiDAR data into zones: left, front, right
    num_zones = len(lidar_distances) // 3
    left_zone = np.mean(lidar_distances[:num_zones])
    front_zone = np.mean(lidar_distances[num_zones:2 * num_zones])
    right_zone = np.mean(lidar_distances[2 * num_zones:])

    # Adjust steering scale based on LiDAR readings
    if front_zone < safe_distance:  # Obstacle in front, allow sharper turns
        steering_scale = 0.02
    elif left_zone < safe_distance:  # Obstacle on left, prioritize right turn
        steering_scale = 1.0  # Scale for sharp right turn
    elif right_zone < safe_distance:  # Obstacle on right, prioritize left turn
        steering_scale = -1.0  # Scale for sharp left turn
    else:  # Path is clear
        steering_scale = 0.005  # Small turn for open path

    return steering_scale
