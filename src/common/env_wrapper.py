import time
import cv2
import gymnasium as gym
import numpy as np

class RacecarWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def _process_observation(self, obs_dict):
        pose = obs_dict['pose']
        velocity = obs_dict['velocity']
        lidar = obs_dict['lidar']

        # 1. Convert LiDAR to polar coordinates with higher resolution
        num_angles = lidar.shape[0]
        full_lidar_angle = np.pi * 270 / 180  # 270 degrees FOV
        angles = np.linspace(-full_lidar_angle / 2, full_lidar_angle / 2, num_angles)
        
        # Extract more granular distances instead of fixed key angles
        # Here, we sample distances at every 15-degree interval for richer input
        angular_step = np.pi / 12  # 15 degrees
        key_angles = np.arange(-full_lidar_angle / 2, full_lidar_angle / 2 + angular_step, angular_step)
        key_indices = np.clip(np.searchsorted(angles, key_angles), 0, num_angles - 1)
        distances = lidar[key_indices]

        # Weighted normalization to emphasize closer obstacles
        max_distance = 15.0  # Max LiDAR range
        distances_normalized = 1.0 - np.clip(distances / max_distance, 0, 1)
        
        # 2. Process and normalize other state components
        position = np.array(pose[:2]).flatten()  # Extract (x, y) position
        speed = np.array(velocity[:2]).flatten()  # Extract linear velocity in (x, y)
        yaw_rate = np.array([velocity[5]]).flatten()  # Extract yaw rate
        orientation = np.array([pose[5]]).flatten()  # Extract orientation (yaw angle)

        # Normalize components
        position_normalized = np.clip(position / 100.0, -1.0, 1.0)  # Position is normalized within a 50x50 area
        speed_normalized = np.clip(speed / 14.0, -1.0, 1.0)  # Max speed assumed to be 14 m/s
        yaw_rate_normalized = np.clip(yaw_rate / 6.0, -1.0, 1.0)  # Yaw rate normalized by a max of 3 rad/s
        orientation_normalized = np.clip(orientation / np.pi, -1.0, 1.0)  # Orientation normalized to [-1, 1]

        # 3. Concatenate all components into a single state vector
        state_components = [
            position_normalized,
            speed_normalized,
            yaw_rate_normalized,
            orientation_normalized,
            distances_normalized
        ]

        # Ensure all components are 1D arrays for concatenation
        for i, component in enumerate(state_components):
            if component.ndim == 0:
                state_components[i] = np.array([component])

        # Final state vector
        final_state = np.concatenate(state_components).astype(np.float32)
        
        return final_state


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed_obs = self._process_observation(obs)
        return processed_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._process_observation(obs)
        print (f"Processed observation: {processed_obs}")

        # Initialize modified reward with environment reward
        modified_reward = reward

        # Penalize large steering to encourage smoother actions
        modified_reward -= abs(action['steering']) * 0.1

        # Progress reward
        if info.get('progress', 0) > 0:
            modified_reward += info['progress'] * 1

        # Encourage small steering adjustments
        if abs(action['steering']) < 0.1:
            modified_reward += 0.5

        # Penalty for collisions with walls
        if info.get('wall_collision', False):
            modified_reward -= 5.0  # Stronger penalty for collisions
            
        # Encourage staying away from walls based on LiDAR
        lidar = obs['lidar']  # Assuming raw LiDAR data is in the observation
        print(f"LiDAR data: {lidar}")
        print(f"LiDAR shape: {lidar.shape}")
        num_angles = 1080  # Assuming LiDAR data has 1080 elements
        full_lidar_angle = np.pi * 270 / 180  # 270 degrees field of view
        angles = np.linspace(-full_lidar_angle / 2, full_lidar_angle / 2, num_angles)  # Angle distribution

        # Define the 6 angles you are interested in (e.g., -90, -60, -30, 0, 30, 60 degrees)
        target_angles = np.array([-90, -60, -30, 0, 30, 60, 90])  # Degrees
        target_angles_rad = np.radians(target_angles)  # Convert to radians

        # Find the corresponding indices in the LiDAR data
        target_indices = np.clip(np.searchsorted(angles, target_angles_rad), 0, num_angles - 1)

        # Print the LiDAR readings at the specified angles
        for angle, idx in zip(target_angles, target_indices):
            print(f"LiDAR distance at {angle} degrees: {lidar[idx]}")
        normalized_distances = np.clip(lidar / 15.0, 0, 1)  # Normalize distances
        num_zones = len(normalized_distances)  # Length of lidar array
        zone_size = num_zones // 12  # Divide by 3 for left, front, right zones
        print(f"Number of zones: {num_zones}, Zone size: {zone_size}")


        # Penalize being too close to walls on the right, left, and front
        left_zone = np.mean(normalized_distances[:zone_size])  # Left-side LiDAR readings
        front_zone = np.mean(normalized_distances[zone_size:2 * zone_size])  # Front LiDAR
        right_zone = np.mean(normalized_distances[2 * zone_size:])  # Right-side LiDAR readings
        print(f"Left zone: {left_zone}, Front zone: {front_zone}, Right zone: {right_zone}")

        # Reward for maintaining safe distance from walls
        safe_distance = 0.5  # Adjust threshold for safe distance (higher = safer)
        
        left_penalty = max(0, left_zone - safe_distance) * 2.0  # Strong penalty if too close on the left
        front_penalty = max(0, front_zone - safe_distance) * 3.0  # Stronger penalty if too close in front
        right_penalty = max(0, right_zone - safe_distance) * 2.0  # Strong penalty if too close on the right
        print(f"Left penalty: {left_penalty}, Front penalty: {front_penalty}, Right penalty: {right_penalty}")
        time.sleep(0.2)

        # Add penalties to the reward
        modified_reward -= left_penalty
        modified_reward -= front_penalty
        modified_reward -= right_penalty


        return processed_obs, modified_reward, terminated, truncated, info



    @property
    def observation_space(self):
        # 2 (position) + 2 (speed) + 1 (yaw_rate) + 1 (orientation) + 7 (lidar) = 13
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )