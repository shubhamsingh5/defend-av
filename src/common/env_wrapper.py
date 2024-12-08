import time
import cv2
import gymnasium as gym
import numpy as np

class RacecarWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.angle_start = -2.36
        self.angle_range = 4.71
        self.num_rays = 1080
        
        self.lidar_half_angle = np.pi / 2.0
        self.num_samples = 19
        self.target_angles = np.linspace(-self.lidar_half_angle, self.lidar_half_angle, self.num_samples)

        self.max_range = 15.0
        self.min_range = 0.25
        self.safe_distance_norm = 0.3

    def _process_observation(self, obs_dict):
        pose = obs_dict['pose']
        velocity = obs_dict['velocity']
        lidar = obs_dict['lidar']

        angles = np.linspace(self.angle_start, self.angle_start + self.angle_range, self.num_rays)

        key_indices = np.searchsorted(angles, self.target_angles)
        key_indices = np.clip(key_indices, 0, self.num_rays - 1)
        distances = lidar[key_indices]

        distances_clipped = np.clip(distances, self.min_range, self.max_range)
        distances_normalized = np.clip(distances_clipped / self.max_range, 0, 1)

        position = np.array(pose[:2]).flatten()     # (x, y)
        speed = np.array(velocity[:2]).flatten()    # (vx, vy)
        yaw_rate = np.array([velocity[5]]).flatten()# yaw rate
        orientation = np.array([pose[5]]).flatten() # orientation (yaw)

        position_normalized = np.clip(position / 100.0, -1.0, 1.0)
        speed_normalized = np.clip(speed / 14.0, -1.0, 1.0)
        yaw_rate_normalized = np.clip(yaw_rate / 6.0, -1.0, 1.0)
        orientation_normalized = np.clip(orientation / np.pi, -1.0, 1.0)

        state_components = [
            position_normalized,
            speed_normalized,
            yaw_rate_normalized,
            orientation_normalized,
            distances_normalized
        ]

        for i, component in enumerate(state_components):
            if component.ndim == 0:
                state_components[i] = np.array([component])

        final_state = np.concatenate(state_components).astype(np.float32)
        return final_state

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed_obs = self._process_observation(obs)
        return processed_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._process_observation(obs)
        # print(f"Processed observation: {processed_obs}")

        modified_reward = reward

        # Forward progress reward
        progress = info.get('progress', 0)
        if progress > 0.05:
            modified_reward += progress * 100.0


        # Penalty for collisions
        if info.get('wall_collision', False):
            eps = 0.01
            effective_progress = max(progress, eps)
            penalty = 100
            modified_reward -= penalty

        # LiDAR-based penalty: if too close to obstacles
        # front = central few rays, left = half on the left, right = half on the right
        lidar_data = processed_obs[-self.num_samples:]
        # print(f"LiDAR data: {lidar_data}")
        mid_idx = self.num_samples // 2
        left_segment = lidar_data[:mid_idx]
        right_segment = lidar_data[mid_idx+1:]
        front_segment = lidar_data[mid_idx:mid_idx+1]

        # Calculate average normalized distances
        avg_left = np.mean(left_segment)
        avg_front = np.mean(front_segment)
        avg_right = np.mean(right_segment)
        # print(f"Average distances: left={avg_left}, front={avg_front}, right={avg_right}")

        # If avg distance is less than safe_distance_norm, we penalize
        left_penalty = max(0, self.safe_distance_norm - avg_left) * 3
        front_penalty = max(0, self.safe_distance_norm - avg_front) * 1.5
        right_penalty = max(0, self.safe_distance_norm - avg_right) * 3
        # time.sleep(0.1)

        modified_reward -= (front_penalty + left_penalty + right_penalty)

        return processed_obs, modified_reward, terminated, truncated, info

    @property
    def observation_space(self):
        # position (2) + speed (2) + yaw_rate (1) + orientation (1) + lidar (num_samples)
        obs_dim = 2 + 2 + 1 + 1 + self.num_samples
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
