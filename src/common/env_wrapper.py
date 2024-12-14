import time
import cv2
import gymnasium as gym
import numpy as np


class RacecarWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.angle_start = -2.36
        self.angle_range = 4.71
        self.num_rays = 1080
        self.lidar_half_angle = np.pi / 2.0
        self.num_samples = 19
        self.target_angles = np.linspace(
            -self.lidar_half_angle, self.lidar_half_angle, self.num_samples
        )
        self.max_range = 15.0
        self.min_range = 0.25
        self.safe_distance_front = 0.15
        self.safe_distance_side = 0.08

        # Explicitly define action space for SB3
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32,
        )

        # Define observation space explicitly
        obs_dim = (
            2 + 2 + 1 + 1 + self.num_samples
        )  # position (2) + speed (2) + yaw_rate (1) + orientation (1) + lidar (num_samples)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _process_observation(self, obs_dict):
        pose = obs_dict["pose"]
        velocity = obs_dict["velocity"]
        lidar = obs_dict["lidar"]

        angles = np.linspace(
            self.angle_start, self.angle_start + self.angle_range, self.num_rays
        )
        key_indices = np.searchsorted(angles, self.target_angles)
        key_indices = np.clip(key_indices, 0, self.num_rays - 1)
        distances = lidar[key_indices]

        distances_clipped = np.clip(distances, self.min_range, self.max_range)
        distances_normalized = np.clip(distances_clipped / self.max_range, 0, 1)

        position = np.array(pose[:2]).flatten()
        speed = np.array(velocity[:2]).flatten()
        yaw_rate = np.array([velocity[5]]).flatten()
        orientation = np.array([pose[5]]).flatten()

        position_normalized = np.clip(position / 10.0, -1.0, 1.0)
        speed_normalized = np.clip(speed / 14.0, -1.0, 1.0)
        yaw_rate_normalized = np.clip(yaw_rate / 6.0, -1.0, 1.0)
        orientation_normalized = np.clip(orientation / np.pi, -1.0, 1.0)

        state_components = [
            position_normalized,
            speed_normalized,
            yaw_rate_normalized,
            orientation_normalized,
            distances_normalized,
        ]

        for i, component in enumerate(state_components):
            if component.ndim == 0:
                state_components[i] = np.array([component])

        final_state = np.concatenate(state_components).astype(np.float32)
        return final_state

    def step(self, action):
        # Convert numpy array action to dict
        action_dict = {"motor": float(action[0]), "steering": float(action[1])}

        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        processed_obs = self._process_observation(obs)

        modified_reward = reward
        progress = info.get("progress", 0)
        if progress > 0:
            modified_reward += progress

        # LiDAR-based penalties
        lidar_data = processed_obs[-self.num_samples :]
        mid_idx = self.num_samples // 2
        left_segment = lidar_data[:mid_idx]
        right_segment = lidar_data[mid_idx + 1 :]
        front_segment = lidar_data[mid_idx - 1 : mid_idx + 2]

        avg_left = np.mean(left_segment)
        avg_front = np.mean(front_segment)
        avg_right = np.mean(right_segment)

        front_penalty = 0.0
        motor = action_dict["motor"]
        steering = action_dict["steering"]

        if avg_front > self.safe_distance_front:
            modified_reward += motor * 0.1  # Encourage speed when safe

        # Stronger penalty for high speed near walls
        if avg_front < self.safe_distance_front and motor > 0.3:
            front_penalty += motor * (1 - avg_front) * 0.1
            
        # Penalize going too slowly generally
        if motor < 0.1:
            modified_reward -= 0.05
            
        # reward turning in correct direction
        if avg_front < self.safe_distance_front:
            # reward sharp turn
            if abs(steering) > 0.7:
                modified_reward += 0.1
                
            left_more_open = avg_left > avg_right
            right_more_open = avg_right > avg_left
            
            if left_more_open:
                turn_reward = -steering
                front_penalty -= max(
                    0, turn_reward * 0.2
                )

            elif right_more_open:
                turn_reward = steering
                front_penalty -= max(0, turn_reward * 0.2)

        modified_reward -= front_penalty

        return processed_obs, modified_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed_obs = self._process_observation(obs)
        return processed_obs, info
