import gymnasium as gym
import numpy as np

class RacecarWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def _process_observation(self, obs_dict):
        """Process the observation to a simpler representation"""
        # All these values are numpy arrays in the observation dict
        pose = obs_dict['pose']          # [x, y, z, roll, pitch, yaw]
        velocity = obs_dict['velocity']   # [x, y, z, roll, pitch, yaw]
        lidar = obs_dict['lidar']        # distances array

        # Extract relevant features
        position = pose[:2]               # x, y only
        speed = velocity[:2]              # x, y only
        orientation = pose[5:6]           # yaw only
        distances = lidar[::10]           # downsample lidar (every 10th reading)

        # Normalize values
        position_normalized = position / 100.0
        speed_normalized = speed / 20.0
        orientation_normalized = orientation / np.pi
        distances_normalized = np.clip(distances, 0, 20.0) / 20.0

        # Combine into single state vector
        state = np.concatenate([
            position_normalized,      # 2 values
            speed_normalized,         # 2 values
            orientation_normalized,   # 1 value
            distances_normalized      # reduced lidar values
        ]).astype(np.float32)

        return state

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed_obs = self._process_observation(obs)
        return processed_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._process_observation(obs)
        
        # Modify reward
        modified_reward = reward
        if info.get('progress', 0) > 0:
            modified_reward += info['progress'] * 0.1
        if info.get('wall_collision', False):
            modified_reward -= 1.0
            
        return processed_obs, modified_reward, terminated, truncated, info

    @property
    def observation_space(self):
        # 2 (position) + 2 (speed) + 1 (orientation) + 108 (lidar) = 113
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(113,),
            dtype=np.float32
        )