import gym
from gym.spaces.box import Box
import numpy as np
import gym_carla
import cv2

import numpy as np

gym.logger.set_level(40)

class CarlaWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat=4, image_size=64):
        super().__init__(env)
        self.env = env
        self.low = self.action_space.low
        self.high = self.action_space.high
        self.action_repeat = action_repeat
        self.image_size = image_size
        self.observation_space = Box(shape=(3, image_size, image_size), low=0, high=255,dtype=np.uint8)
        
    def _action(self, action):
        """Map action space [-1, 1] of model output to new action space
        [low_bound, high_bound].
        """
        action = self.low + (action + 1.0) * 0.5 * (self.high - self.low)
        action = np.clip(action, self.low, self.high)
        return action

    def step(self, action):
        action = self._action(action)
        total_reward = 0
        done = False
        truncated = False
        for _ in range(self.action_repeat):
            next_observation, reward, done, truncated, info  = self.env.step(action)
            total_reward += reward
            if done or truncated:
                done = True
                break

        state = self.parse_observation(next_observation)
        
        return state, total_reward, done, truncated, info
    
    def reset(self,seed=None, options=None):
        observation,info = self.env.reset()
        state = self.parse_observation(observation)
        return state, info
    
    def parse_observation(self,observation):

        camera  = cv2.resize(observation['camera'],
                             (self.image_size,self.image_size)).transpose((2, 0, 1))
        lidar   = cv2.resize(observation['lidar'],
                             (self.image_size,self.image_size)).transpose((2, 0, 1))
        birdeye = cv2.resize(observation['birdeye'],
                             (self.image_size,self.image_size)).transpose((2, 0, 1))
        
        state = {'camera': camera, 
                 'lidar':lidar,
                 'birdeye':birdeye}

        return state


def make_carla(env_name, params, action_repeat=4, image_size=64, seed =0):

    env =CarlaWrapper(
        gym.make('carla-bev', params=params),
        action_repeat=action_repeat,
        image_size=image_size)
    setattr(env, 'action_repeat', action_repeat)
    setattr(env, 'seed', seed)
    setattr(env,'_max_episode_steps',1000 / action_repeat)
    return env
