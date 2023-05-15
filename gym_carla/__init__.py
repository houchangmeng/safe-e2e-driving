from gym.envs.registration import register

register(
    id='carla-bev',
    entry_point='gym_carla.envs:CarlaEnv',
)
