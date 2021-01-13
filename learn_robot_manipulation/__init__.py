from gym.envs.registration import register

register(
    id='TargetReaching-v0',
    entry_point='learn_robot_manipulation.envs:TargetReachingEnv',
    )
