from gym.envs.registration import register

register(
    id='TargetReaching-v0',
    entry_point='learn_robot_manipulation.envs:TargetReachingEnv',
    max_episode_steps=200
    )

register(
    id='TargetReachingRRBot-v0',
    entry_point='learn_robot_manipulation.envs:TargetReachingRRBotEnv',
    max_episode_steps=200
    )

register(
    id='TargetReachingUR5-v0',
    entry_point='learn_robot_manipulation.envs:TargetReachingUR5Env',
    max_episode_steps=200
    )