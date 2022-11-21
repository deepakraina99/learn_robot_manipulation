from .target_reaching import TargetReachingEnv

class TargetReachingRRBotEnv(TargetReachingEnv):
    def __init__(self):
        ## Joint action
        # super(TargetReachingRRBotEnv, self).__init__(robot_type = 'rrbot', action_robot_len=2, obs_robot_len=16, reward_type=2)
        ## Task action
        super(TargetReachingRRBotEnv, self).__init__(robot_type = 'rrbot', action_robot_len=6, obs_robot_len=16, reward_type=2, action_type='task_space')

class TargetReachingUR5Env(TargetReachingEnv):
    def __init__(self):
        ## Joint action
        # super(TargetReachingUR5Env, self).__init__(robot_type = 'ur5', action_robot_len=3, obs_robot_len=16, reward_type=3, target_size=0.1)
        ## Task action
        # super(TargetReachingUR5Env, self).__init__(robot_type='ur5', action_robot_len=6, obs_robot_len=16, reward_type=3, action_type='task_space', target_size=0.1)
        ## Task action with orientation
        super(TargetReachingUR5Env, self).__init__(robot_type = 'ur5', action_robot_len=6, obs_robot_len=16, orientation_ctrl=True, reward_type=4, action_type='task_space', target_size=0.1)
        
        # super(TargetReachingUR5Env, self).__init__(robot_type = 'ur5', action_robot_len=3, obs_robot_len=16, reward_type=2, max_speed=1.0, target_size=0.1)
