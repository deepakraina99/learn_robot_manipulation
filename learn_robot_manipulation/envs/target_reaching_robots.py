from .target_reaching import TargetReachingEnv

class TargetReachingRRBotEnv(TargetReachingEnv):
    def __init__(self):
        super(TargetReachingRRBotEnv, self).__init__(robot_type = 'rrbot', action_robot_len=2, obs_robot_len=16)
class TargetReachingUR5Env(TargetReachingEnv):
    def __init__(self):
        super(TargetReachingUR5Env, self).__init__(robot_type = 'ur5', action_robot_len=6, obs_robot_len=16)