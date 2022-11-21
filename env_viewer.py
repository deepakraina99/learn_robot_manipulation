import gym, sys, argparse
import numpy as np
import learn_robot_manipulation

if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()

parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
parser.add_argument('--env', default='TargetReachingUR5-v0',
                    help='Environment to test (default: TargetReachingUR5-v0)')
args = parser.parse_args()

env = gym.make(args.env)

# for _ in range(1):
while True:
    done = False
    env.render()
    observation = env.reset()
    action = env.action_space.sample()
    # print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))   

    while not done:
        env.render()
        action = env.action_space.sample()
        print('action:', action)
        # action=[0,0,0,0,0,0]
        observation, reward, done, info = env.step(action)
        print('Reward:', reward)
        print('info:', info)
